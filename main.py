# ============================================================
# QCar Taxi Mission (Node-Based Routing - RH Traffic)
# - Keeps your Frenet NMPC intact
# - Adds pickup/dropoff/hub logic using SDCSRoadMap node routing
# Mission:
#   Hub (10) -> Pickup (20) -> stop -> Dropoff (9) -> stop -> Hub (10) -> repeat
# ============================================================

#region : Imports
import time
import signal
import numpy as np
import cv2
import pyqtgraph as pg

from dataclasses import dataclass
from enum import Enum
from threading import Thread, Lock

import pal.resources.images as images
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from hal.products.mats import SDCSRoadMap
from hal.content.qcar_functions import QCarEKF
from hal.utilities.control import PID

# Your existing helpers
from Path2D import Path2D
from FrenetNonLinearMPCController import FrenetNonlinearMPCController



# ============================================================
# ======================== CONFIG =============================
# ============================================================

tf = 300
startDelay = 1
controllerUpdateRate = 30  # Hz
scopeUpdateRate = 10       # Hz

enableVehicleControl = True
v_ref = 1.0  # m/s

Ts = 1.0 / controllerUpdateRate

# ---------- RH Traffic Nodes ----------
HUB_NODE = 10
PICKUP_NODE = 20
DROPOFF_NODE = 9

# ---------- Mission behavior ----------
STOP_DURATION = 2.0   # seconds stopped at pickup/dropoff
GOAL_TOL = 0.18       # meters (arrival radius around final waypoint)

# ---------- Lane bounds ----------
W = 0.20  # lane width [m]
b = 0.01  # buffer [m]
ey_min = -(W/2 - b)
ey_max = +(W/2 - b)

# ---------- Vehicle geometry ----------
lr = lf = 0.128  # meters

# ---------- Speed profile ----------
a_lat_max = 1.5
a_accel = 1.5

# ---------- MPC weights ----------
alpha = 20
Q = alpha * np.array([0.001, 3, 2.5, 0.001])
beta = 2
R = beta * np.array([0.5, 1])
gamma = 20
Sdu = gamma * np.array([1, 0.01])
weights = {"Q": Q, "R": R, "Sdu": Sdu}


# ============================================================
# ======================== FSM ================================
# ============================================================

class TaxiState(Enum):
    GO_TO_PICKUP = 0
    PICKUP_STOP = 1
    GO_TO_DROPOFF = 2
    DROPOFF_STOP = 3
    RETURN_TO_HUB = 4


class RouteLeg(Enum):
    HUB_TO_PICKUP = 0
    PICKUP_TO_DROPOFF = 1
    DROPOFF_TO_HUB = 2


@dataclass
class LegData:
    seq: np.ndarray
    path: Path2D
    controller: FrenetNonlinearMPCController


# ============================================================
# ======================== UTILS ==============================
# ============================================================

lock = Lock()
KILL_THREAD = False

def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True

signal.signal(signal.SIGINT, sig_handler)


def reached_node_goal(x, y, waypointSequence, tol=GOAL_TOL):
    """Goal is the LAST waypoint of the current routed path."""
    gx = float(waypointSequence[0, -1])
    gy = float(waypointSequence[1, -1])
    return (x - gx)**2 + (y - gy)**2 <= tol**2


def rebuild_path(start_node, end_node):
    """
    Builds a new roadmap route segment and Path2D with speed profile.
    Returns:
        waypointSequence: shape (2, N)
        path: Path2D instance
    """
    seq = roadmap.generate_path([start_node, end_node])  # (2,N)
    new_path = Path2D(seq.T, Ts)                         # Path2D expects (N,2)
    new_path.compute_speed_profile(
        a_lat_max=a_lat_max,
        v_min=0.0,
        v_max=v_ref,
        a_accel=a_accel,
        a_decel=a_accel,
    )
    return seq, new_path


def create_controller(path):
    """Recreate NMPC whenever path changes."""
    return FrenetNonlinearMPCController(
        path=path,
        Ts=Ts,
        L=lf + lr,
        N=30,
        ey_max=ey_max,
        delta_max=np.pi / 6,
        ddelta_max=2.0,
        a_min=-a_accel,
        a_max=a_accel,
        da_max=2.0,
        v_min=0.001,
        v_max=v_ref,
        w_ey=weights["Q"][1],
        w_epsi=weights["Q"][2],
        w_v=weights["Q"][3],
        w_delta=weights["R"][0],
        w_a=weights["R"][1],
        w_ddelta=weights["Sdu"][0],
        w_da=weights["Sdu"][1],
        use_curv_speed_ref=True,
        v_ref_base=v_ref,
        kappa_speed_gain=60.0,
    )


def reset_path_projection_state(path_obj):
    path_obj.last_i = 0
    path_obj.last_s = 0.0


def build_leg_data(start_node, end_node):
    seq, path = rebuild_path(start_node, end_node)
    controller = create_controller(path)
    return LegData(seq=seq, path=path, controller=controller)


def build_all_legs():
    return {
        RouteLeg.HUB_TO_PICKUP: build_leg_data(HUB_NODE, PICKUP_NODE),
        RouteLeg.PICKUP_TO_DROPOFF: build_leg_data(PICKUP_NODE, DROPOFF_NODE),
        RouteLeg.DROPOFF_TO_HUB: build_leg_data(DROPOFF_NODE, HUB_NODE),
    }


# ============================================================
# ===================== INITIAL SETUP =========================
# ============================================================

roadmap = SDCSRoadMap()

# Start at HUB node pose
initialPose = roadmap.get_node_pose(HUB_NODE).squeeze()  # [x,y,theta]
x_hat = initialPose
t_hat = 0.0

print("Initial pose:", x_hat)

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    hQCar = qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate = 'y' in input('do you want to recalibrate?(y/n)')

# Used to enable safe keyboard triggered shutdown
def _flush_gps(gps_obj):
    while (not KILL_THREAD) and (gps_obj.readGPS() or gps_obj.readLidar()):
        pass

gps = QCarGPS(initialPose=initialPose, calibrate=calibrate)
_flush_gps(gps)


# ============================================================
# ===================== CONTROL LOOP ==========================
# ============================================================

def controlLoop():
    global KILL_THREAD, x_hat, t_hat

    # ---- Control vars ----
    u = 0.0
    v_cmd = 0.0
    delta = 0.0
    delta_ref = 0.0
    kappa_ref = 0.0

    ey = 0.0
    epsi = 0.0

    last_print_t = 0.0
    last_scope_t = 0.0
    mpc_ms_avg = 0.0

    # ---- EKF ----
    with lock:
        ekf = QCarEKF(x_0=x_hat)

    # ---- Low-level speed control ----
    pid_controller = PID(Kp=0.2, Ki=1.0, Kd=0.0, uLimits=(0.0, v_ref))

    # ---- QCar IO ----
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)

    # ---- Mission ----
    mission_state = TaxiState.GO_TO_PICKUP
    stop_timer = 0.0

    leg_cache = build_all_legs()
    current_leg = RouteLeg.HUB_TO_PICKUP
    leg = leg_cache[current_leg]
    reset_path_projection_state(leg.path)

    waypointSequence = leg.seq
    path = leg.path
    f_controller = leg.controller

    # Update referencePath plot (if UI exists)
    if "referencePath" in globals():
        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

    with qcar:
        t0 = time.time()
        t = 0.0

        while (t < tf + startDelay) and (not KILL_THREAD):

            tp = t
            t = time.time() - t0
            dt = max(t - tp, 1e-3)

            # --- Read sensors ---
            qcar.read()
            v = float(qcar.motorTach)

            # EKF update uses last-applied delta
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([v, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([v, delta], Ts, None, qcar.gyroscope[2])

            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = float(ekf.x_hat[0, 0])
            y = float(ekf.x_hat[1, 0])
            th = float(ekf.x_hat[2, 0])

            # ======================================================
            # ====================== MISSION FSM ====================
            # ======================================================

            if mission_state == TaxiState.GO_TO_PICKUP:
                if reached_node_goal(x, y, waypointSequence):
                    mission_state = TaxiState.PICKUP_STOP
                    stop_timer = 0.0

            elif mission_state == TaxiState.PICKUP_STOP:
                qcar.write(0.0, 0.0)
                stop_timer += dt

                if stop_timer >= STOP_DURATION:
                    mission_state = TaxiState.GO_TO_DROPOFF
                    current_leg = RouteLeg.PICKUP_TO_DROPOFF
                    leg = leg_cache[current_leg]
                    reset_path_projection_state(leg.path)

                    waypointSequence = leg.seq
                    path = leg.path
                    f_controller = leg.controller
                    pid_controller.reset()

                    if "referencePath" in globals():
                        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

                    continue

            elif mission_state == TaxiState.GO_TO_DROPOFF:
                if reached_node_goal(x, y, waypointSequence):
                    mission_state = TaxiState.DROPOFF_STOP
                    stop_timer = 0.0

            elif mission_state == TaxiState.DROPOFF_STOP:
                qcar.write(0.0, 0.0)
                stop_timer += dt

                if stop_timer >= STOP_DURATION:
                    mission_state = TaxiState.RETURN_TO_HUB
                    current_leg = RouteLeg.DROPOFF_TO_HUB
                    leg = leg_cache[current_leg]
                    reset_path_projection_state(leg.path)

                    waypointSequence = leg.seq
                    path = leg.path
                    f_controller = leg.controller
                    pid_controller.reset()

                    if "referencePath" in globals():
                        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

                    continue

            elif mission_state == TaxiState.RETURN_TO_HUB:
                if reached_node_goal(x, y, waypointSequence):
                    # Loop again
                    mission_state = TaxiState.GO_TO_PICKUP
                    current_leg = RouteLeg.HUB_TO_PICKUP
                    leg = leg_cache[current_leg]
                    reset_path_projection_state(leg.path)

                    waypointSequence = leg.seq
                    path = leg.path
                    f_controller = leg.controller
                    pid_controller.reset()

                    if "referencePath" in globals():
                        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

                    continue

            # ======================================================
            # ======================== SAFETY =======================
            # ======================================================

            if abs(ey) > (abs(ey_max) + 0.40):
                print(f"STOP: |ey|={abs(ey):.2f} exceeds ey_max+0.10={abs(ey_max)+0.10:.2f}")
                qcar.write(0.0, 0.0)
                KILL_THREAD = True
                return

            # ======================================================
            # ====================== NMPC CONTROL ===================
            # ======================================================

            mpc_dt_ms = 0.0

            if t < startDelay or (not enableVehicleControl):
                u = 0.0
                delta = 0.0
            else:
                mpc_t0 = time.time()
                delta_cmd, v_cmd, kappa_ref, ey, epsi = f_controller.compute_control(
                    x_world=np.array([x, y]),
                    psi=th,
                    v_meas=v,
                )
                mpc_dt_ms = (time.time() - mpc_t0) * 1000.0
                mpc_ms_avg = 0.9 * mpc_ms_avg + 0.1 * mpc_dt_ms

                delta = float(delta_cmd)
                delta_ref = float(np.arctan((lf + lr) * kappa_ref))
                u = float(pid_controller.update(v_cmd, v, dt))

            qcar.write(u, delta)

            # ======================================================
            # ======================== SCOPE ========================
            # ======================================================

            if (
                t > startDelay
                and "steeringScope" in globals()
                and (t - last_scope_t) >= (1.0 / scopeUpdateRate)
            ):
                t_plot = t - startDelay
                steeringScope.axes[0].sample(t_plot, [ey])
                steeringScope.axes[1].sample(t_plot, [epsi])
                steeringScope.axes[2].sample(t_plot, [u, v])
                steeringScope.axes[3].sample(t_plot, [delta, delta_ref])
                steeringScope.axes[4].sample(t_plot, [[x, y]])

                if "arrow" in globals():
                    arrow.setPos(x, y)
                    arrow.setStyle(angle=180 - th * 180 / np.pi)

                last_scope_t = t

            # ======================================================
            # ======================== PRINT ========================
            # ======================================================

            if t - last_print_t >= 0.2:
                prof = f_controller.last_profile
                budget_ms = 1000.0 / controllerUpdateRate
                if mpc_dt_ms > budget_ms:
                    pass
                    # print(f"WARNING: MPC time {mpc_dt_ms:.1f} ms > budget {budget_ms:.1f} ms")

                # print(
                #     f"[{mission_state.name}] "
                #     f"kappa={kappa_ref:.3f}, ey={ey:.3f}, epsi={epsi:.3f}, "
                #     f"delta={delta:.3f}, v_cmd/u/v={v_cmd:.2f}/{u:.2f}/{v:.2f}, "
                #     f"mpc_ms={mpc_dt_ms:.1f}, avg={mpc_ms_avg:.1f}, "
                #     f"proj={prof['project_ms']:.2f}, hor={prof['horizon_ms']:.2f}, "
                #     f"solve={prof['solve_ms']:.2f}, unpack={prof['unpack_ms']:.2f}, "
                #     f"total={prof['total_ms']:.2f}, status={prof['status']}"
                # )
                last_print_t = t

        # print("Control thread terminated.")


# ============================================================
# ===================== RUN + SCOPES =========================
# ============================================================

if __name__ == '__main__':

    # --------- Scope FPS ----------
    fps = 10 if IS_PHYSICAL_QCAR else 30

    # --------- Create Scope ----------
    steeringScope = MultiScope(
        rows=4,
        cols=2,
        title='Vehicle Steering Control',
        fps=fps
    )

    steeringScope.addAxis(row=0, col=0, timeWindow=tf, yLabel='ey', yLim=(-2.5, 2.5))
    steeringScope.axes[0].attachSignal(name='ey')

    steeringScope.addAxis(row=1, col=0, timeWindow=tf, yLabel='epsi', yLim=(-1, 5))
    steeringScope.axes[1].attachSignal(name='epsi')

    steeringScope.addAxis(row=2, col=0, timeWindow=tf, yLabel='Velocity [m/s]', yLim=(-3.5, 3.5))
    steeringScope.axes[2].attachSignal(name='u')
    steeringScope.axes[2].attachSignal(name='v_ref')

    steeringScope.addAxis(row=3, col=0, timeWindow=tf, yLabel='Steering Angle [rad]', yLim=(-0.6, 0.6))
    steeringScope.axes[3].attachSignal(name='delta_cmd')
    steeringScope.axes[3].attachSignal(name='delta_ref')
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(
        row=0, col=1, rowSpan=4,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-2.5, 2.5),
        yLim=(-1, 5)
    )

    # --------- Background map image ----------
    if images is not None:
        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            steeringScope.axes[4].attachImage(
                scale=(-0.002035, 0.002035),
                offset=(1125, 2365),
                rotation=180,
                levels=(0, 255)
            )
            steeringScope.axes[4].images[0].setImage(image=im)

    # --------- Plot reference path ----------
    referencePath = pg.PlotDataItem(
        pen={'color': (85, 168, 104), 'width': 2},
        name='Reference'
    )
    steeringScope.axes[4].plot.addItem(referencePath)

    # initial segment reference
    _init_seq = roadmap.generate_path([HUB_NODE, PICKUP_NODE])
    referencePath.setData(_init_seq[0, :], _init_seq[1, :])

    # Estimated trace
    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    # Arrow
    arrow = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'fillColor': [196, 78, 82], 'width': 1},
        brush=[196, 78, 82]
    )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)

    # --------- Start thread ----------
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive():
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        gps.terminate()

    if not IS_PHYSICAL_QCAR:
        try:
            qlabs_setup.terminate()
        except Exception:
            pass

    input('Experiment complete. Press any key to exit...')
