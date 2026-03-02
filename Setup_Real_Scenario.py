import os
import sys
import numpy as np
import time
import math
import threading

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape

from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.yield_sign import QLabsYieldSign
from qvl.stop_sign import QLabsStopSign
from qvl.traffic_cone import QLabsTrafficCone
from qvl.traffic_light import QLabsTrafficLight

import pal.resources.rtmodels as rtmodels
from pal.products.qcar import QCAR_CONFIG


# --- CONFIGURATION ---
right_hand_driving = True
GREEN_TIME = 5   
YELLOW_TIME = 2  
RED_CLEAR = 1

_TRAFFIC_THREAD = None
_TRAFFIC_STOP_EVENT = threading.Event()

def setup(
        initialPosition=[0, 0, 0.000],
        initialOrientation=[0, 0, 0],
        rtModel=rtmodels.QCAR2
    ):

    os.system('cls')
    qlabs = QuanserInteractiveLabs()

    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        sys.exit()
    print("Connected to QLabs")

    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # ------------------------------------------------
    # SPAWN SIGNAGE & TRAFFIC INFRASTRUCTURE
    # ------------------------------------------------

    # spawn_crosswalks(qlabs)
    # spawn_signs(qlabs, right_hand_driving)
    traffic_lights = spawn_traffic_lights(qlabs, right_hand_driving)
    spawn_cones(qlabs)
    start_traffic_light_manager(traffic_lights)

    # ------------------------------------------------
    # SPAWN QCAR
    # ------------------------------------------------

    if QCAR_CONFIG['cartype'] == 1:
        hqcar = QLabsQCar(qlabs)
        rtModel = rtmodels.QCAR
    else:
        hqcar = QLabsQCar2(qlabs)
        rtModel = rtmodels.QCAR2

    hqcar.spawn_id(
        actorNumber=0,
        location=[x * 10 for x in initialPosition],
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # Camera
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()

    # ------------------------------------------------
    # START REAL-TIME MODEL
    # ------------------------------------------------

    QLabsRealTime().start_real_time_model(rtModel)

    return hqcar, traffic_lights


def spawn_crosswalks(qlabs):
    crosswalk = QLabsCrosswalk(qlabs)

    crosswalk.spawn_degrees([-12.992, -7.407, 0.005], [0,0,48], configuration=0)
    crosswalk.spawn_degrees([-6.788, 45, 0.00], [0,0,90], configuration=1)
    crosswalk.spawn_degrees([21.733, 3.347, 0.005], [0,0,0], configuration=2)
    crosswalk.spawn_degrees([21.733, 16, 0.005], [0,0,0], configuration=2)


def spawn_signs(qlabs, right_hand_driving):
    # Like the crosswalks, we don't need to access the actors again after
    # creating them.

    roundabout_sign = QLabsRoundaboutSign(qlabs)
    yield_sign = QLabsYieldSign(qlabs)
    stop_sign = QLabsStopSign(qlabs)

    if (right_hand_driving):
        stop_sign.spawn_degrees([17.561, 17.677, 0.215], [0,0,90])
        stop_sign.spawn_degrees([24.3, 1.772, 0.2], [0,0,-90])
        stop_sign.spawn_degrees([14.746, 6.445, 0.215], [0,0,180])

        roundabout_sign.spawn_degrees([3.551, 40.353, 0.215], [0,0,180])
        roundabout_sign.spawn_degrees([10.938, 28.824, 0.215], [0,0,-135])
        roundabout_sign.spawn_degrees([24.289, 32.591, 0.192], [0,0,-90])

        yield_sign.spawn_degrees([-2.169, -12.594, 0.2], [0,0,180])
    else:
        stop_sign.spawn_degrees([24.333, 17.677, 0.215], [0,0,90])
        stop_sign.spawn_degrees([18.03, 1.772, 0.2], [0,0,-90])
        stop_sign.spawn_degrees([14.746, 13.01, 0.215], [0,0,180])

        roundabout_sign.spawn_degrees([16.647, 28.404, 0.215], [0,0,-45])
        roundabout_sign.spawn_degrees([6.987, 34.293, 0.215], [0,0,-130])
        roundabout_sign.spawn_degrees([9.96, 46.79, 0.2], [0,0,-180])

        yield_sign.spawn_degrees([-21.716, 7.596, 0.2], [0,0,-90])


def spawn_traffic_lights(qlabs, right_hand_driving):
    # Initialize handles
    tl1 = QLabsTrafficLight(qlabs)
    tl2 = QLabsTrafficLight(qlabs)
    tl3 = QLabsTrafficLight(qlabs)
    tl4 = QLabsTrafficLight(qlabs)

    if (right_hand_driving):
        tl1.spawn_id_degrees(actorNumber=0, location=[5.889, 16.048, 0.215], rotation=[0,0,0])
        tl2.spawn_id_degrees(actorNumber=1, location=[-2.852, 1.65, 0], rotation=[0,0,180])
        tl3.spawn_id_degrees(actorNumber=3, location=[8.443, 5.378, 0], rotation=[0,0,-90])
        tl4.spawn_id_degrees(actorNumber=4, location=[-4.202, 13.984, 0.186], rotation=[0,0,90])
    else:
        tl1.spawn_id_degrees(actorNumber=0, location=[-2.831, 16.643, 0.186], rotation=[0,0,180])
        tl2.spawn_id_degrees(actorNumber=1, location=[5.653, 1.879, 0], rotation=[0,0,0])
        tl3.spawn_id_degrees(actorNumber=3, location=[8.779, 13.7, 0.215], rotation=[0,0,90])
        tl4.spawn_id_degrees(actorNumber=4, location=[-4.714, 4.745, 0], rotation=[0,0,-90])

    return {'NS': [tl1, tl2], 'EW': [tl3, tl4]}

def traffic_light_manager(lights, stop_event=None):
    """ Background loop to cycle lights automatically """
    if stop_event is None:
        stop_event = _TRAFFIC_STOP_EVENT

    while not stop_event.is_set():
        # Phase 1: NS Green
        # for l in lights['NS']: l.set_color(l.COLOR_RED)
        for l in lights['NS']: l.set_color(l.COLOR_GREEN)
        for l in lights['EW']: l.set_color(l.COLOR_RED)
        if stop_event.wait(GREEN_TIME):
            break

        # Phase 2: NS Yellow
        # for l in lights['NS']: l.set_color(l.COLOR_RED)
        for l in lights['NS']: l.set_color(l.COLOR_YELLOW)
        if stop_event.wait(YELLOW_TIME):
            break

        # Phase 3: All Red
        for l in lights['NS']: l.set_color(l.COLOR_RED)
        if stop_event.wait(RED_CLEAR):
            break

        # Phase 4: EW Green
        # for l in lights['EW']: l.set_color(l.COLOR_RED)
        for l in lights['EW']: l.set_color(l.COLOR_GREEN)
        if stop_event.wait(GREEN_TIME):
            break

        # Phase 5: EW Yellow
        # for l in lights['EW']: l.set_color(l.COLOR_RED)
        for l in lights['EW']: l.set_color(l.COLOR_YELLOW)
        if stop_event.wait(YELLOW_TIME):
            break

        # Phase 6: All Red
        for l in lights['EW']: l.set_color(l.COLOR_RED)
        if stop_event.wait(RED_CLEAR):
            break


def start_traffic_light_manager(lights):
    global _TRAFFIC_THREAD
    _TRAFFIC_STOP_EVENT.clear()

    _TRAFFIC_THREAD = threading.Thread(
        target=traffic_light_manager,
        args=(lights, _TRAFFIC_STOP_EVENT),
        daemon=True,
    )
    _TRAFFIC_THREAD.start()


def terminate():
    _TRAFFIC_STOP_EVENT.set()
    if _TRAFFIC_THREAD is not None:
        _TRAFFIC_THREAD.join(timeout=2.0)
        
def spawn_cones(qlabs):
    cone = QLabsTrafficCone(qlabs)
    # cone.spawn(location=[2.313, 19.408, 0.005], configuration=1)
    # for i in range(5):
    #     cone.spawn(
    #         location=[-15.313, 35.374 + i*-1.3, 0.25],
    #         configuration=1
    #     )
