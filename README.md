# Autonomous QCar Taxi System with Frenet NMPC and Perception-Aware Safety

## 1. Project Description
This project implements an autonomous urban taxi mission for the Quanser QCar platform in QLabs and on hardware-compatible software interfaces. The system combines:

- A **path-following nonlinear model predictive controller (NMPC)** in Frenet coordinates.
- A **route-level mission manager** for repeated pickup and dropoff operations.
- A **vision-based perception module** for traffic-light and obstacle awareness.
- A **safety policy layer** that modifies commanded speed based on perception outcomes.

The objective is to achieve robust lane-centered tracking with smooth steering/acceleration while respecting road geometry, control limits, and traffic constraints.

## 2. Mission and Operational Scenario
The mission is implemented in `vehicle_control_traffic.py` as a finite-state machine:

- Hub (`node 10`) to Pickup (`node 20`)
- Stop at pickup for a fixed dwell time
- Pickup to Dropoff (`node 9`)
- Stop at dropoff for a fixed dwell time
- Return to hub
- Repeat cyclically

Each route leg is converted to a spline-smoothed geometric path and assigned its own NMPC solver instance.

## 3. Repository Structure (Phase 3)
- `vehicle_control_traffic.py`: Main integrated mission script (control, perception policy, FSM, visualization).
- `FrenetNonLinearMPC.py`: acados-based NMPC model and solver formulation.
- `FrenetNonLinearMPCController.py`: Runtime interface from world state to NMPC commands.
- `Path2D.py`: Path smoothing, Frenet projection, curvature, and speed-profile generation.
- `qlabs_setup.py`: QLabs world setup and traffic-light actor manager.
- `RacingPerception.py`: Standalone lightweight perception module (classical vision design).
- `SharedState.py`, `Perception_thread.py`: Alternative simplified thread/state helpers.
- `yolo.py`, `yoloObjectDetection.py`, `object_detection.py`: Experimental/object-detection utilities.

## 4. System Architecture
The implemented control stack follows:

1. **Route generation**
   - Roadmap nodes are converted into waypoint sequences for each mission leg.

2. **Geometric preprocessing**
   - `Path2D` fits and samples a smooth centerline.
   - Arc-length `s`, heading `psi(s)`, curvature `kappa(s)`, and speed profile `v_ref(s)` are computed.

3. **State estimation**
   - `QCarEKF` fuses IMU/GPS (when available) to estimate vehicle pose.

4. **Frenet projection**
   - Current `(x, y, psi)` is projected to `(s, e_y, e_psi)` relative to the centerline.

5. **NMPC solve**
   - acados SQP-RTI solves a constrained nonlinear OCP for steering and acceleration.

6. **Low-level speed loop**
   - A PID controller tracks NMPC speed command.

7. **Perception and safety**
   - Traffic/obstacle observations are mapped to a speed-cap policy (`v_cmd -> 0` when required).

8. **Actuation**
   - Commands are written to QCar steering and throttle interfaces.

## 5. Vehicle Dynamics Model
The controller uses a kinematic bicycle model in Frenet coordinates with state:

\[
x = [s,\ e_y,\ e_\psi,\ v]
\]

and control:

\[
u = [\delta,\ a]
\]

Augmented state used for rate constraints:

\[
x_{aug} = [s,\ e_y,\ e_\psi,\ v,\ \delta_{prev},\ a_{prev}]
\]

### 5.1 Continuous-Time Frenet Dynamics
In `FrenetNonLinearMPC.py`, with path curvature parameter \(\kappa\):

\[
\dot{s} = \frac{v\cos(e_\psi)}{1-\kappa e_y}, \quad
\dot{e}_y = v\sin(e_\psi)
\]
\[
\dot{e}_\psi = \frac{v}{L}\tan(\delta) - \kappa\dot{s}, \quad
\dot{v}=a
\]
\[
\dot{\delta}_{prev}=0,\quad \dot{a}_{prev}=0
\]

where denominator \(1-\kappa e_y\) is lower-bounded numerically for stability.

### 5.2 Path and Lane Quantities
`Path2D` computes:

- Smoothed centerline via spline fitting.
- Arc-length parameter \(s\).
- Heading from spline derivative.
- Curvature from first and second derivatives.
- Curvature smoothing and slope-limiting to reduce spikes.

## 6. NMPC Formulation
The optimal control problem is discretized over horizon \(N\) with step \(T_s\), and solved by acados (`SQP_RTI` with `PARTIAL_CONDENSING_HPIPM`).

### 6.1 Cost Function
Stage output vector in code:

\[
y = [e_y,\ e_\psi,\ v,\ \delta,\ a,\ (\delta-\delta_{prev}),\ (a-a_{prev})]
\]

Weighted least-squares penalty:

- Lateral deviation and heading error.
- Velocity tracking.
- Absolute steering and acceleration effort.
- Steering rate and acceleration rate penalties.

Terminal cost is currently set to zero weighting.

### 6.2 Constraints
Implemented hard constraints include:

- Steering and acceleration bounds: \(\delta \in [-\delta_{max}, \delta_{max}],\ a \in [a_{min}, a_{max}]\)
- Speed bounds: \(v \in [v_{min}, v_{max}]\)
- Lateral error bounds via path constraint: \(|e_y| \le e_{y,max}\)
- Rate constraints:
  - \(|\delta-\delta_{prev}| \le \dot{\delta}_{max}T_s\)
  - \(|a-a_{prev}| \le \dot{a}_{max}T_s\)

### 6.3 Solver Settings (Current Code)
- QP solver: `PARTIAL_CONDENSING_HPIPM`
- NLP type: `SQP_RTI`
- Hessian approximation: `GAUSS_NEWTON`
- Integrator: `ERK` (4 stages, 1 step)
- Tolerance: `1e-4`

## 7. Speed Planning and Curvature-Aware Reference
The system uses two complementary speed concepts:

1. **Offline path speed profile** (`Path2D.compute_speed_profile`)
   - Curvature-limited speed from lateral acceleration:
     \[
     v_{curve}(s)=\sqrt{\frac{a_{lat,max}}{|\kappa(s)|}}
     \]
   - Forward/backward passes enforce acceleration/deceleration limits.

2. **MPC horizon speed reference**
   - Runtime reference sampled along predicted \(s\)-horizon.
   - Optionally curvature-modulated in controller settings.

## 8. Perception Pipeline
In the integrated mission script (`vehicle_control_traffic.py`), a `RacingPerception` class performs camera/depth-based perception with temporal filtering.

### 8.1 Traffic-Light Inference
- Uses detector proposals (`ObjectDetection`) and geometric priors (upper image region).
- Classifies light color using HSV masks on cropped traffic-light ROI.
- Uses band-based and threshold-based logic for RED/YELLOW/GREEN.
- Applies persistence counters before committing state transitions.

### 8.2 Obstacle Inference
- Uses image-space priors (lower ROI and ego-lane center gating).
- Combines lane/region filtering with mask and contour logic.
- Uses depth-based distance estimation and range validity checks.
- Applies temporal confirmation and decay to suppress flicker.

### 8.3 Threading and Shared State
- Perception is designed to run in its own thread and publish:
  - `traffic_light`
  - `obstacle`
  - `distance_m`
  - timestamp and debug buffers
- Control loop consumes only latest shared values and applies stale-data checks.

## 9. Perception-to-Control Safety Policy
Safety policy in `apply_traffic_caps(...)`:

- Forces `v_cmd = 0` when:
  - light is RED or YELLOW (with hold timer),
  - obstacle is active (with hold timer),
  - optional distance gate confirms obstacle is within stop range.
- Clears held red state when GREEN is observed.
- Uses timestamp age (`PERCEPTION_STALE_S`) to discard stale observations.

This design avoids abrupt toggling while preserving conservative stop behavior.

## 10. Important Implementation Note
In the current `__main__` section of `vehicle_control_traffic.py`, the perception thread startup lines are present but commented out. As written, `shared_state` remains at defaults unless the thread is enabled.

To activate live perception-driven stopping, uncomment:

- creation of `perceptionThread = Thread(...)`
- `perceptionThread.start()`

## 11. Key Tunable Parameters
Primary tuning groups in code:

- **Vehicle/control rates**: `controllerUpdateRate`, `Ts`, `scopeUpdateRate`
- **Lane bounds**: `W`, `b`, `ey_min`, `ey_max`
- **MPC weights**: `Q`, `R`, `Sdu`
- **MPC limits**: `delta_max`, `ddelta_max`, `a_min`, `a_max`, `da_max`, `v_min`, `v_max`
- **Perception safety**: `PERCEPTION_STALE_S`, `RED_HOLD_S`, `OBS_HOLD_S`, `OBS_DISTANCE_STOP_M`
- **Mission logic**: `STOP_DURATION`, `GOAL_TOL`

## 12. How to Run
From this directory:

```bash
python vehicle_control_traffic.py
```

Recommended order:

1. Verify acados and interfaces are available.
2. Run in QLabs simulation first.
3. Enable perception thread after baseline NMPC tracking is stable.
4. Tune hold/debounce constants to balance responsiveness vs false stops.

## 13. Validation Checklist
Suggested experimental checks:

- Path tracking RMS lateral error on each leg.
- Maximum steering-rate usage vs configured bound.
- Solver status and solve-time budget at runtime.
- False-stop and missed-stop counts for traffic light logic.
- Obstacle stop distance consistency vs depth estimate.
- Mission completion success across multiple Hub→Pickup→Dropoff cycles.

## 14. Current Limitations and Future Improvements
- Perception logic is primarily heuristic and simulator-oriented.
- Light/obstacle detection robustness may degrade under unseen lighting or clutter.
- Terminal cost shaping in NMPC can be further improved.
- Perception thread is not enabled by default in the current entrypoint.

High-value next steps:

- Add systematic logging of solver/perception/mission metrics.
- Enable perception thread by default with watchdog diagnostics.
- Integrate learned detectors with confidence calibration and fallback policies.
- Add unit/integration tests for FSM transitions and safety policy edges.

## 15. Academic Positioning
This implementation demonstrates a practical autonomous driving stack integrating:

- trajectory representation and Frenet projection,
- constrained NMPC with real-time iteration,
- perception-informed supervisory safety,
- mission-level task sequencing.

The resulting structure is suitable for graduate-level demonstration of model-based control integrated with perception and traffic-rule logic on embedded autonomous platforms.

