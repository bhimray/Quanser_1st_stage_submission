import numpy as np
import time
from pal.utilities.math import wrap_to_pi

from FrenetNonLinearMPC import FrenetNonlinearMPC


class FrenetNonlinearMPCController:
    """
    Measurement -> Frenet (s, ey, epsi) via path.project_frenet()
    NMPC -> (delta_cmd, a_cmd)
    Speed command returned as v_cmd (optional external PID can track it).
    """

    def __init__(self, path, Ts, L, N=40, **nmpc_kwargs):
        self.path = path
        self.Ts = float(Ts)
        self.N = int(N)

        self.nmpc = FrenetNonlinearMPC(path=path, Ts=Ts, N=N, L=L, **nmpc_kwargs)

        self.delta_prev = 0.0

        self._s_horizon = np.zeros(self.N + 1, dtype=float)
        self._kappa_horizon = np.zeros(self.N, dtype=float)
        self._delta_ref_horizon = np.zeros(self.N, dtype=float)
        self._vref_horizon = np.zeros(self.N + 1, dtype=float)
        self._x0 = np.zeros(4, dtype=float)  # [s, ey, epsi, v]
        self._t_idx = self.Ts * np.arange(self.N + 1, dtype=float)

        self.last_profile = {
            "project_ms": 0.0,
            "horizon_ms": 0.0,
            "solve_ms": 0.0,
            "unpack_ms": 0.0,
            "total_ms": 0.0,
            "status": 0,
        }

    def compute_control(self, x_world, psi, v_meas):
        t_total = time.perf_counter()

        # --- Project to Frenet
        t0 = time.perf_counter()
        s, ey, epsi, _, _ = self.path.project_frenet(x_world[0], x_world[1], psi)
        project_ms = (time.perf_counter() - t0) * 1000.0

        self._x0[0] = float(s)
        self._x0[1] = float(ey)
        self._x0[2] = wrap_to_pi(float(epsi))
        self._x0[3] = float(v_meas)

        # --- Horizon κ(s)
        t1 = time.perf_counter()
        speed = max(float(v_meas), 0.0)
        self._s_horizon[:] = s + self._t_idx * speed

        # If your Path2D can vectorize kappa_at_s, use it here.
        for k in range(self.N):
            self._kappa_horizon[k] = self.path.kappa_at_s(self._s_horizon[k])

        kappa_ref = float(self._kappa_horizon[0])

        vref_seq = None
        if getattr(self.path, "v_ref", None) is not None:
            for k in range(self.N + 1):
                self._vref_horizon[k] = self.path.v_ref_at_s(self._s_horizon[k])
            vref_seq = self._vref_horizon

        horizon_ms = (time.perf_counter() - t1) * 1000.0
        delta_ref_seq = np.arctan(self.nmpc.L * self._kappa_horizon, out=self._delta_ref_horizon)
        # --- NMPC solve
        X_opt, U_opt = self.nmpc.solve(
            self._x0,
            self._kappa_horizon,
            delta_prev=self.delta_prev,
            vref_seq_np=vref_seq,
            delta_ref_seq_np=delta_ref_seq,
        )

        delta_cmd = float(U_opt[0, 0])
        a_cmd = float(U_opt[0, 1])
        v_clip = self.path.v_ref_at_s(self._s_horizon[0])
        v_cmd = float(np.clip(v_meas + self.Ts * a_cmd, 0.0, v_clip))

        self.delta_prev = delta_cmd

        self.last_profile = {
            "project_ms": project_ms,
            "horizon_ms": horizon_ms,
            "solve_ms": float(self.nmpc.last_solve_ms),
            "unpack_ms": float(self.nmpc.last_unpack_ms),
            "total_ms": (time.perf_counter() - t_total) * 1000.0,
            "status": int(self.nmpc.last_status),
        }
        return delta_cmd, v_cmd, kappa_ref, float(ey), float(epsi)
