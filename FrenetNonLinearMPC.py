import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
from pal.utilities.math import wrap_to_pi

_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE
while ROOT != ROOT.parent and not (ROOT / "interfaces" / "acados_template").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "interfaces" / "acados_template"))

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class FrenetNonlinearMPC:
    """
    Frenet NMPC solved with acados (SQP_RTI).
    Vehicle states: [s, ey, epsi, v]
    Inputs: [delta, a]
    Augmented states: [s, ey, epsi, v, delta_prev, a_prev]
    Parameter: kappa
    """

    def __init__(
        self,
        path,
        Ts=0.02,
        L=0.2,
        N=20,
        ey_max=0.25,
        delta_max=0.52,
        ddelta_max=2.0,          # [rad/s]
        a_min=-1.5,
        a_max=1.5,
        da_max=3.0,              # [m/s^3]  (ADD: rate constraint for accel)
        v_min=0.0,
        v_max=1.0,
        w_ey=50.0,
        w_epsi=20.0,
        w_v=5.0,
        w_delta=10.0,
        w_a=1.0,
        w_ddelta=10.0,
        w_da=1.0,
        use_curv_speed_ref=True,
        v_ref_base=0.8,
        kappa_speed_gain=2.0,
        solver_tag=None,
        build_on_init=True,
    ):
        self.path = path
        self.Ts = float(Ts)
        self.N = int(N)
        self.L = float(L)

        self.ey_max = float(ey_max)
        self.delta_max = float(delta_max)
        self.ddelta_max = float(ddelta_max)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.da_max = float(da_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.w_ey = float(w_ey)
        self.w_epsi = float(w_epsi)
        self.w_v = float(w_v)
        self.w_delta = float(w_delta)
        self.w_a = float(w_a)
        self.w_ddelta = float(w_ddelta)
        self.w_da = float(w_da)

        self.use_curv_speed_ref = bool(use_curv_speed_ref)
        self.v_ref_base = float(v_ref_base)
        self.kappa_speed_gain = float(kappa_speed_gain)
        self.solver_tag = None if solver_tag is None else str(solver_tag).strip()

        if self.solver_tag:
            safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in self.solver_tag)
            self.model_name = f"frenet_bicycle_nmpc_{safe}"
            self.codegen_suffix = safe
        else:
            self.model_name = "frenet_bicycle_nmpc"
            self.codegen_suffix = "frenet_nmpc"

        # dims
        self.nx = 4
        self.nu = 2
        self.nx_aug = 6

        # indices in augmented state
        self.IDX_S = 0
        self.IDX_EY = 1
        self.IDX_EPSI = 2
        self.IDX_V = 3
        self.IDX_DELTA_PREV = 4
        self.IDX_A_PREV = 5

        # Cost output y = [ey, epsi, v, delta, a, (delta - delta_prev), (a - a_prev)]
        self.ny = 7
        self.IDX_VREF_IN_Y = 2
        self.IDX_DELTA_REF_IN_Y = 3

        # buffers
        self._p_stage = np.zeros(1, dtype=float)
        self._yref_stage = np.zeros(self.ny, dtype=float)
        self._yref_terminal = np.zeros(1, dtype=float)
        self._vref_buffer = np.zeros(self.N + 1, dtype=float)
        self._delta_ref_buffer = np.zeros(self.N, dtype=float)
        self._x0_aug = np.zeros(self.nx_aug, dtype=float)

        # last solution for warm-start
        self.last_X = None
        self.last_U = None
        self.last_status = 0
        self.last_solve_ms = 0.0
        self.last_unpack_ms = 0.0
        self._a_prev_applied = 0.0  # used to seed a_prev

        # Windows acados env helpers
        if os.name == "nt":
            os.environ.setdefault("ACADOS_SOURCE_DIR", str(ROOT))
            os.environ.setdefault("CC", "gcc")
            os.environ.setdefault("RM", "echo")

        self.solver = None
        if build_on_init:
            self._build_solver()

    def _build_solver(self):
        # %statuses = {
        # %    0: 'ACADOS_SUCCESS',
        # %    1: 'ACADOS_NAN_DETECTED',
        # %    2: 'ACADOS_MAXITER',
        # %    3: 'ACADOS_MINSTEP',
        # %    4: 'ACADOS_QP_FAILURE',
        # %    5: 'ACADOS_READY'
        ocp = AcadosOcp()

        # -------- States and Inputs
        x = ca.SX.sym("x", self.nx_aug)
        u = ca.SX.sym("u", self.nu)
        kappa = ca.SX.sym("kappa", 1)

        s     = x[self.IDX_S]
        ey    = x[self.IDX_EY]
        epsi  = x[self.IDX_EPSI]
        v     = x[self.IDX_V]
        delta_prev = x[self.IDX_DELTA_PREV]
        a_prev     = x[self.IDX_A_PREV]

        delta = u[0]
        a     = u[1]

        # -------- Continuous Frenet Dynamics

        denom = 1.0 - kappa[0] * ey
        denom = ca.fmax(denom, 1e-3)

        s_dot    = v * ca.cos(epsi) / denom
        ey_dot   = v * ca.sin(epsi)
        epsi_dot = (v / self.L) * ca.tan(delta) - kappa[0] * s_dot
        v_dot    = a

        # augmented states: delta_prev_dot = 0, a_prev_dot = 0
        delta_prev_dot = 0.0
        a_prev_dot     = 0.0

        xdot = ca.vertcat(
            s_dot,
            ey_dot,
            epsi_dot,
            v_dot,
            delta_prev_dot,
            a_prev_dot,
        )

        # -------- Model Definition

        model = AcadosModel()
        model.name = self.model_name

        model.x = x
        model.u = u
        model.p = kappa

        # Continuous dynamics (IMPORTANT)
        model.f_expl_expr = xdot

        # -------- Path Constraints

        # |ey| <= ey_max
        h_ey = ca.vertcat(ey, -ey)

        # Rate constraints
        ddelta = delta - delta_prev
        da     = a - a_prev

        h_rate = ca.vertcat(
            ddelta,
            -ddelta,
            da,
            -da,
        )

        model.con_h_expr = ca.vertcat(h_ey, h_rate)

        ocp.model = model

        # -------- Horizon
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.Ts

        # -------- Cost (Linear LS)

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny = self.ny
        ny_e = 1  # terminal cost on s only (can be zero if not needed)
        Vx = np.zeros((ny, self.nx_aug))
        Vu = np.zeros((ny, self.nu))

        # 1) e_y
        Vx[0, 1] = 1.0

        # 2) e_psi
        Vx[1, 2] = 1.0

        # 3) v
        Vx[2, 3] = 1.0

        # 4) delta (absolute steering)
        Vu[3, 0] = 1.0

        # 5) a (absolute accel)
        Vu[4, 1] = 1.0

        # 6) delta rate
        Vx[5, self.IDX_DELTA_PREV] = -1.0
        Vu[5, 0] = 1.0

        # 7) accel rate
        Vx[6, self.IDX_A_PREV] = -1.0
        Vu[6, 1] = 1.0


        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.W = np.diag([
            self.w_ey,
            self.w_epsi,
            self.w_v,
            self.w_delta,
            self.w_a,
            self.w_ddelta,
            self.w_da
        ])


        ocp.cost.yref = np.zeros(ny)
        
        ocp.cost.Vx_e = np.zeros((ny_e, self.nx_aug))
        ocp.cost.W_e = np.zeros((ny_e, ny_e))
        ocp.cost.yref_e = np.zeros(ny_e)

        # -------- Input Bounds
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([-self.delta_max, self.a_min])
        ocp.constraints.ubu = np.array([ self.delta_max, self.a_max])

        # -------- State Bounds
        ocp.constraints.idxbx = np.array([
            self.IDX_V,
            self.IDX_DELTA_PREV,
            self.IDX_A_PREV,
        ])

        ocp.constraints.lbx = np.array([
            self.v_min,
            -self.delta_max,
            self.a_min,
        ])

        ocp.constraints.ubx = np.array([
            self.v_max,
            self.delta_max,
            self.a_max,
        ])

        # -------- Path Constraint Bounds

        ddelta_lim = self.ddelta_max * self.Ts
        da_lim     = self.da_max * self.Ts

        ocp.constraints.lh = np.array([
            -1e9,
            -1e9,
            -1e9,
            -1e9,
            -1e9,
            -1e9,
        ])

        ocp.constraints.uh = np.array([
            self.ey_max,
            self.ey_max,
            ddelta_lim,
            ddelta_lim,
            da_lim,
            da_lim,
        ])

        ocp.constraints.x0 = np.zeros(self.nx_aug)
        ocp.parameter_values = np.array([0.0])

        # -------- Solver Options ---------------------

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_exact_hessian = "NO"


        # model discretization (important for SQP_RTI)
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1

        ocp.solver_options.tol = 1e-4

        # -------- Build ------------

        code_export_dir = str(Path(__file__).parent / f"c_generated_code_{self.codegen_suffix}")
        json_file = str(Path(__file__).parent / f"acados_ocp_{self.codegen_suffix}.json")
        ocp.code_export_directory = code_export_dir

        if os.name == "nt":
            AcadosOcpSolver.generate(ocp, json_file=json_file)
            makefile_path = Path(code_export_dir) / "Makefile"
            if makefile_path.exists():
                makefile_text = makefile_path.read_text()
                makefile_text = makefile_text.replace(
                    "LIBACADOS_OCP_SOLVER=libacados_ocp_solver_",
                    "LIBACADOS_OCP_SOLVER=acados_ocp_solver_",
                )
                makefile_path.write_text(makefile_text)
            AcadosOcpSolver.build(code_export_dir, with_cython=False)
            self.solver = AcadosOcpSolver(None, json_file=json_file, build=False, generate=False)
        else:
            self.solver = AcadosOcpSolver(ocp, json_file=json_file)

    def _build_vref(self, kappa_seq):
        vref = self._vref_buffer
        if not self.use_curv_speed_ref:
            vref.fill(self.v_ref_base)
            return vref

        vref[:-1] = self.v_ref_base / (1.0 + self.kappa_speed_gain * np.abs(kappa_seq))
        vref[-1] = vref[-2]
        np.clip(vref, self.v_min, self.v_max, out=vref)
        return vref

    def _warm_start(self, x0_aug, delta_prev):
        # If no previous solution: flat initial guess
        if self.last_X is None or self.last_U is None:
            X_guess = np.tile(x0_aug.reshape(1, -1), (self.N + 1, 1))
            U_guess = np.zeros((self.N, self.nu))
            U_guess[:, 0] = np.clip(delta_prev, -self.delta_max, self.delta_max)
        else:
            X_guess = np.vstack([self.last_X[1:, :], self.last_X[-1, :]])
            U_guess = np.vstack([self.last_U[1:, :], self.last_U[-1, :]])

        for k in range(self.N):
            self.solver.set(k, "x", X_guess[k])
            self.solver.set(k, "u", U_guess[k])
        self.solver.set(self.N, "x", X_guess[self.N])

    def solve(self, x0_np, kappa_seq_np, delta_prev=0.0, vref_seq_np=None, delta_ref_seq_np=None):
        if self.solver is None:
            self._build_solver()

        x0_np = np.asarray(x0_np, dtype=float).reshape(-1)
        if x0_np.size != self.nx:
            raise ValueError("x0_np must be shape (4,) -> [s, ey, epsi, v]")

        kappa_seq_np = np.asarray(kappa_seq_np, dtype=float).reshape(-1)
        if kappa_seq_np.size != self.N:
            raise ValueError("kappa_seq_np must have length N")

        # build augmented x0
        x0_aug = self._x0_aug
        x0_aug[:4] = x0_np
        x0_aug[self.IDX_EPSI] = wrap_to_pi(x0_aug[self.IDX_EPSI])
        x0_aug[self.IDX_DELTA_PREV] = float(delta_prev)
        x0_aug[self.IDX_A_PREV] = float(self._a_prev_applied)

        # vref
        if vref_seq_np is None:
            vref = self._build_vref(kappa_seq_np)
        else:
            vref_in = np.asarray(vref_seq_np, dtype=float).reshape(-1)
            if vref_in.size != self.N + 1:
                raise ValueError("vref_seq_np must have length N+1")
            np.clip(vref_in, self.v_min, self.v_max, out=self._vref_buffer)
            vref = self._vref_buffer

        # steering reference for stage cost
        if delta_ref_seq_np is None:
            delta_ref = self._delta_ref_buffer
            np.arctan(self.L * kappa_seq_np, out=delta_ref)
        else:
            delta_ref_in = np.asarray(delta_ref_seq_np, dtype=float).reshape(-1)
            if delta_ref_in.size != self.N:
                raise ValueError("delta_ref_seq_np must have length N")
            np.clip(delta_ref_in, -self.delta_max, self.delta_max, out=self._delta_ref_buffer)
            delta_ref = self._delta_ref_buffer

        # x0 equality via bounds at stage 0
        self.solver.set(0, "lbx", x0_aug)
        self.solver.set(0, "ubx", x0_aug)

        # parameters + references
        yref = self._yref_stage
        yref[:] = 0.0  # ensure clean
        for k in range(self.N):
            self._p_stage[0] = kappa_seq_np[k]
            self.solver.set(k, "p", self._p_stage)
            yref[self.IDX_VREF_IN_Y] = vref[k]
            yref[self.IDX_DELTA_REF_IN_Y] = delta_ref[k]
            self.solver.set(k, "yref", yref)

        self.solver.set(self.N, "yref", self._yref_terminal)

        # warm start
        self._warm_start(x0_aug, delta_prev)

        # solve
        t0 = time.perf_counter()
        status = self.solver.solve()
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self.last_status = int(status)

        # unpack
        t1 = time.perf_counter()
        X_opt = np.zeros((self.N + 1, self.nx_aug))
        U_opt = np.zeros((self.N, self.nu))
        for k in range(self.N):
            X_opt[k] = self.solver.get(k, "x")
            U_opt[k] = self.solver.get(k, "u")
        X_opt[self.N] = self.solver.get(self.N, "x")

        # wrap epsi trajectory
        X_opt[:, self.IDX_EPSI] = np.arctan2(np.sin(X_opt[:, self.IDX_EPSI]), np.cos(X_opt[:, self.IDX_EPSI]))

        self.last_X = X_opt.copy()
        self.last_U = U_opt.copy()
        self._a_prev_applied = float(U_opt[0, 1])
        self.last_unpack_ms = (time.perf_counter() - t1) * 1000.0

        return X_opt[:, : self.nx], U_opt
