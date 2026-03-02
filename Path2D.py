import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from pal.utilities.math import wrap_to_pi


class Path2D:
    """
    Waypoints as (N,2). Builds centerline points, arc-length s, heading psi(s), curvature kappa(s).
    Projection is performed on the SAME polyline used for psi/kappa (important for accuracy).
    """
    def __init__(self, waypoints_xy: np.ndarray, Ts: float, spline_smooth=0.3):
        assert waypoints_xy.ndim == 2 and waypoints_xy.shape[1] == 2

        self.Ts = float(Ts)

        # Keep raw waypoints if you need debug
        self.wp_raw = np.array(waypoints_xy, dtype=float)

        # --- initial arc length for parameterization
        dx = np.diff(self.wp_raw[:, 0])
        dy = np.diff(self.wp_raw[:, 1])
        ds = np.hypot(dx, dy)
        s_raw = np.hstack(([0.0], np.cumsum(ds)))
        L = float(s_raw[-1] + 1e-9)

        # Normalize parameter u ∈ [0,1]
        u = s_raw / L

        # Fit spline through raw points
        tck, _ = splprep([self.wp_raw[:, 0], self.wp_raw[:, 1]], u=u, s=float(spline_smooth))

        # Resample uniformly along path length (same number of points as input)
        N = self.wp_raw.shape[0]
        u_uniform = np.linspace(0.0, 1.0, N)

        x, y = splev(u_uniform, tck, der=0)
        dxu, dyu = splev(u_uniform, tck, der=1)
        ddxu, ddyu = splev(u_uniform, tck, der=2)

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

        # IMPORTANT: projection polyline uses smoothed points
        self.wp = np.column_stack([self.x, self.y])
        self.N = self.wp.shape[0]

        # Recompute arc-length on the smoothed polyline
        dpx = np.diff(self.wp[:, 0])
        dpy = np.diff(self.wp[:, 1])
        ds_smooth = np.hypot(dpx, dpy)
        self.s = np.hstack(([0.0], np.cumsum(ds_smooth)))
        self.length = float(self.s[-1] + 1e-9)

        # Heading psi (from spline derivatives)
        self.psi = np.arctan2(dyu, dxu)

        # Curvature from spline derivatives (parameter u)
        denom = (dxu * dxu + dyu * dyu) ** 1.5 + 1e-9
        kappa = (dxu * ddyu - dyu * ddxu) / denom

        # Smooth kappa to suppress spikes
        win = max(9, (self.N // 20) | 1)  # odd
        kappa = savgol_filter(kappa, window_length=win, polyorder=3)

        # Optional: limit curvature slope dκ/ds (more correct than using Ts)
        # Units: [1/m^2]
        kappa_slope_max = 3.0  # tune
        kappa_limited = np.zeros_like(kappa)
        kappa_limited[0] = kappa[0]
        for i in range(1, self.N):
            ds_i = max(self.s[i] - self.s[i - 1], 1e-6)
            dk = kappa[i] - kappa_limited[i - 1]
            dk = np.clip(dk, -kappa_slope_max * ds_i, +kappa_slope_max * ds_i)
            kappa_limited[i] = kappa_limited[i - 1] + dk

        self.kappa = kappa_limited

        # Projection continuity helpers
        self.last_i = 0
        self.last_s = 0.0

        # Optional speed profile
        self.v_ref = None

        # Debug dumps (optional)
        np.savetxt("path2d_s.csv", self.s, delimiter=",", header="s", comments="")
        np.savetxt("path2d_psi.csv", self.psi, delimiter=",", header="psi", comments="")
        np.savetxt("path2d_kappa.csv", self.kappa, delimiter=",", header="kappa", comments="")

    def _closest_segment(self, p, i_center=None, window=80):
        """
        Vectorized closest-point-on-segment search on a window around last_i.
        Returns: (dist, i, t, proj)
        """
        if i_center is None:
            i_center = self.last_i

        i0 = max(0, i_center - window)
        i1 = min(self.N - 2, i_center + window)

        P0 = self.wp[i0:i1+1]
        P1 = self.wp[i0+1:i1+2]
        V = P1 - P0

        L2 = np.sum(V * V, axis=1)
        good = L2 > 1e-12
        if not np.any(good):
            return (1e9, i_center, 0.0, self.wp[i_center])

        P0g = P0[good]
        Vg = V[good]
        L2g = L2[good]

        w = p - P0g
        t = np.clip(np.sum(w * Vg, axis=1) / L2g, 0.0, 1.0)
        proj = P0g + (t[:, None] * Vg)
        d = np.linalg.norm(p - proj, axis=1)

        j = int(np.argmin(d))
        # map back to original segment index
        good_idx = np.flatnonzero(good)
        i_best = int(i0 + good_idx[j])
        return float(d[j]), i_best, float(t[j]), proj[j]

    def project_frenet(self, x, y, psi_vehicle):
        """
        Returns Frenet (s, ey, epsi, psi_r, kappa) at closest projection.
        """
        p = np.array([x, y], dtype=float)
        dist, i, t, proj = self._closest_segment(p, i_center=self.last_i, window=100)
        self.last_i = i

        seg_len = float(self.s[i + 1] - self.s[i])
        s_proj = float(self.s[i] + t * seg_len)

        # interpolate psi,kappa at s_proj (reduces jumps)
        psi_r = float(np.interp(s_proj, self.s, self.psi))
        kappa = float(np.interp(s_proj, self.s, self.kappa))

        n = np.array([-np.sin(psi_r), np.cos(psi_r)], dtype=float)
        ey = float(n @ (p - proj))

        epsi = wrap_to_pi(float(psi_vehicle - psi_r))

        # Optional continuity guard (recommended for high-speed)
        # ds_max = 2.0 * (v_max + 0.1) * Ts
        # s_proj = float(np.clip(s_proj, self.last_s - ds_max, self.last_s + ds_max))
        # s_proj = max(s_proj, self.last_s)  # prevent going backwards if desired

        self.last_s = s_proj
        return s_proj, ey, epsi, psi_r, kappa

    def kappa_at_s(self, s_query):
        return float(np.interp(float(s_query), self.s, self.kappa))

    def v_ref_at_s(self, s_query):
        if self.v_ref is None:
            raise ValueError("v_ref has not been computed. Call compute_speed_profile() first.")
        return float(np.interp(float(s_query), self.s, self.v_ref))

    def compute_speed_profile(
        self,
        a_lat_max,
        v_min,
        v_max,
        a_accel,
        a_decel,
        kappa_min=1e-3,
    ):
        kappa_abs = np.maximum(np.abs(self.kappa), float(kappa_min))
        v_curve = np.sqrt(float(a_lat_max) / kappa_abs)
        v = np.clip(v_curve, float(v_min), float(v_max))

        ds = np.diff(self.s)
        for i in range(len(ds)):
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * float(a_accel) * ds[i]))
        for i in range(len(ds) - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * float(a_decel) * ds[i]))

        self.v_ref = np.clip(v, float(v_min), float(v_max))
        np.savetxt("path2d_vref.csv", self.v_ref, delimiter=",", header="v_ref", comments="")
        return self.v_ref
