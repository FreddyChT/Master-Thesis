"""Geometry utilities for blade processing.

These helpers are used by ``DataBladeAnalysis v8.py`` to read airfoil data
and compute blade geometry prior to meshing or analysis.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.gp import gp_Pnt


def read_selig_airfoil(path):
    """Read a Selig formatted airfoil file.

    Parameters
    ----------
    path : str
        Path to ``.dat`` or ``.csv`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of ``x`` and ``y`` coordinates. Used by
        :func:`process_airfoil_file`.
    """
    x, y = [], []
    with open(path, 'r') as f:
        next(f); next(f)
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            x.append(float(toks[0])); y.append(float(toks[1]))
    return np.array(x), np.array(y)


def calculate_intersection_point(P0, vr, P3, vs, d):
    """Compute intersection control points for the trailing edge NURBS closure.

    Used inside :func:`process_airfoil_file` when building the trailing edge
    spline.
    """

    def promote(v):
        v = np.asarray(v, float).ravel()
        return np.array([v[0], v[1], 0.0]) if v.size == 2 else v

    P0 = promote(P0); vr = promote(vr)
    P3 = promote(P3); vs = promote(vs)

    v_norm = (P3 - P0)
    v_norm /= np.linalg.norm(v_norm)

    n = np.cross(v_norm, np.cross(v_norm, vr))
    n /= np.linalg.norm(n)

    P2 = P0 + d * n
    P2_P3 = P3 + d * n

    CP1 = P0 + (np.dot(P2 - P0, n) / np.dot(vr, n)) * vr
    CP2 = P3 + (np.dot(P2_P3 - P3, n) / np.dot(vs, n)) * vs

    return CP1, CP2, P2, P2_P3


def create_nurbs_curve(P0, CP1, CP2, P3, weights=None):
    """Build a degree‑4 B‑spline for the trailing edge closure."""

    def promote_vec(v):
        arr = np.asarray(v, float).ravel()
        return np.array([arr[0], arr[1], 0.0]) if arr.size == 2 else arr

    P0_3 = promote_vec(P0)
    CP1_3 = promote_vec(CP1)
    CP2_3 = promote_vec(CP2)
    P3_3 = promote_vec(P3)

    PM = (CP1_3 + CP2_3) / 2.0
    cps = TColgp_Array1OfPnt(1, 5)
    for i, P in enumerate((P0_3, CP1_3, PM, CP2_3, P3_3), start=1):
        cps.SetValue(i, gp_Pnt(P[0], P[1], P[2]))

    if weights is None:
        weights = [1.0] * 5
    w_arr = TColStd_Array1OfReal(1, 5)
    for i, w in enumerate(weights, start=1):
        w_arr.SetValue(i, w)

    knots = TColStd_Array1OfReal(1, 2)
    knots.SetValue(1, 0.0); knots.SetValue(2, 1.0)
    mults = TColStd_Array1OfInteger(1, 2)
    mults.SetValue(1, 5); mults.SetValue(2, 5)

    return Geom_BSplineCurve(cps, w_arr, knots, mults, 4)


def sample_nurbs(curve, n_te):
    """Sample ``n_te`` points from the given NURBS curve."""
    ts = np.linspace(0.0, 1.0, n_te)
    pts = []
    for t in ts:
        p = curve.Value(t)
        pts.append((p.X(), p.Y()))
    return np.array(pts)


def resample_side(x, y, n_pts):
    """Resample a set of ``x``/``y`` points using arc length."""
    dx = np.diff(x); dy = np.diff(y)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    csx, csy = CubicSpline(s, x), CubicSpline(s, y)
    s_r = np.linspace(0.0, s[-1], n_pts)
    return csx(s_r), csy(s_r), s_r, s_r / s_r[-1]


def process_airfoil_file(path, n_points=1000, n_te=60, d_factor=0.5):
    """Return closed and resampled blade geometry.

    This routine is called both by the mesh generation code and by the
    geometry analysis utilities in ``DataBladeAnalysis v8.py``.
    """
    x_raw, y_raw = read_selig_airfoil(path)
    i_le = int(np.argmin(x_raw ** 2 + y_raw ** 2))
    x_ss0 = x_raw[i_le:]
    y_ss0 = y_raw[i_le:]
    x_ps0 = x_raw[:i_le + 1][::-1]
    y_ps0 = y_raw[:i_le + 1][::-1]

    P0 = [x_ss0[-1], y_ss0[-1]]
    P3 = [x_ps0[-1], y_ps0[-1]]
    dist = np.hypot(P3[0] - P0[0], P3[1] - P0[1])
    d = d_factor * dist

    vr = [x_ss0[-2] - P0[0], y_ss0[-2] - P0[1]]
    vs = [x_ps0[-2] - P3[0], y_ps0[-2] - P3[1]]

    CP1, CP2, _, _ = calculate_intersection_point(P0, vr, P3, vs, d)
    curve = create_nurbs_curve(P0, CP1, CP2, P3)
    te_curve = sample_nurbs(curve, n_te)
    mid_idx = n_te // 2
    TE_mid = tuple(te_curve[mid_idx])

    ss_closure = te_curve[:mid_idx + 1]
    ps_closure = te_curve[mid_idx:][::-1]

    x_ss_full = np.concatenate([x_ss0, ss_closure[1:, 0]])
    y_ss_full = np.concatenate([y_ss0, ss_closure[1:, 1]])
    x_ps_full = np.concatenate([x_ps0, ps_closure[1:, 0]])
    y_ps_full = np.concatenate([y_ps0, ps_closure[1:, 1]])

    x_ss, y_ss, s_ss, sn_ss = resample_side(x_ss_full, y_ss_full, n_points)
    x_ps, y_ps, s_ps, sn_ps = resample_side(x_ps_full, y_ps_full, n_points)

    return {
        'ss': (x_ss, y_ss, s_ss, sn_ss),
        'ps': (x_ps, y_ps, s_ps, sn_ps),
        'TE': TE_mid,
        'te_open_thickness': dist,
    }


def compute_d_factor(wedge_angle_deg: float,
                     axial_chord: float,
                     te_thickness: float,
                     *,
                     ref_angle: float = 10.0,
                     ref_offset: float = 0.005,
                     clamp: tuple[float, float] = (0.2, 1.0)) -> float:
    """Empirical trailing‑edge control‑point factor used by meshing."""
    if te_thickness <= 1.0e-8 or wedge_angle_deg <= 0.0:
        return 0.5
    d_f = ref_offset * axial_chord / te_thickness * (ref_angle / wedge_angle_deg)
    d_f = max(clamp[0], min(clamp[1], d_f))
    return d_f


def fit_circle(x, y):
    """Least‑squares circle fit used inside :func:`compute_geometry`."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r


def compute_geometry(path_to_airfoil,
                     pitch,
                     n_points=1000,
                     n_te=60,
                     d_factor_guess=0.5,
                     frac_fit=0.02,
                     n_circle=30):
    """Return a dictionary with basic blade geometric quantities."""
    geo = process_airfoil_file(path_to_airfoil,
                               n_points=n_points,
                               n_te=n_te,
                               d_factor=d_factor_guess)

    xSS, ySS, sSS, _ = geo['ss']
    xPS, yPS, sPS, _ = geo['ps']
    TE = np.array(geo['TE'])

    if sSS is None:
        sSS = np.insert(np.cumsum(np.hypot(np.diff(xSS), np.diff(ySS))), 0, 0)
        sPS = np.insert(np.cumsum(np.hypot(np.diff(xPS), np.diff(yPS))), 0, 0)

    sSS /= sSS[-1]
    sPS /= sPS[-1]

    cssx = CubicSpline(sSS, xSS);  cssy = CubicSpline(sSS, ySS)
    cspx = CubicSpline(sPS, xPS);  cspy = CubicSpline(sPS, yPS)

    s_cmb = np.linspace(0.0, 1.0, n_points)
    xcmb = 0.5 * (cssx(s_cmb) + cspx(s_cmb))
    ycmb = 0.5 * (cssy(s_cmb) + cspy(s_cmb))

    dxcmb = 0.5 * (cssx(s_cmb, 1) + cspx(s_cmb, 1))
    dycmb = 0.5 * (cssy(s_cmb, 1) + cspy(s_cmb, 1))

    LE = np.array([xcmb[0], ycmb[0]])
    chord_vec = TE - LE
    chord_len = np.hypot(*chord_vec)
    axial_chord = chord_vec[0]
    stagger_ang = np.degrees(np.arctan2(chord_vec[1], chord_vec[0]))

    n_fit = max(5, int(frac_fit * n_points))
    A_in = np.c_[xcmb[:n_fit], np.ones(n_fit)]
    m_in, _ = np.linalg.lstsq(A_in, ycmb[:n_fit], rcond=None)[0]
    alpha_in = np.degrees(np.arctan(m_in))

    A_out = np.c_[xcmb[-n_fit:], np.ones(n_fit)]
    m_out, _ = np.linalg.lstsq(A_out, ycmb[-n_fit:], rcond=None)[0]
    alpha_out = np.degrees(np.arctan(m_out))

    dxSS_T, dySS_T = cssx(1.0, 1), cssy(1.0, 1)
    dxPS_T, dyPS_T = cspx(1.0, 1), cspy(1.0, 1)
    ang_SS = np.arctan2(dySS_T, dxSS_T)
    ang_PS = np.arctan2(dyPS_T, dxPS_T)
    outlet_wedge_angle = np.degrees(abs(ang_SS - ang_PS))

    v_to_SS = np.vstack([cssx(s_cmb) - cspx(s_cmb),
                         cssy(s_cmb) - cspy(s_cmb)]).T
    tangents = np.vstack([dxcmb, dycmb]).T
    n_hat = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    n_hat /= np.linalg.norm(n_hat, axis=1)[:, None]
    thickness = (v_to_SS * n_hat).sum(axis=1)
    max_thickness = thickness.max()
    x_tmax, y_tmax = xcmb[thickness.argmax()], ycmb[thickness.argmax()]

    idx_le = np.arange(n_circle)
    idx_te = np.arange(-n_circle, 0)
    x_fit_LE = np.concatenate([xSS[idx_le], xPS[idx_le]])
    y_fit_LE = np.concatenate([ySS[idx_le], yPS[idx_le]])
    _, _, le_radius = fit_circle(x_fit_LE, y_fit_LE)

    x_fit_TE = np.concatenate([xSS[idx_te], xPS[idx_te]])
    y_fit_TE = np.concatenate([ySS[idx_te], yPS[idx_te]])
    _, _, te_radius = fit_circle(x_fit_TE, y_fit_TE)

    arc_SS = np.hypot(np.diff(xSS), np.diff(ySS)).sum()
    arc_PS = np.hypot(np.diff(xPS), np.diff(yPS)).sum()

    geom = dict(
        pitch=pitch,
        axial_chord=axial_chord,
        chord_length=chord_len,
        stagger_angle=np.radians(stagger_ang),
        metal_inlet=np.radians(alpha_in),
        metal_outlet=np.radians(alpha_out),
        wedge_angle=np.radians(outlet_wedge_angle),
        te_radius=te_radius,
        max_thickness=max_thickness,
        x_tmax=x_tmax,
        y_tmax=y_tmax,
        s_camber=s_cmb,
        x_camber=xcmb,
        y_camber=ycmb,
        ss=(xSS, ySS),
        ps=(xPS, yPS),
        TE=TE,
        arc_SS=arc_SS,
        arc_PS=arc_PS,
        te_open_thickness=geo['te_open_thickness'],
    )

    return geom
