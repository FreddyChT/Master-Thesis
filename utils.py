import numpy as np
import math
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from pathlib import Path
import subprocess
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.gp import gp_Pnt
from math import log10, sqrt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}"
})


# ──────────────────────────────────────────────────────────────────────────────
# File Validation Helper
# ──────────────────────────────────────────────────────────────────────────────

def file_nonempty(path: Path) -> bool:
    """Return ``True`` if *path* exists and has a non-zero size."""
    p = Path(path)
    try:
        if p.is_file() and p.stat().st_size > 0:
            return True
    except OSError:
        pass
    print(f"[WARNING] Missing or empty file: {p}")
    return False

# ────────────────────────────────────────────────────────────────────────────────
# File reading and extraction
# ────────────────────────────────────────────────────────────────────────────────
def extract_from_ises(file_path):
    #Opens the given file, skips header lines, and extracts the first numerical value from the next line.
    with open(file_path, 'r') as f:
        # Skip the first two lines to extract Alpha1
        next(f)
        next(f)
        # Read the next line and split into tokens
        line = f.readline()
        tokens = line.split()
        # Extract the third token (index 2) and convert it to a float
        alpha1 = np.float64(tokens[2])
        
        # Read the next line and split into tokens
        line = f.readline()
        tokens = line.split()
        # Extract the third token (index 2) and convert it to a float
        M2 = np.float64(tokens[0])
        P2_P0a = np.float64(tokens[1])
        alpha2 = np.float64(tokens[2])
        
        # Skip to next lines to extract Reynolds No.
        next(f)
        #Read and extract first token
        line = f.readline()
        tokens = line.split()
        reynolds = np.float64(tokens[0])
        
        print("Inlet flow angle (deg):", int(np.degrees(np.arctan(alpha1))))
        print("Outlet flow angle (deg):", int(np.degrees(np.arctan(alpha2))))
        print("Reynolds number:", reynolds)
        print("Outlet Mach:", M2)
        print("Outlet Pressure Ratio:", P2_P0a)
        
    return alpha1, alpha2, reynolds, M2, P2_P0a

def extract_from_blade(file_path):
    #Opens the given file, skips header lines, and extracts the first numerical value from the next line.
    with open(file_path, 'r') as f:
        # Skip the first line to extract Pitch value
        next(f)
        # Read the next line and split into tokens
        line = f.readline()
        tokens = line.split()
        # Extract the third token (index 2) and convert it to a float
        pitch = np.float64(tokens[4])
        
        print("Blade pitch:", pitch)
        
    return pitch

def extract_from_outlet(file_path):
    #Opens the given file, skips header lines, and extracts the first numerical value from the next line.
    with open(file_path, 'r') as f:
        # Skip the first line to extract Pitch value
        for _ in range(19):
            next(f)
        # Read the next line and split into tokens
        line = f.readline()
        tokens = line.split()
        # Extract the third token (index 2) and convert it to a float
        M1_ref = np.float64(tokens[2])
        
        for _ in range(3):
            next(f)
        # Read the next line and split into tokens
        line = f.readline()
        tokens = line.split()
        # Extract the third token (index 2) and convert it to a float
        P21_ratio = np.float64(tokens[2])
        
        print("Inlet Mach number:", M1_ref)
        print("P2/P1:", P21_ratio)
        
    return M1_ref, P21_ratio

def read_selig_airfoil(path):
    x, y = [], []
    with open(path, 'r') as f:
        next(f); next(f)  # skip header
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2: continue
            x.append(float(toks[0])); y.append(float(toks[1]))
    return np.array(x), np.array(y)

#Data Blade Validation file creation
def copy_blade_file(original_filename, blade_dir):
    original_filepath = blade_dir / original_filename
    new_filename = original_filename + ".databladeValidation"     # Construct the new file name
    new_filepath = blade_dir / new_filename
    shutil.copyfile(original_filepath, new_filepath) # Copy the file
    #print(f"Copied '{original_filename}' to '{new_filename}'.")


# ────────────────────────────────────────────────────────────────────────────────
# NURBS-based TE closure
# ────────────────────────────────────────────────────────────────────────────────
def calculate_intersection_point(P0, vr, P3, vs, d):
    """
    Promote 2D→3D, compute control points CP1 & CP2 for C2-continuous closure.
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

    P2    = P0 + d * n
    P2_P3 = P3 + d * n

    CP1 = P0 + (np.dot(P2 - P0, n) / np.dot(vr, n)) * vr
    CP2 = P3 + (np.dot(P2_P3 - P3, n) / np.dot(vs, n)) * vs

    return CP1, CP2, P2, P2_P3

def create_nurbs_curve(P0, CP1, CP2, P3, weights=None):
    """
    Build degree-4 B-Spline through [P0, CP1, midpoint, CP2, P3].
    Automatically promotes any 2D inputs to 3D.
    """
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
        weights = [1.0]*5
    w_arr = TColStd_Array1OfReal(1, 5)
    for i, w in enumerate(weights, start=1):
        w_arr.SetValue(i, w)

    knots = TColStd_Array1OfReal(1, 2)
    knots.SetValue(1, 0.0); knots.SetValue(2, 1.0)
    mults = TColStd_Array1OfInteger(1, 2)
    mults.SetValue(1, 5); mults.SetValue(2, 5)

    return Geom_BSplineCurve(cps, w_arr, knots, mults, 4)

def sample_nurbs(curve, n_te):
    ts = np.linspace(0.0, 1.0, n_te)
    pts = []
    for t in ts:
        p = curve.Value(t)
        pts.append((p.X(), p.Y()))
    return np.array(pts)


# ────────────────────────────────────────────────────────────────────────────────
# Airfoil resampling via arc-length + cubic spline
# ────────────────────────────────────────────────────────────────────────────────
def resample_side(x, y, n_pts):
    dx = np.diff(x); dy = np.diff(y)
    s  = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    csx, csy = CubicSpline(s, x), CubicSpline(s, y)
    s_r = np.linspace(0.0, s[-1], n_pts)
    return csx(s_r), csy(s_r), s_r, s_r / s_r[-1]

def process_airfoil_file(path, n_points=1000, n_te=60, d_factor=0.5):
    """
    Return closed & resampled blade geometry.

    ss / ps : (x, y, s, s_norm) each with n_points samples, LE→TE
    TE      : (x, y) midpoint of the NURBS closure
    """

    # ---------------- 1) read raw Selig file -------------------------------
    x_raw, y_raw = read_selig_airfoil(path)

    # identify LE (closest to origin)
    i_le = int(np.argmin(x_raw**2 + y_raw**2))
    x_ss0 = x_raw[i_le:]          # suction side, LE→original TE
    y_ss0 = y_raw[i_le:]
    x_ps0 = x_raw[:i_le+1][::-1]  # pressure side, LE→original TE
    y_ps0 = y_raw[:i_le+1][::-1]

    # ---------------- 2) close TE with supplied d_factor -------------------
    P0 = [x_ss0[-1], y_ss0[-1]]         # last point on SS  (= TE-SS)
    P3 = [x_ps0[-1], y_ps0[-1]]         # last point on PS  (= TE-PS)
    dist = np.hypot(P3[0]-P0[0], P3[1]-P0[1])
    d    = d_factor * dist              # ✱no internal call to compute_d_factor✱

    vr = [x_ss0[-2]-P0[0], y_ss0[-2]-P0[1]]
    vs = [x_ps0[-2]-P3[0], y_ps0[-2]-P3[1]]

    CP1, CP2, _, _ = calculate_intersection_point(P0, vr, P3, vs, d)
    curve          = create_nurbs_curve(P0, CP1, CP2, P3)
    te_curve       = sample_nurbs(curve, n_te)
    mid_idx        = n_te // 2
    TE_mid         = tuple(te_curve[mid_idx])    # ✱unique TE point✱

    # split the closure into two halves
    ss_closure = te_curve[:mid_idx+1]            # P0 → midpoint
    ps_closure = te_curve[mid_idx:][::-1]        # P3 → midpoint (reversed)

    # build full sides  (LE→midpoint)
    x_ss_full = np.concatenate([x_ss0, ss_closure[1:, 0]])
    y_ss_full = np.concatenate([y_ss0, ss_closure[1:, 1]])
    x_ps_full = np.concatenate([x_ps0, ps_closure[1:, 0]])
    y_ps_full = np.concatenate([y_ps0, ps_closure[1:, 1]])

    # ---------------- 3) resample each side --------------------------------
    x_ss, y_ss, s_ss, sn_ss = resample_side(x_ss_full, y_ss_full, n_points)
    x_ps, y_ps, s_ps, sn_ps = resample_side(x_ps_full, y_ps_full, n_points)

    return {'ss': (x_ss, y_ss, s_ss, sn_ss),
            'ps': (x_ps, y_ps, s_ps, sn_ps),
            'TE': TE_mid,
            'te_open_thickness': dist}

def compute_d_factor(wedge_angle_deg: float,
                     axial_chord: float,
                     te_thickness: float,
                     *,
                     ref_angle: float = 10.0,
                     ref_offset: float = 0.005,
                     clamp: tuple[float, float] = (0.2, 1.0)
                     ) -> float:
    """
    Empirical law F-1  ▸  returns a trailing-edge control-point factor
        d_factor = 0.005 c_a / t_te · (10° / θ)

    Parameters
    ----------
    wedge_angle_deg : float   -- metal (wedge) angle θ   [deg]
    axial_chord     : float   -- c_a                     [same unit as te_thickness]
    te_thickness    : float   -- open TE thickness t_te  [same unit as axial_chord]
    ref_angle       : float   -- reference angle in formula (default 10°)
    ref_offset      : float   -- reference offset fraction of c_a (default 0.5 % → 0.005)
    clamp           : (min,max) tuple to stop extreme values

    Returns
    -------
    d_factor : float   -- value to pass to `process_airfoil_file`
    """
    if te_thickness <= 1.0e-8 or wedge_angle_deg <= 0.0:
        return 0.5                                            # safe fallback

    d_f = ref_offset * axial_chord / te_thickness * (ref_angle / wedge_angle_deg)
    d_f = max(clamp[0], min(clamp[1], d_f))                   # keep in practical band
    return d_f


# ────────────────────────────────────────────────────────────────────────────────
# Blade Geometry
# ────────────────────────────────────────────────────────────────────────────────
#  Helper – least–squares circle fit  (Taubin 1991 algebraic form)
def fit_circle(x, y):
    """
    Return center (xc, yc) and radius r that best fit the N points (x, y)
    in a least–squares sense.
    """
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)     # [A]·[xc, yc, c0] = b
    xc, yc = c[0], c[1]
    r  = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def compute_geometry(path_to_airfoil,
                     pitch,
                     n_points=1000,
                     n_te=60,
                     d_factor_guess=0.5,
                     frac_fit=0.02,
                     n_circle=30):
    """
    Return a dict with geometric quantities for an LPT blade.

    Parameters
    ----------
    path_to_airfoil : str   – .dat or .csv file (same formats as before)
    pitch           : float – pitch length (passed through unchanged)
    n_points        : int   – samples per surface after resampling (default 1000)
    n_te, d_factor_guess    – forwarded to `process_airfoil_file`
    frac_fit        : float – % of camber (at LE / TE) used in metal-angle fits
    n_circle        : int   – number of points per side used in LE / TE circle fits
    """
    # ------------------------------------------------ 1) closed geometry
    geo = process_airfoil_file(path_to_airfoil,
                               n_points=n_points,
                               n_te=n_te,
                               d_factor=d_factor_guess)

    xSS, ySS, sSS, _ = geo['ss']          # suction side   (LE→TE)
    xPS, yPS, sPS, _ = geo['ps']          # pressure side  (LE→TE)
    TE               = np.array(geo['TE'])

    # Ensure arc-length arrays exist (resample_side already returns them;
    # if you read .csv the value might be None → compute quickly)
    if sSS is None:
        sSS = np.insert(np.cumsum(np.hypot(np.diff(xSS), np.diff(ySS))), 0, 0)
        sPS = np.insert(np.cumsum(np.hypot(np.diff(xPS), np.diff(yPS))), 0, 0)

    # normalised s for each surface
    sSS /= sSS[-1]
    sPS /= sPS[-1]

    # ------------------------------------------------ 2) parametric cubic splines (x(s), y(s))
    cssx = CubicSpline(sSS, xSS);  cssy = CubicSpline(sSS, ySS)
    cspx = CubicSpline(sPS, xPS);  cspy = CubicSpline(sPS, yPS)

    # common camber parameter s  (0 → LE, 1 → TE)
    s_cmb = np.linspace(0.0, 1.0, n_points)

    xcmb = 0.5*(cssx(s_cmb) + cspx(s_cmb))
    ycmb = 0.5*(cssy(s_cmb) + cspy(s_cmb))

    # first derivatives wrt param s
    dxSS, dySS = cssx(sSS, 1), cssy(sSS, 1)
    dxPS, dyPS = cspx(sPS, 1), cspy(sPS, 1)
    dxcmb      = 0.5*(cssx(s_cmb, 1)+cspx(s_cmb, 1))
    dycmb      = 0.5*(cssy(s_cmb, 1)+cspy(s_cmb, 1))

    # ------------------------------------------------ 3) basic lengths & angles
    LE   = np.array([xcmb[0], ycmb[0]])
    chord_vec   = TE - LE
    chord_len   = np.hypot(*chord_vec)
    axial_chord = chord_vec[0]
    stagger_ang = np.degrees(np.arctan2(chord_vec[1], chord_vec[0]))

    # inlet / outlet metal angles from least-squares line fits
    n_fit = max(5, int(frac_fit*n_points))

    # inlet (first n_fit camber points)
    A_in  = np.c_[xcmb[:n_fit], np.ones(n_fit)]
    m_in, c_in = np.linalg.lstsq(A_in, ycmb[:n_fit], rcond=None)[0]
    alpha_in   = np.degrees(np.arctan(m_in))

    # outlet (last n_fit camber points)
    A_out = np.c_[xcmb[-n_fit:], np.ones(n_fit)]
    m_out, c_out = np.linalg.lstsq(A_out, ycmb[-n_fit:], rcond=None)[0]
    alpha_out  = np.degrees(np.arctan(m_out))

    # wedge angle (difference of surface tangents at TE)
    # evaluate at s=1 exactly
    dxSS_T, dySS_T = cssx(1.0, 1), cssy(1.0, 1)
    dxPS_T, dyPS_T = cspx(1.0, 1), cspy(1.0, 1)
    ang_SS = np.arctan2(dySS_T, dxSS_T)
    ang_PS = np.arctan2(dyPS_T, dxPS_T)
    outlet_wedge_angle = np.degrees(abs(ang_SS - ang_PS))  # positive value

    # ------------------------------------------------ 4) thickness normal to camber
    # vector from PS→SS at every camber station and camber tangents
    v_to_SS = np.vstack([cssx(s_cmb)-cspx(s_cmb),
                         cssy(s_cmb)-cspy(s_cmb)]).T    # (n,2)
    tangents = np.vstack([dxcmb, dycmb]).T
    # unit normals (rotate tangents 90° CCW and normalise)
    n_hat = np.column_stack([-tangents[:,1], tangents[:,0]])
    n_hat /= np.linalg.norm(n_hat, axis=1)[:,None]
    thickness = (v_to_SS * n_hat).sum(axis=1)
    max_thickness = thickness.max()
    x_tmax, y_tmax = xcmb[thickness.argmax()], ycmb[thickness.argmax()]

    # ------------------------------------------------ 5) LE / TE radii via circle fit
    # grab points around LE and TE on EACH surface, concatenate
    idx_le  = np.arange(n_circle)
    idx_te  = np.arange(-n_circle, 0)

    x_fit_LE = np.concatenate([xSS[idx_le], xPS[idx_le]])
    y_fit_LE = np.concatenate([ySS[idx_le], yPS[idx_le]])
    _, _, le_radius = fit_circle(x_fit_LE, y_fit_LE)

    x_fit_TE = np.concatenate([xSS[idx_te], xPS[idx_te]])
    y_fit_TE = np.concatenate([ySS[idx_te], yPS[idx_te]])
    _, _, te_radius = fit_circle(x_fit_TE, y_fit_TE)

    # surface lengths (LE→TE)  – useful for later mesh spacing
    arc_SS = np.hypot(np.diff(xSS), np.diff(ySS)).sum()
    arc_PS = np.hypot(np.diff(xPS), np.diff(yPS)).sum()
    
    # ------------------------------------------------ 6) Print computed values
    #print("Stagger angle (deg):", stagger_ang)
    #print("Inlet metal angle (deg):", alpha_in)
    #print("Outlet metal angle (deg):", alpha_out)
    #print("Chord length:", chord_len)
    #print("Axial chord (x-span):", axial_chord)
    #print("LE radius:", le_radius)
    #print("TE radius:", te_radius)
    #print("Outlet wedge angle (deg):", outlet_wedge_angle)
    
    # ------------------------------------------------ 7) pack & return
    geom = dict(pitch         = pitch,
                axial_chord   = axial_chord,
                chord_length  = chord_len,
                stagger_angle = np.radians(stagger_ang),
                metal_inlet   = np.radians(alpha_in),
                metal_outlet  = np.radians(alpha_out),
                wedge_angle   = np.radians(outlet_wedge_angle),
                te_radius     = te_radius,
                max_thickness = max_thickness,
                x_tmax        = x_tmax,
                y_tmax        = y_tmax,
                s_camber      = s_cmb,
                x_camber      = xcmb,
                y_camber      = ycmb,
                ss = (xSS, ySS),
                ps = (xPS, yPS),
                TE = TE,
                arc_SS = arc_SS,
                arc_PS = arc_PS,
                te_open_thickness = geo['te_open_thickness'])
    
    return geom

def surface_fraction(xvals, yvals):
    """
    Normalizes arc lengths to get xSSnorm, xPSnorm for plotting on [0..1].
    """
    dx = np.diff(xvals)
    dy = np.diff(yvals)
    seg = np.sqrt(dx**2 + dy**2)
    arc = np.cumsum(seg)
    arc = np.insert(arc, 0, 0.0)
    if arc[-1] != 0:
        return arc / arc[-1]
    else:
        return arc


# ────────────────────────────────────────────────────────────────────────────────
# Boundary Layer Functions
# ────────────────────────────────────────────────────────────────────────────────

# Boundary-layer sizing helpers

def first_cell_height_yplus_1(U_inf: float,
                              rho:   float,
                              mu:    float,
                              y_plus_target: float = 1.0,
                              L_ref: float     = 1.0,
                              ) -> float:
    """
    Returns y₁ so that y⁺ = y_plus_target at a reference location *L_ref*
    downstream of the leading edge (use ≈ 0.02–0.05 c for conservative sizing).
    """
    Re_x = rho * U_inf * L_ref / mu
    Cf   = (2.0 * log10(Re_x) - 0.65) ** -2.3   # For Re < 10^9
    tau_w  = 0.5 * Cf * rho * U_inf**2     # wall shear stress
    u_tau  = np.sqrt(tau_w / rho)             # friction velocity
    y1     = y_plus_target * mu / (rho * u_tau)
    return y1


def bl_thickness_flat_plate(U_inf: float,
                            rho: float,
                            mu: float,
                            x: float,
                            Re_transition: float = 5.0e5
                            ) -> float:
    """
    Classical flat-plate δ₉₉ correlations.
      – laminar (Blasius): δ = 5 x / √Re_x
      – turbulent 1/7-power: δ = 0.37 x / Re_x⁰·²
    """
    Re_x = rho * U_inf * x / mu
    if Re_x <= Re_transition:                       # laminar
        return 5.0 * x / sqrt(Re_x)
    return 0.37 * x / (Re_x ** 0.2)                 # turbulent


def bl_growth_ratio(n_layers: int, y1: float, delta: float) -> float:
    """
    Solves   δ = y₁ (rⁿ – 1)/(r – 1)     for the geometric ratio r.
    Uses Newton iteration – usually converges in <6 steps for 1 < r < 1.4.
    """
    if n_layers < 2:
        raise ValueError("Need at least 2 layers to determine a growth ratio.")

    def f(r):        # residual
        return y1 * (r**n_layers - 1.0) / (r - 1.0) - delta

    def df(r):       # derivative
        return y1 * (
            (n_layers * r**(n_layers - 1) * (r - 1.0) -
             (r**n_layers - 1.0)) / (r - 1.0)**2
        )

    r = 1.2  # good initial guess
    for _ in range(20):
        r_new = r - f(r) / df(r)
        if abs(r_new - r) < 1e-6:
            return r_new
        r = r_new
    raise RuntimeError("Growth-ratio solver did not converge.")


# Convenience wrapper that gives everything Gmsh wants

def compute_bl_parameters(U_inf: float,
                          rho: float,
                          mu: float,
                          chord_axial: float,
                          *,
                          n_layers: int       = 25,
                          y_plus_target: float = 1.0,
                          x_ref_yplus: float   = 0.02,
                          x_ref_delta: float   = 1.0   # TE ≈ 1 cₐ
                          ):
    """
    Returns a dict with:
        first_layer_height  → Field[1].hwall_n
        bl_growth           → Field[1].ratio
        bl_thickness        → Field[1].thickness
    All lengths are in meters, ready to inject into the *.geo* template.
    Parameters
    ----------
    U_inf, rho, mu : inlet freestream conditions
    chord_axial    : axial chord [m]
    n_layers       : number of prism layers you intend to use
    y_plus_target  : your y⁺ goal (default = 1)
    x_ref_yplus    : streamwise station for y⁺ sizing,
                     given as a multiple of *cₐ* (default 0.02 cₐ)
    x_ref_delta    : station for δ₉₉ (default 1 cₐ → near TE)
    """
    x_yplus = x_ref_yplus * chord_axial
    x_delta = x_ref_delta * chord_axial

    y1 = first_cell_height_yplus_1(U_inf, rho, mu,
                                   y_plus_target=y_plus_target,
                                   L_ref=x_yplus)
    delta = bl_thickness_flat_plate(U_inf, rho, mu, x_delta)
    r = bl_growth_ratio(n_layers, y1, delta)

    return dict(first_layer_height=y1,
                bl_growth=r,
                bl_thickness=delta)


# ────────────────────────────────────────────────────────────────────────
#  Boundary-layer integrals along an outward normal
# ────────────────────────────────────────────────────────────────────────

def _normal_at_surface_point(x_surf, y_surf, x_prev, y_prev, x_next, y_next):
    """Unit normal pointing outside the blade (2-D)."""
    # tangent = next − prev   (already LE→TE ordering)
    tx, ty = x_next - x_prev, y_next - y_prev
    # outward = (+ty, −tx) for a left-hand (anti-clockwise) contour
    nx, ny =  ty, -tx
    mag = np.hypot(nx, ny)
    return nx/mag, ny/mag


def _bl_integrals(y, rho, u):
    """Return θ, δ*, H given wall-normal profiles (already non-dim)."""
    # ρ_e, U_e at last entry
    rho_e, ue = rho[-1], u[-1]
    f1 = (rho/rho_e)*(u/ue)
    theta      = np.trapz(f1*(1 - u/ue), y)
    delta_star = np.trapz((rho/rho_e)*(1 - u/ue), y)
    H          = delta_star/theta if theta > 0 else np.nan
    return theta, delta_star, H


def bl_distributions(surface_df: "pd.DataFrame",
                     volume_df:  "pd.DataFrame",
                     y_max: float = 0.01,
                     n_samples: int = 50):
    """
    Loop over every surface node and integrate θ,  δ*,  Re_θ,  H.

    Returns
    -------
    dict with arrays keyed by 's', 'Re_theta', 'H', split into SS/PS later.
    """
    from scipy.spatial import cKDTree

    # --- 1) accelerator for nearest-neighbour interpolation ------------------
    vol_xy   = volume_df[['x', 'y']].values
    vol_u    = (volume_df['Momentum_x']**2 +
                volume_df['Momentum_y']**2).pow(0.5).values / volume_df['Density'].values
    vol_rho  = volume_df['Density'].values
    vol_mu   = volume_df['Laminar_Viscosity'].values
    tree = cKDTree(vol_xy)

    # --- 2) prepare outputs --------------------------------------------------
    theta_arr, Re_theta_arr, H_arr = [], [], []
    s_coord = surface_df['s_norm'].values

    xs, ys = surface_df['x'].values, surface_df['y'].values
    for i, (xs_i, ys_i) in enumerate(zip(xs, ys)):
        # tangent neighbours (cyclic indexing)
        ip = (i+1) % len(xs); im = (i-1) % len(xs)
        nx, ny = _normal_at_surface_point(xs_i, ys_i,
                                          xs[im], ys[im], xs[ip], ys[ip])

        # sample points along the normal
        y_local = np.linspace(0.0, y_max, n_samples)
        x_samp  = xs_i + nx * y_local
        y_samp  = ys_i + ny * y_local
        _, idxs = tree.query(np.column_stack([x_samp, y_samp]), k=1)

        u_prof   = vol_u[idxs]
        rho_prof = vol_rho[idxs]
        mu_prof  = vol_mu[idxs]

        # trim at u / u_e >= 0.99
        mask = u_prof / u_prof[-1] < 0.99
        if mask.any():
            cut = np.where(~mask)[0][0] + 1  # include first ≥0.99 point
            y_loc = y_local[:cut];  u_p = u_prof[:cut];  rho_p = rho_prof[:cut]
            mu_e  = mu_prof[cut-1];  # last available μ
        else:
            y_loc, u_p, rho_p = y_local, u_prof, rho_prof
            mu_e = mu_prof[-1]

        θ, δs, H = _bl_integrals(y_loc, rho_p, u_p)
        Re_theta = rho_p[-1]*u_p[-1]*θ / mu_e

        theta_arr.append(θ); Re_theta_arr.append(Re_theta); H_arr.append(H)

    return dict(s=s_coord,
                Re_theta=np.array(Re_theta_arr),
                H=np.array(H_arr))



# ────────────────────────────────────────────────────────────────────────────────
# Computation Helper Functions
# ────────────────────────────────────────────────────────────────────────────────

def compute_Mx(P0x, Px, gamma):
    Mx = np.sqrt( (2/(gamma - 1)) * ((P0x/Px)**((gamma-1)/gamma) - 1) )
    return Mx

def compute_Tx(T0x, Mx, gamma):
    Tx = T0x / (1 + (gamma-1)/2 * Mx**2)
    return Tx

def compute_Vx(Mx, gamma, R, Tx):
    Vx = Mx * np.sqrt(gamma * R * Tx)
    return Vx

def compute_rhox(Px, Tx, R):
    rhox = Px / (R*Tx)
    return rhox

def compute_miux(mu0, T01, Tx, S):
    mux = mu0 * (T01+S)/(Tx+S)*(Tx/T01)**1.5
    return mux

def compute_mux(Tx):
    mux = (1.458e-6 * Tx**1.5) / (Tx + 110.4)
    return mux

def compute_nux(mux, rhox):
    nux = mux / rhox
    return nux

def compute_Rex(rhox, Vx, axial_chord, mux):
    Rex = rhox * Vx * axial_chord / mux
    return Rex

def compute_Losses(P01, P06, P1, P6, gamma):
    Loss_P0 = (P01 - P06) / P01
    Loss_P = (P01 - P06) / (P01 - P1)
    Loss_K = 1 - (1 - (P6/P06)**((gamma-1)/gamma) ) / ( 1 - (P6/P01)**((gamma-1)/gamma) )
    return Loss_P0, Loss_P, Loss_K

def compute_TurbulentQtys(TI, Umean, ILS, nu):              # From  Chapter 3.2.1 Numerical Parameters
    C_mu    = 0.09
    k       = 3/2 * (Umean * TI)**2         #[m2/s2]
    epsilon = C_mu**(3/4) * k**(3/2)/ILS    #[m2/s3]
    omega   = epsilon / k                   #[1/s]          # Note valid under isotropic turbulent conditions
    nu_t = C_mu * k**2 / epsilon            #[m2/s]
    nu_factor = nu_t / nu
    return k, epsilon, omega, nu_factor

def compute_TKE(TI, Umean):
    k = 3/2 * (Umean * TI)**2
    return k
    
def compute_TKE_Dissipation(k, ILS):
    C_mu = 0.09
    epsilon = C_mu**(3/4) * k**(3/2)/ILS
    return epsilon

def compute_Spec_Dissipation(k, ILS):
    C_mu = 0.09
    omega = C_mu**(3/4) * k**(1/2)/ILS
    return omega

def freestream_total_pressure(Re, M, L, T,
                              mu=1.8464e-5,   # dynamic viscosity of air at ~300 K [kg m-1 s-1]
                              gamma=1.4,      # ratio of specific heats for air
                              R=287.058):     # gas constant for air [J kg-1 K-1]
    """
    Returns (p_static, p_total) in Pa.
    """
    a = np.sqrt(gamma * R * T)                 # speed of sound
    rho = Re * mu / (M * L * a)                  # density from Re definition
    p_static = rho * R * T                       # ideal-gas static pressure
    pressure_ratio = (1 + 0.5*(gamma-1)*M**2)**(gamma/(gamma-1))
    p_total = p_static * pressure_ratio          # stagnation pressure
    return p_static, p_total


# --- example with your numbers ---------------------------------------------
Re = 6.0e5      # Reynolds number
M  = 0.5        # Mach number
L  = 0.20       # chord length [m] – change to your model’s chord
T  = 288.0      # static temp [K] – change to test temperature

p_static, p_total = freestream_total_pressure(Re, M, L, T)

print(f"Static pressure : {p_static/1000:.2f} kPa")
print(f"Total pressure  : {p_total/1000:.2f} kPa")


# ─────────────────────────────────────────────────────────────────────────────
#   SU2 Post-Processing
# ─────────────────────────────────────────────────────────────────────────────
def roll_array(arr, shift):
    """
    Rolls the arrays so that idx_maxP is placed at index 0.
    Ensures the suction side starts at the max‐pressure location for easy plotting.
    """
    return np.concatenate([arr[shift:], arr[:shift]])

def SU2_organize(df):
    """
    Reorganizes the surface CSV data from SU2 to separate
    leading_edge, upper_surface, trailing_edge, lower_surface.
    """
    leading_edge  = df.iloc[0:1].copy()      # row 0
    leading_ss    = df.iloc[1:2].copy()      # row 1
    trailing_edge = df.iloc[2:3].copy()      # row 2
    trailing_ss   = df.iloc[3:4].copy()      # row 3
    leading_ps    = df.iloc[4:5].copy()      # row 4
    trailing_ps   = df.iloc[5:6].copy()      # row 5

    geo = df.iloc[6:].copy().reset_index(drop=True)
    x, y = geo['x'].values, geo['y'].values

    # find the largest jump between consecutive points = break TE→LE
    dist = np.hypot(np.diff(x), np.diff(y))
    idx_break = np.argmax(dist) + 1          # first point of pressure surface
    
    upper_surface  = geo.iloc[:idx_break].copy()     # suction side
    lower_surface  = geo.iloc[idx_break:].copy()     # pressure, reversed LE→TE

    return leading_edge, trailing_edge, upper_surface, lower_surface
    
def SU2_extract_plane_data(df, x_plane, pitch, alpha_m, atol=1e-4):
    """
    Extracts data at a given x-plane from the restart SU2 file.
    Normalizes y by pitch.
    """
    columns=['y', 'Density', 'Pressure', 'Velocity_x', 'Velocity_y', 'Mach']
    #columns=['y', 'Density', 'Pressure', 'Velocity_x', 'Velocity_y', 'Mach', 'Turb_Kin_Energy', 'Turb_index']
    
    # Find all rows where x ≈ x_plane (to tolerance)
    mask = np.isclose(df['x'], x_plane, atol=atol)
    if not mask.any():
        print(f"[WARNING] No data found at x = {x_plane} (tol={atol}). Try increasing tolerance.")
        return None
    
    # Extract and organize the data
    sub_df = df.loc[mask, columns + ['x']].copy()
    sub_df['y_norm'] = sub_df['y'] / pitch  # Normalize y
    sub_df['flow_angle'] = np.atan2(sub_df['Velocity_y'], sub_df['Velocity_x']) * 180 / np.pi - alpha_m
    
    # Sort by normalized y for clean plots
    sub_df = sub_df.sort_values('y_norm').reset_index(drop=True)
    return sub_df    

def SU2_DataPlotting(
        sSSnorm,    # suction side arc fraction
        sPSnorm,    # pressure side arc fraction
        dataSS,     # suction side quantity
        dataPS,     # pressure side quantity
        quantity,   # label for the plotted quantity
        string,     # name suffix
        run_dir,
        bladeName,
        mirror_PS=False,
        exp_s=None, # optional experimental x array
        exp_data=None # optional experimental Mach array
    ):
    """
    Plots SU2 results in Non-Norm style (direct values) plus
    optional experimental data for direct comparison.
    """
    # Plot SU2 (suction & pressure side)
    plt.plot(sSSnorm, dataSS, marker='o', markersize=0.5, linestyle='-', color='darkblue', label='SU2 (SS)')
    
    s_ps = -sPSnorm if mirror_PS else sPSnorm
    plt.plot(s_ps, dataPS, marker='o', markersize=0.5, linestyle='-', color='lightblue', label='SU2 (PS)')

    # Overlay optional experimental distribution
    if (exp_s is not None) and (exp_data is not None):
        plt.scatter(exp_s, exp_data, s=0.5, color='red', label='Mises Data')

    plt.ylabel(f'{quantity} - {bladeName}')
    plt.xlabel(r'S/S_{total}')
    #plt.grid(visible=True, color='lightgray')
    if mirror_PS:
        plt.xlim(-1, 1)       # show full mirror
    else:
        plt.xlim(0, 1)
    plt.legend(loc='upper left', edgecolor='k', fancybox=False)
    plt.savefig(run_dir / f"non-normalized_{quantity}_{string}_{bladeName}.svg", format='svg', bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#   MISES Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

def MISES_blDataGather(file_path):
    """Parse ``bl`` output from MISES and organize it by surface.

    The file contains three data chunks separated by blank lines.  The first
    chunk corresponds to the pressure side, the second to the suction side and
    the third (if present) is ignored.  Each chunk is returned as a ``pandas``
    ``DataFrame`` with the same column names used for the SU2 surface data.

    The ``s`` (surface fraction) column of each DataFrame is normalised to
    ``[0, 1]``.
    """

    file_path = Path(file_path)
    if not file_nonempty(file_path):
        return pd.DataFrame(), pd.DataFrame()

    column_names = [
        "x", "y", "s", "b", "Ue/a0", "delta_star", "theta", "theta_star",
        "H", "Hbar", "Cf", "CD", "Rtheta", "M",
    ]

    # Read file and skip the two header lines
    with open(file_path, "r") as f:
        lines = f.readlines()[2:]

    chunks = []
    current = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < len(column_names):
            if current:
                chunks.append(current)
                current = []
            continue
        try:
            row = [float(t) for t in tokens[: len(column_names)]]
        except ValueError:
            continue
        current.append(row)

    if current:
        chunks.append(current)

    # Expect at least pressure and suction side chunks
    if len(chunks) < 2:
        raise ValueError(
            "bl file does not contain the expected pressure and suction data blocks"
        )

    ps_df = pd.DataFrame(chunks[0], columns=column_names)
    ss_df = pd.DataFrame(chunks[1], columns=column_names)

    def normalise_surface(df):
        smin, smax = df["s"].min(), df["s"].max()
        if smax != smin:
            df["s"] = (df["s"] - smin) / (smax - smin)

    normalise_surface(ps_df)
    normalise_surface(ss_df)

    return ps_df, ss_df

def MISES_fieldDataGather(file_path):
    """
    Reads field data from file_path, skipping the first two header lines.
    Columns (0..7) => x, y, rho/rho0, p/p0, u/a0, v/a0, q/a0, M
    Blank lines separate the data into streamtubes.
    Returns all_x, all_y, all_rho, all_p, all_u, all_v, all_q, all_m
    """
    file_path = Path(file_path)
    if not file_nonempty(file_path):
        return [], [], [], [], [], [], [], []

    all_x  = []
    all_y  = []
    all_rho= []
    all_p  = []
    all_u  = []
    all_v  = []
    all_q  = []
    all_m  = []

    # Temporary arrays
    x_tmp, y_tmp = [], []
    rho_tmp, p_tmp = [], []
    u_tmp, v_tmp = [], []
    q_tmp, m_tmp = [], []

    with open(file_path, 'r') as f:
        for _ in range(2):
            next(f, None)

        for line in f:
            tokens = line.split()

            if len(tokens) < 8:
                # End of one streamtube
                if x_tmp:
                    all_x.append(x_tmp)
                    all_y.append(y_tmp)
                    all_rho.append(rho_tmp)
                    all_p.append(p_tmp)
                    all_u.append(u_tmp)
                    all_v.append(v_tmp)
                    all_q.append(q_tmp)
                    all_m.append(m_tmp)
                x_tmp, y_tmp = [], []
                rho_tmp, p_tmp = [], []
                u_tmp, v_tmp = [], []
                q_tmp, m_tmp = [], []
                continue

            x_tmp.append(float(tokens[0]))
            y_tmp.append(float(tokens[1]))
            rho_tmp.append(float(tokens[2]))
            p_tmp.append(float(tokens[3]))
            u_tmp.append(float(tokens[4]))
            v_tmp.append(float(tokens[5]))
            q_tmp.append(float(tokens[6]))
            m_tmp.append(float(tokens[7]))

    # Catch final tube if no trailing blank
    if x_tmp:
        all_x.append(x_tmp)
        all_y.append(y_tmp)
        all_rho.append(rho_tmp)
        all_p.append(p_tmp)
        all_u.append(u_tmp)
        all_v.append(v_tmp)
        all_q.append(q_tmp)
        all_m.append(m_tmp)

    return all_x, all_y, all_rho, all_p, all_u, all_v, all_q, all_m

def MISES_machDataGather(file_path):
    """
    Reads field data from file_path, skipping the first two header lines.
    Columns (0,1) => s, M
    Blank lines separate the data into upper and lower surface.
    Returns blade_frac, blade_mach
    """
    file_path = Path(file_path)
    if not file_nonempty(file_path):
        return np.array([]), np.array([]), np.array([]), np.array([])

    #We extract the MISES surface infromation in lists for upper and lower surfaces
    with open(file=file_path, mode='r') as f:
        next(f)
        _ = f.readline().split()[0]
        next(f)
        next(f)
        lines = f.readlines()
        upperSurf = []
        lowerSurf = []
        endupper  = False
        # The following code takes into account the machDistribution file and
        for line in lines:
            if not endupper:
                values = line.split()
                try:
                    xPos      = np.float64(values[0])
                    dataValue = np.float64(values[1])
                    upperSurf.append([xPos, dataValue]) #This saves a list of vectors [x,y]
                except:
                    endupper = True
            else:
                values = line.split()
                try:
                    xPos      = np.float64(values[0])
                    dataValue = np.float64(values[1])
                    lowerSurf.append([xPos, dataValue]) #This saves a list of vectors [x,y]
                except:
                    pass
    
    #We now modify the extracted lists into 2D-arrays
    upperValues = np.zeros((len(upperSurf),2))
    for ii, values in enumerate(upperSurf):
        upperValues[ii, 0] = values[0] #This saves a 2D-array of the list contents in the upper surface
        upperValues[ii, 1] = values[1] 
    lowerValues = np.zeros((len(lowerSurf), 2))
    for ii, values in enumerate(lowerSurf):
        lowerValues[ii, 0] = values[0] #This saves a 2D-array of the list contents in the lower surface
        lowerValues[ii, 1] = values[1] 
    
    #We partition the arrays for plotting purposes
    ps_frac    = upperValues[:, 0] #Upper values are pressure side
    ps_mach    = upperValues[:, 1]
    ss_frac    = lowerValues[:, 0] #Lower values are suction side
    ss_mach    = lowerValues[:, 1]
    
    blade_frac = np.concatenate([ps_frac, ss_frac])
    blade_mach = np.concatenate([ps_mach, ss_mach])
    
    return(ps_frac, ss_frac, ps_mach, ss_mach,)

def MISES_DataGather(data, xNorm, y, n):
    index_closest_to_zero = np.abs(data - max(data)).argmin() #Finds the index where the pressure value is closest to pmax (argmin used since the abs difference is an array)
    xSS = xNorm[index_closest_to_zero:]
    xSS = np.concatenate((xSS, xNorm[:n*3]))
    ySS = y[index_closest_to_zero:]
    ySS = np.concatenate((ySS, y[:n*3]))
    dataSS = data[index_closest_to_zero:]
    dataSS = np.concatenate((dataSS, data[:n*3]))
    dataSS = savgol_filter(dataSS, window_length=15, polyorder=3) #Smooth out the mach number data
    
    #X and Y pressure side values organizing to obtain mach numbers for pressure side
    xPS = xNorm[index_closest_to_zero:n*7-3:-1]
    xPS = np.concatenate((xPS, xNorm[n*7-3:n*4-2:-1]))
    yPS = y[index_closest_to_zero:n*7-3:-1]
    yPS = np.concatenate((yPS, y[n*7-3:n*4-2:-1]))
    dataPS = data[index_closest_to_zero:n*7-3:-1]
    dataPS = np.concatenate((dataPS, data[n*7-3:n*4-2:-1]))
    dataPS = savgol_filter(dataPS, window_length=15, polyorder=3) #Smooth out the mach number data
    
    #Normalization of Suction Side x-component
    dxSS = np.diff(xSS)
    dySS = np.diff(ySS)
    segment_lengthsSS = np.sqrt(dxSS**2 + dySS**2)
    lengths_cumulativeSS = np.cumsum(segment_lengthsSS) #Array where each element is the sum of all previous segment lengths. Needed to normalize
    xSSnorm = lengths_cumulativeSS/lengths_cumulativeSS[-1] #Suction side x component normalization
    
    #Normalization of Pressure Side x-component
    dxPS = np.diff(xPS)
    dyPS = np.diff(yPS)
    segment_lengthsPS = np.sqrt(dxPS**2 + dyPS**2)
    lengths_cumulativePS = np.cumsum(segment_lengthsPS)
    xPSnorm = lengths_cumulativePS/lengths_cumulativePS[-1]
    
    #Suction side trailing edge mach number
    dataSSTrial = data[index_closest_to_zero:]
    dataSSTrial = np.concatenate((dataSSTrial, data[:n*3]))
    dataSSTE = dataSSTrial[-1]
    
    return(xSSnorm, xPSnorm, dataSS, dataPS, dataSSTE)



# ─────────────────────────────────────────────────────────────────────────────
#   Paraview Integration
# ─────────────────────────────────────────────────────────────────────────────

def launch_paraview_live(run_dir, bladeName, suffix):
    """Launch the Paraview live visualization script."""
    script_path = Path(__file__).resolve().parent / 'liveParaview_datablade.py'
    os.environ["PATH"] = ";".join([*os.environ["PATH"].split(";"), "C:\\Program Files\\ParaView-5.12.0-MPI-Windows-Python3.10-msvc2017-AMD64\\bin"])
    pvpython = shutil.which('pvpython') or shutil.which('pvpython.exe')
    if pvpython is None:
        raise FileNotFoundError('pvpython executable not found')
    subprocess.Popen([
        pvpython,
        str(script_path),
        str(run_dir),
        bladeName,
        suffix,
    ])