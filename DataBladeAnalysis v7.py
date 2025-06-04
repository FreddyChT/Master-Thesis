1# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:54:39 2025

@author: Freddy Chica
@co-author: Francesco Porta

Notice: GPT-4o was heavily used for the elaboration of this script
"""

import numpy as np
import os
import shutil
import math
import sys
import gmsh
import meshio
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D  
import openpyxl                                    # Excel reader
# pyOCC imports
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.gp import gp_Pnt


#####################################################################################
#                                                                                   #
#                                  INITIALIZATION                                   #
#                                                                                   #
##################################################################################### 


# You will need the following files to run this analysis:
# - Ises file
# - Blade file
# - Gridpar file
# - Mach Distribution file
# Others to be determined when checking other files

# --------------------------- FILE SELECTION ---------------------------

bladeName = "blade" #Change this name depending on the blade you want to study
no_cores = 12 # Change this to switch the processing power of the computation (numbers of used cores)
string = "databladeVALIDATION" # File names
string2 = "safe_start" # For SU2 optimization
fileExtension = "csv"

current_directory = os.path.dirname(os.path.abspath(__file__))
isesFileName = f"ises.{string}"
bladeFileName = f"{bladeName}.{string}"
outletFileName = f"outlet.{string}"
isesFilePath = os.path.join(current_directory, isesFileName)
bladeFilePath = os.path.join(current_directory, bladeFileName)
outletFilePath = os.path.join(current_directory, outletFileName)


# --------------------------- HELPER FUNCTIONS ---------------------------

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
        
    return alpha1, alpha2, reynolds

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

def extract_airfoil(file_path):
    # Initialize empty lists to store x and y values
    x_values, y_values = [], []
    with open(file_path, 'r') as f:
        # Skip the first two lines
        for _ in range(2):
            next(f)
        for line in f:
            # Split each line into parts based on whitespace
            parts = line.split()
            # Extract the first two values as floats
            x, y = map(float, parts[:2])
            # Append values to the lists
            x_values.append(x)
            y_values.append(y)
            
    return np.array(x_values), np.array(y_values)

def compute_geometry(file_path, n_points=200, le_radius=0.035, te_radius=0.035):
    # --- Step 1: Extract blade points ---
    x_blade, y_blade = extract_airfoil(file_path)
    
    # --- Step 2: Determine LE and split surfaces ---
    # Use a tolerance to check if x and y are zero
    idx_LE = np.argmin( x_blade**2 + y_blade**2 )
    LE   = np.array([x_blade[idx_LE], y_blade[idx_LE]])
    
    # Upper surface: points BEFORE the LE (excluding the LE point)
    upper_x = x_blade[:idx_LE+1]
    upper_y = y_blade[:idx_LE+1]
    # Lower surface: points FROM the LE onward (including the LE point)
    lower_x = x_blade[idx_LE:]
    lower_y = y_blade[idx_LE:]
    
    # --- Step 3: Define TE as the average of the first and last points ---
    TE = (np.array([x_blade[0], y_blade[0]]) + np.array([x_blade[-1], y_blade[-1]])) / 2.0

    # --- Step 4: Build the chord line ---
    # Chord line from LE to TE; axial chord is horizontal span.
    chord_x = np.linspace(LE[0], TE[0], n_points)
    chord_y_line = np.linspace(LE[1], TE[1], n_points)
    axial_chord = TE[0] - LE[0]
    chord_length = np.hypot(TE[0]-LE[0], TE[1]-LE[1])
    # Stagger angle is the angle between chord and x-axis (in degrees)
    stagger_angle = np.degrees(np.arctan2(TE[1]-LE[1], TE[0]-LE[0]))
    
    # --- Step 5: Interpolate upper and lower surfaces over chord x-range ---
    f_upper = interp1d(upper_x, upper_y, kind='slinear', fill_value="extrapolate")
    f_lower = interp1d(lower_x, lower_y, kind='slinear', fill_value="extrapolate")
    upper_interp_y = f_upper(chord_x)
    lower_interp_y = f_lower(chord_x)
    
    # --- Step 6: Compute the camber line as the average of upper and lower surfaces ---
    camber_y = (upper_interp_y + lower_interp_y) / 2.0
    # Force endpoints to exactly equal LE and TE y-values.
    camber_y[0] = LE[1]
    camber_y[-1] = TE[1]
    # For plotting/reconstruction, also create camber line points:
    camber_line_pts = np.column_stack((chord_x, camber_y))
    
    # --- Step 7: Metal angles and camber angle ---
    # Inlet metal angle: use the line from LE to the second camber point.
    inlet_vec = camber_line_pts[1] - LE
    inlet_metal_angle = np.degrees(np.arctan2(inlet_vec[1], inlet_vec[0]))
    # Outlet metal angle: use the line from the second-to-last camber point to TE.
    outlet_vec = TE - camber_line_pts[-2]
    outlet_metal_angle = np.degrees(np.arctan2(outlet_vec[1], outlet_vec[0]))
    # Camber angle is the absolute difference between these metal angles.
    camber_angle = abs(outlet_metal_angle - inlet_metal_angle)
    
    # --- Step 8: LE and TE radii are input parameters (le_radius, te_radius) ---
    LE_radius = le_radius
    TE_radius = te_radius
    
    # --- Step 9: Compute wedge angles ---
    # For the inlet wedge, we search along the partitioned chord for the points
    # on the upper and lower surfaces where the distance from LE is closest to LE_radius.
    distances_upper_LE = np.hypot(chord_x - LE[0], upper_interp_y - LE[1])
    distances_lower_LE = np.hypot(chord_x - LE[0], lower_interp_y - LE[1])
    i_upper_LE = np.argmin(np.abs(distances_upper_LE - LE_radius))
    i_lower_LE = np.argmin(np.abs(distances_lower_LE - LE_radius))
    # Compute approximate tangent slopes at these points (using finite differences)
    upper_dy = np.gradient(upper_interp_y, chord_x)
    lower_dy = np.gradient(lower_interp_y, chord_x)
    tangent_upper_LE = np.degrees(np.arctan(upper_dy[i_upper_LE]))
    tangent_lower_LE = np.degrees(np.arctan(lower_dy[i_lower_LE]))
    inlet_wedge_angle = abs(tangent_upper_LE - tangent_lower_LE)
    
    # For the outlet wedge, do the same using TE as center.
    distances_upper_TE = np.hypot(chord_x - TE[0], upper_interp_y - TE[1])
    distances_lower_TE = np.hypot(chord_x - TE[0], lower_interp_y - TE[1])
    i_upper_TE = np.argmin(np.abs(distances_upper_TE - TE_radius))
    i_lower_TE = np.argmin(np.abs(distances_lower_TE - TE_radius))
    tangent_upper_TE = np.degrees(np.arctan(upper_dy[i_upper_TE]))
    tangent_lower_TE = np.degrees(np.arctan(lower_dy[i_lower_TE]))
    outlet_wedge_angle = abs(tangent_upper_TE - tangent_lower_TE)
    
    # --- Step 10: Maximum thickness ---
    thickness = upper_interp_y - lower_interp_y
    max_thickness = np.max(thickness)
    
    """
    # --- Plotting for visualization ---
    plt.figure(figsize=(10,7))
    # Plot original extracted points.
    plt.plot(x_blade, y_blade, 'ko-', markersize=1, label='Extracted Points')
    # Plot chord line.
    plt.plot(chord_x, chord_y_line, 'r--', linewidth=1, label='Chord Line')
    # Plot interpolated upper and lower surfaces.
    plt.plot(chord_x, upper_interp_y, 'b-', label='Upper Surface')
    plt.plot(chord_x, lower_interp_y, 'g-', label='Lower Surface')
    # Plot camber line.
    plt.plot(chord_x, camber_y, 'm-', linewidth=1, label='Camber Line')
    # Mark LE and TE.
    plt.plot(LE[0], LE[1], 'go', markersize=2, label='Leading Edge')
    plt.plot(TE[0], TE[1], 'mo', markersize=2, label='Trailing Edge')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Airfoil Geometry, Chord, Camber, and Derived Angles')
    plt.legend()
    plt.axis('equal')
    plt.grid(False)
    plt.show()
    """
    # --- Print computed values ---
    print("Stagger angle (deg):", stagger_angle)
    print("Inlet metal angle (deg):", inlet_metal_angle)
    print("Outlet metal angle (deg):", outlet_metal_angle)
    print("Chord length:", chord_length)
    print("Axial chord (x-span):", axial_chord)
    #print("Camber angle (deg):", camber_angle)
    print("LE radius:", LE_radius)
    print("TE radius:", TE_radius)
    #print("Inlet wedge angle (deg):", inlet_wedge_angle)
    print("Outlet wedge angle (deg):", outlet_wedge_angle)
    #print("Maximum thickness:", max_thickness)
    
    # --- Return values as a dictionary ---
    results = {
        'stagger_angle': np.radians(stagger_angle),
        'inlet_metal_angle': np.radians(inlet_metal_angle),
        'outlet_metal_angle': np.radians(outlet_metal_angle),
        'chord_length': chord_length,
        'axial_chord': axial_chord,
        'camber_angle': np.radians(camber_angle),
        'LE_radius': LE_radius,
        'TE_radius': TE_radius,
        'inlet_wedge_angle': np.radians(inlet_wedge_angle),
        'outlet_wedge_angle': np.radians(outlet_wedge_angle),
        'max_thickness': max_thickness,
        'LE': LE,
        'TE': TE,
        'x_points': x_blade,
        'y_points': y_blade
    }
    
    return results

#Data Blade Validation file creation
def copy_blade_file(original_filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    original_filepath = os.path.join(current_directory, original_filename)
    new_filename = original_filename + ".databladeValidation"     #Construct the new file name
    new_filepath = os.path.join(current_directory, new_filename)
    shutil.copyfile(original_filepath, new_filepath) # Copy the file
    #print(f"Copied '{original_filename}' to '{new_filename}'.")

#---- FUNCTIONS ----
# Isentropic relation functions and other functions
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

#def compute_miux(mu0, T01, Tx, S):
#    mux = mu0 * (T01+S)/(Tx+S)*(Tx/T01)**1.5
#    return mux

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


# --------------------------- BLADE DATA EXTRACTION ---------------------------

alpha1 , alpha2, Re = extract_from_ises(isesFilePath)
pitch = extract_from_blade(bladeFilePath)
M1 , P21_ratio = extract_from_outlet(outletFilePath)
results = compute_geometry(bladeFilePath)

# --------------------------- BOUNDARY CONDITIONS ---------------------------
# Fluid Properties
R = 287.058 #[J/kg K]
gamma = 1.4
mu = 1.716e-5 # [Pa s] dynamic viscosity

# Blade Geometry
stagger = results['stagger_angle']
chord = results['chord_length']
axial_chord = results['axial_chord']

# Inlet Conditions
T01 = 314.15 #K
T1 = T01 / (1 + (gamma - 1)/2 * M1**2)
c1 = np.sqrt(gamma * R * T1)
u1 = M1 * c1
rho1 = mu * Re / (u1 * np.cos(stagger))
P1 = rho1 * R * T1 
P01 = P1 * (1 + (gamma - 1)/2 * M1**2)**(gamma/(gamma - 1))

# Outlet Conditions
P2 = P21_ratio * P1

print(f"Upstream pressure is : {P01}")
print(f"Outlet pressure is : {P2}")

# Mesh Boundary Parameters
alpha1 = int(np.degrees(np.arctan(alpha1)))
alpha2 = int(np.degrees(np.arctan(alpha2)))
dist_inlet = 2          # How many axial chords upstream will the inlet be placed
dist_outlet = 3        # How many axial chords downstream will the outlet be placed

# Turbulence Properties
TI2 = 2.2 #[%]


#%%

#####################################################################################
#                                                                                   #
#                                   MESH CREATION                                   #
#                                                                                   #
#####################################################################################                                                                                  

# --------------------------- HELPER FUNCTIONS ---------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Airfoil processing utility:
  • Read CSV or Selig-format (.databladeVALIDATION) airfoil
  • For Selig input:
      – Find LE, split raw into SS0 & PS0
      – Close TE via pyOCC NURBS
      – Identify true TE at midpoint of NURBS sample
      – Build full SS & PS curves (LE→TE)
  • Resample each side to equal n_points via arc-length + cubic spline
  • Return (x, y, s, s_norm) for SS and PS
"""


# ────────────────────────────────────────────────────────────────────────────────
# 1) I/O helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_spleen_airfoil(csvfile):
    """
    Reads a CSV that has:
      Row 1-2: Titles "PressureX,PressureY,SuctionX,SuctionY"
      Then many rows of numeric data
    Returns arrays: px, py, sx, sy
    Also identifies LE at (0,0), trailing edge at last row
    """
    data = np.loadtxt(csvfile, delimiter=',', dtype=float, skiprows=2)
    px, py, sx, sy = data[:,0], data[:,1], data[:,2], data[:,3]
    return px, py, sx, sy

def read_selig_airfoil(path):
    x, y = [], []
    with open(path, 'r') as f:
        next(f); next(f)  # skip header
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2: continue
            x.append(float(toks[0])); y.append(float(toks[1]))
    return np.array(x), np.array(y)


# ────────────────────────────────────────────────────────────────────────────────
# 2) NURBS-based TE closure
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
# 3) Resampling via arc-length + cubic spline
# ────────────────────────────────────────────────────────────────────────────────
def resample_side(x, y, n_pts):
    dx = np.diff(x); dy = np.diff(y)
    s  = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    csx, csy = CubicSpline(s, x), CubicSpline(s, y)
    s_r = np.linspace(0.0, s[-1], n_pts)
    return csx(s_r), csy(s_r), s_r, s_r / s_r[-1]


# ────────────────────────────────────────────────────────────────────────────────
# 4) Main processing
# ────────────────────────────────────────────────────────────────────────────────
def process_airfoil_file(path,
                         n_points=1000,
                         n_te=50,
                         d_factor=0.5):
    if path.lower().endswith('.csv'):
        px, py, sx, sy = read_spleen_airfoil(path)
        return {'csv': (px, py, sx, sy)}

    # 1) Read raw Selig
    x_raw, y_raw = read_selig_airfoil(path)

    # 2) Identify LE and split raw
    i_le = int(np.argmin(x_raw**2 + y_raw**2))
    x_ss0 = x_raw[i_le:];      y_ss0 = y_raw[i_le:]
    x_ps0 = x_raw[:i_le+1][::-1]; y_ps0 = y_raw[:i_le+1][::-1]
    # Now: x_ss0 runs LE→TE-SS,  x_ps0 runs LE→TE-PS

    # 3) Close TE via NURBS
    def compute_d_factor():
        wedge_deg   = np.degrees(results['outlet_wedge_angle'])
        t_te        = dist            # already computed inside process_airfoil_file
        d_factor    = 0.005*axial_chord/t_te * (10.0/wedge_deg)
        return d_factor
    
    P0 = [x_ss0[-1], y_ss0[-1]]    # TE-SS
    P3 = [x_ps0[-1], y_ps0[-1]]    # TE-PS
    dist = np.hypot(P3[0]-P0[0], P3[1]-P0[1])
    d_factor = compute_d_factor()
    d    = d_factor * dist
    vr   = [x_ss0[-2]-P0[0], y_ss0[-2]-P0[1]]
    vs   = [x_ps0[-2]-P3[0], y_ps0[-2]-P3[1]]

    CP1, CP2, _, _ = calculate_intersection_point(P0, vr, P3, vs, d)
    curve    = create_nurbs_curve(P0, CP1, CP2, P3)
    te_curve = sample_nurbs(curve, n_te)
    mid      = n_te // 2

    # 4) Split closure into halves
    ss_closure = te_curve[: mid+1]       # P0→mid
    ps_closure = te_curve[mid:][::-1]    # P3→mid

    # 5) Build full SS & PS (LE→mid)
    x_ss_full = np.concatenate([x_ss0,       ss_closure[1:,0]])
    y_ss_full = np.concatenate([y_ss0,       ss_closure[1:,1]])

    # **HERE**: use x_ps0 (LE→TE-PS) directly, not reversed
    x_ps_full = np.concatenate([x_ps0,       ps_closure[1:,0]])
    y_ps_full = np.concatenate([y_ps0,       ps_closure[1:,1]])

    # 6) Resample both sides
    x_ss, y_ss, s_ss, sn_ss = resample_side(x_ss_full, y_ss_full, n_points)
    x_ps, y_ps, s_ps, sn_ps = resample_side(x_ps_full, y_ps_full, n_points)

    return {
        'ss': (x_ss,  y_ss,  s_ss,  sn_ss),
        'ps': (x_ps,  y_ps,  s_ps,  sn_ps)
    }

def boundary_layer_props(x, rhoFlow, velFlow, muFlow, ReTurb=5e5):
    """
    Flat‑plate correlations → deltaBL, theta_mom, Cf, muTao and the y⁺=1 first‑cell height.
    Input
      x      : 1‑D arc‑length array [m] from leading edge
      rhoFlow, vel_flow : freestream density [kg/m³] and velocity [m/s]
      muFlow     : dynamic viscosity [Pa·s]
      ReTurb  : transition Re_x – below laminar, above turbulent formulas used
    """
    Rex   = rhoFlow * velFlow * x / muFlow
    
    deltaBL     = np.empty_like(x)
    thetaMom      = np.empty_like(x)
    Cf     = np.empty_like(x)
    
    for i in range(len(x)):
        if Rex[i] <= ReTurb:
            # --- Laminar Blasius -------------------------------------------------
            deltaBL[i]  = 5.0   * x[i] / np.sqrt(Rex[i])
            thetaMom[i]  = 0.664 * x[i] / np.sqrt(Rex[i])
            Cf[i] = 0.664 / np.sqrt(Rex[i])
        else:
            # --- Turbulent 1/7‑power ----------------------------------------------
            deltaBL[i]  = 0.37  * x[i] / Rex[i]**0.2
            thetaMom[i]  = 0.037 * x[i] / Rex[i]**0.2
            Cf[i] = 0.0592 / Rex[i]**0.2

    muTao   = velFlow * np.sqrt(Cf / 2.0)
    yPlus    = muFlow / (rhoFlow * muTao)          # first‑cell height for yplus = 1
    
    return (deltaBL, thetaMom, Cf, muTao, yPlus)


# --------------------------- GENERAL MESH PARAMETERS ---------------------------
sizeCellFluid: float = 0.02 * axial_chord       # Fluid related cell size
sizeCellAirfoil: float = 0.02 * axial_chord     # Airfoil related cell size
nCellAirfoil: int = 549 # 525                   # BL lines number of cells in y
nCellPerimeter: int = 183                       # BL and Mesh Boundary number of cells
nBoundaryPoints = 50                            # Top and bottom mesh boundary lines number of points 
    
# ----------------------------- PORTA BLADES SPECIFIC -----------------------------
n_points        = 1000      # For resampling of airfoil (x,y) points
n_te            = 60        # For airfoil closing TE nurbs curve
d_factor        = 0.5       # For airfoil closing TE (MUST DEFINE relation to axial chord)
    
# ----------------------------- MESH BL PARAMETERS -----------------------------
first_layer_height  = 0.01 * sizeCellAirfoil    # 1st‑cell height  (m)
bl_growth           = 1.17                      # geometric growth
bl_thickness        = 0.03 * pitch              # total BL thickness (m)
size_LE             = 0.1  * sizeCellAirfoil    # For LE refinement
dist_LE             = 0.01 * axial_chord        # For LE refinement
size_TE             = 0.1  * sizeCellAirfoil    # For TE refinement
dist_TE             = 0.01 * axial_chord        # For TE refinement

# -------------------------- REFINEMENT PARAMETERS -----------------------------
VolWAkeIn           = 0.35 * sizeCellFluid
VolWAkeOut          = sizeCellFluid
WakeXMin            = -0.1 * axial_chord 
WakeXMax            = (dist_outlet - 1.5) * axial_chord


def main_from_datablade():       
    
    # --------------------------- GEOMETRY EXTRACTION ---------------------------
    # read_spleen_airfoil returns: PS then SS.
    out = process_airfoil_file(bladeFilePath, n_points=1000, n_te=60, d_factor=0.5)
    if 'csv' in out:
        xPS, yPS, xSS, ySS = out['csv']
    else:
        xSS, ySS, _, _ = out['ss']
        xPS, yPS, _, _ = out['ps']
    
    '''
    # ── GLOBAL BL THICKNESS & y⁺‑based first‑cell height (uses inlet ρ₂, U₂) ──
    n_bl_layers = 25                         # how many prism layers you want
    x_grid      = xSS   # 0 ➔ cₐ arc‑length
    #bl          = boundary_layer_props(x_grid, rho2, V2, mu2)
    
    # --------------------------- BL PARAMETERS CALCUL ---------------------------
    FIRST_LAYER_HEIGHTxSS = float(bl["y1"].min())     # smallest y₁ ⇒ y⁺ ≤ 1 everywhere
    BL_THICKNESSxSS      = float(bl["δ"].max())       # thickest layer @ TE
    BL_RATIOxSS          = (BL_THICKNESSxSS / FIRST_LAYER_HEIGHTxSS) ** (1/(n_bl_layers-1))
    
    # ── GLOBAL BL THICKNESS & y⁺‑based first‑cell height (uses inlet ρ₂, U₂) ──                        # how many prism layers you want
    x_grid      = xPS   # 0 ➔ cₐ arc‑length
    
    FIRST_LAYER_HEIGHTxPS = float(bl["y1"].min())     # smallest y₁ ⇒ y⁺ ≤ 1 everywhere
    BL_THICKNESSxPS      = float(bl["δ"].max())       # thickest layer @ TE
    BL_RATIOxPS          = (BL_THICKNESSxSS / FIRST_LAYER_HEIGHTxSS) ** (1/(n_bl_layers-1))
    
    FIRST_LAYER_HEIGHT = (FIRST_LAYER_HEIGHTxSS + FIRST_LAYER_HEIGHTxPS) / 2
    BL_THICKNESS      = (BL_THICKNESSxSS + BL_THICKNESSxPS) / 2
    BL_RATIO          = (BL_RATIOxSS + BL_RATIOxPS) / 2
    '''
    # --------------------------- BOUNDARY POINTS ---------------------------
    L1x = dist_inlet * axial_chord
    #L1 = L1x / abs(np.cos(alpha1 * np.pi/180))
    #L1y = L1 * abs(np.sin(alpha1 * np.pi/180))
    L2x = (dist_outlet - 1) * axial_chord                     # distance from leading edge is 1 axial chord
    #L2 = L2x / abs(np.cos(alpha2 * np.pi/180))
    #L2y = L2 * abs(np.sin(alpha2 * np.pi/180))
    
    m1 = np.tan(alpha1*np.pi/180)
    m2 = np.tan(alpha2*np.pi/180)

    geo_file = os.path.join(current_directory, f"cascade2D{string}_{bladeName}.geo")
    with open(geo_file, 'w') as f:
        # ------------------ AIRFOIL CURVES ------------------
        # Top Airfoil (SS)
        f.write("// AIRFOIL TOP \n")
        for i, (x, y) in enumerate(zip(xSS, ySS)):
            f.write(f"Point({i}) = {{{x}, {y}, 0, {sizeCellAirfoil}}}; \n")
        f.write("BSpline(1000) = {")
        for j in range(0, i):
            f.write(f"{j}, ")
        f.write(f"{i}}}; \n")
        LE_ID = 0    # LE is first node of top airfoil.
        TE_ID = i    # TE is last node of top airfoil.
        
        # Bottom Airfoil (PS)
        f.write("\n// AIRFOIL BOTTOM \n")
        bottomPts = []
        for i, (x, y) in enumerate(zip(xPS, yPS)):
            ptID = 2000 + i
            bottomPts.append(ptID)
            f.write(f"Point({ptID}) = {{{x}, {y}, 0, {sizeCellAirfoil}}}; \n")
        # Override first and last node to match top airfoil:
        f.write("BSpline(2000) = {0, ")
        for pt in bottomPts[1:-1]:
            f.write(f"{pt}, ")
        f.write(f"{TE_ID}}}; \n")
    
        
        # Outer boundary points (IDs unchanged)
        x15000 = -L1x
        y15000 = m1*(x15000 - xPS[0]) + yPS[0] - pitch/2
        
        x15001 = L2x
        y15001 = m2*(x15001 - xPS[-1]) + yPS[-1] - pitch/2
        
        x15002 = x15001
        y15002 = y15001 + pitch
        
        x15003 = x15000
        y15003 = y15000 + pitch
        
        x15004 = x15001 + axial_chord
        y15004 = y15001
        
        x15005 = x15004
        y15005 = y15002
        
        # ------------------ OUTER BOUNDARY POINTS & LINES ------------------
        f.write(f"k = {sizeCellFluid}; \n")
        f.write("\n")
        f.write(f"Point(15000) = {{{x15000:.16e}, {y15000:.16e}, 0, k}};\n")   # inlet bottom
        f.write(f"Point(15001) = {{{x15001:.16e}, {y15001:.16e}, 0, k}};\n") 
        f.write(f"Point(15002) = {{{x15002:.16e}, {y15002:.16e}, 0, k}};\n") 
        f.write(f"Point(15003) = {{{x15003:.16e}, {y15003:.16e}, 0, k}};\n\n") # inlet top
        f.write(f"Point(15004) = {{{x15004:.16e}, {y15004:.16e}, 0, k}};\n") # outlet bottom
        f.write(f"Point(15005) = {{{x15005:.16e}, {y15005:.16e}, 0, k}};\n\n") # outlet top
        
        
        # ------------------ OUTER PERIMETER (node‑to‑node periodic) ------------------
        xMean = (np.array(xSS) + np.array(xPS)) / 2
        yMean = (np.array(ySS) + np.array(yPS)) / 2
        
        f.write("\n// --- bottom boundary polyline --------------------------------\n")
        # sample the mean line at nBoundaryPoints, excluding the two endpoints
        idxs = np.linspace(0, len(xMean)-1, nBoundaryPoints).astype(int)
        bottom_idxs = idxs[1:-1]  # keep for reuse
        
        # build bottom
        bottom_ids = [15000]
        for ii, idx in enumerate(bottom_idxs):
            pid = 15100 + ii
            xb, yb = xMean[idx], yMean[idx] - pitch/2
            f.write(f"Point({pid}) = {{{xb:.16e}, {yb:.16e}, 0, k}};\n")
            bottom_ids.append(pid)
        bottom_ids.append('15001, 15004')
        f.write(f"Line(150) = {{{', '.join(map(str, bottom_ids))}}};\n")
        
        f.write("\n// --- top boundary polyline (translate bottom_ids by +pitch) ---\n")
        top_ids = [15003]
        for ii, idx in enumerate(bottom_idxs):
            tpid = 15100 + ii + 100
            xt, yt = xMean[idx], yMean[idx] + pitch/2
            f.write(f"Point({tpid}) = {{{xt:.16e}, {yt:.16e}, 0, k}};\n")
            top_ids.append(tpid)
        top_ids.append('15002, 15005')
        f.write(f"Line(152) = {{{', '.join(map(str, top_ids))}}};\n")
        
        f.write("\n// --- single inlet/outlet lines ------------------------------\n")
        f.write("Line(153) = {15000, 15003};   // inlet\n")
        f.write("Line(151) = {15004, 15005};   // outlet\n")
        
        f.write("\n// --- mesh boundary loop --------------------------------------\n")
        f.write("Curve Loop(50) = {150, 151, -152, -153};\n\n")
        
        # ------------------ CURVE LOOPS ------------------
        f.write("\n// Curve Loop 10 (airfoil)\n")
        f.write("Curve Loop(10) = {1000, -2000};\n")
        f.write("\n// already wrote Curve Loop 50 above\n\n")
        
        # ------------------ PLANE SURFACES ------------------
        # Now define plane surfaces from the curve loops.
        f.write("Plane Surface(5) = {50, 10}; \n") # Fluid subdomain
    
        # ------------------ TRANSFINITE MESH DEFINITIONS ------------------
        f.write("\n// Transfinite definitions for connector lines\n")
        # Airfoil and Boundary layer curves
        f.write(f"Transfinite Curve {{1000}} = {nCellAirfoil} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{2000}} = {nCellAirfoil} Using Progression 1; \n")
        # Airfoil and Mesh boundary curves
        f.write(f"Transfinite Curve {{10}} = {nCellPerimeter} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{50}} = {nCellPerimeter} Using Progression 1; \n")
    
        # --------------------------------------------------------------------- #
        #  NEW 1  ─ Boundary‑Layer field (curved, orthogonal grid lines)        #
        # --------------------------------------------------------------------- #
        
        '''
        first_layer_height  = FIRST_LAYER_HEIGHT            # 1st‑cell height  (m)
        bl_growth           = BL_RATIO                       # geometric growth
        bl_thickness        = BL_THICKNESS              # total BL thickness (m)
        '''
        f.write("\n// --- BOUNDARY‑LAYER FIELD (curved normals) ---------------\n")
        f.write("Field[1] = BoundaryLayer;\n")
        f.write("Field[1].EdgesList   = {1000, 2000};   // SS & PS splines\n")
        f.write(f"Field[1].hwall_n     = {first_layer_height};\n")
        f.write(f"Field[1].ratio       = {bl_growth};\n")
        f.write(f"Field[1].thickness   = {bl_thickness};\n")
        f.write(f"Field[1].hfar        = {sizeCellFluid};\n")
        f.write("Field[1].Quads       = 1;              // keep quads after recombine\n")
        f.write("BoundaryLayer Field = 1;\n")
        
        # --------------------------------------------------------------------- #
        #  NEW 2  ─ LE & TE refinement via Attractor + Threshold            #
        # --------------------------------------------------------------------- #
        #  LE
        f.write("\nField[2] = Attractor;\n")
        f.write("Field[2].EdgesList = {1000};   // SS spline (LE)\n")
        f.write("Field[3] = Threshold;\n")
        f.write("Field[3].InField   = 2;\n")
        f.write(f"Field[3].SizeMin   = {size_LE};\n")
        f.write(f"Field[3].SizeMax   = {sizeCellFluid};\n")
        f.write(f"Field[3].DistMin   = 0;\n")
        f.write(f"Field[3].DistMax   = {dist_LE};\n")

        #  TE
        f.write("\nField[4] = Attractor;\n")
        f.write("Field[4].EdgesList = {2000};   // PS spline (TE)\n")
        f.write("Field[5] = Threshold;\n")
        f.write("Field[5].InField   = 4;\n")
        f.write(f"Field[5].SizeMin   = {size_TE};\n")
        f.write(f"Field[5].SizeMax   = {sizeCellFluid};\n")
        f.write(f"Field[5].DistMin   = 0;\n")
        f.write(f"Field[5].DistMax   = {dist_TE};\n")

        # Merge BL + LE + TE
        f.write("\nField[6] = Min;\n")
        f.write("Field[6].FieldsList = {1, 3, 5};\n")
        f.write("Background Field = 6;\n")
        
        # ---------------------------------------------------------------------
        #  NEW 4 ─ Wake strip refinement via Box field
        # ---------------------------------------------------------------------
        f.write("\nField[7] = Box;\n")
        f.write(f"Field[7].VIn   = { VolWAkeIn };\n")      # 0.25 background size inside box
        f.write(f"Field[7].VOut  = { VolWAkeOut };\n")          # background size outside
        # box from just upstream of LE (−0.1·c) to outlet (+dist_outlet·c)
        f.write(f"Field[7].XMin  = { WakeXMin };\n")
        f.write(f"Field[7].XMax  = { WakeXMax };\n")
        # full pitch height, centered on camber line (y=0)
        f.write(f"Field[7].YMin  = { y15001 };\n")
        f.write(f"Field[7].YMax  = { pitch };\n")
        # flat 2D mesh
        f.write("Field[7].ZMin  = 0;\n")
        f.write("Field[7].ZMax  = 0;\n")
        
        # now merge this wake field with the existing BL+LE+TE field 6
        f.write("\nField[8] = Min;\n")
        f.write("Field[8].FieldsList = {6, 7};\n")
        f.write("Background Field = 8;\n")
        
        # --------------------------------------------------------------------- #
        #  NEW 3  ─ Elliptic (Laplacian) smoother for interior node positions   #
        # --------------------------------------------------------------------- #
        f.write("\n// --- LAPLACIAN SMOOTHING ----------------------------------\n")
        f.write("Mesh.Smoothing = 100;\n")
        f.write("Mesh.OptimizeNetgen = 1; \n")    # cleans skewed quads after recombine
        
        # ------------------ PHYSICAL GROUPS ------------------
        f.write('Physical Curve("inlet", 18001) = {153};\n')
        f.write('Physical Curve("simmetricWallsBOTTOM", 18002) = {150};\n')
        f.write('Physical Curve("simmetricWallsTOP",    18003) = {152};\n')
        f.write('Physical Curve("outlet", 18004) = {151};\n')
        f.write('Physical Curve("blade1", 18005) = {2000, 1000};\n')
        f.write('Physical Surface("fluid", 18008) = {5};\n') 
        
        
    print(f"Geo file written at: {geo_file}")
    
    # Run gmsh to generate the SU2 mesh.
    print("STARTING mesh generation...")
    try:
        if os.path.exists(geo_file):
            print(f"File exists at: {geo_file}")
        else:
            print(f"File not found at: {geo_file}")
            
        os.system(f'gmsh "{geo_file}" -2 -format su2')
        print("Mesh successfully created!")
    except Exception as e:
        print("Error", e)       


if __name__ == "__main__":
    # Run the main mesh creation and SU2 simulation.
    main_from_datablade()  
    
    
#raise SystemExit("Stopping script execution.")
    
#%%

#####################################################################################
#                                                                                   #
#                              SU2 CONFIG FILE CREATION                             #
#                                                                                   #
##################################################################################### 


def configFile():
 
    data_airfoil = f'''

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                   
%                                                                              %
% SU2 AIRFOIL configuration file                                               %
% Case description: General Airfoil                                            %
% Author: Freddy Chica	                                                       %
% Institution: Université Catholique de Louvain                                %
% Date: 11, Nov 2024                                                           %
% File Version                                                                 %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
SOLVER                  = RANS
KIND_TURB_MODEL         = SA
SA_OPTIONS              = BCM
KIND_TRANS_MODEL        = NONE
MATH_PROBLEM            = DIRECT
RESTART_SOL             = NO


% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%
MACH_NUMBER                     = {M1}              % Inlet Mach number
AOA                             = {alpha1}          % Midspan cascade aligned with the flow
FREESTREAM_PRESSURE             = {P01}              % Free-stream static pressure in Pa
FREESTREAM_TEMPERATURE          = {T01}              % Free-stream static temperature
REYNOLDS_NUMBER                 = {Re}             % Free-stream Reynolds number
REYNOLDS_LENGTH                 = {axial_chord}     % Normalization length
FREESTREAM_TURBULENCEINTENSITY  = 0.001 % 0.025{TI2/100}         % (If SST used) freestream turbulence intensity (2% as example)
FREESTREAM_TURB2LAMVISCRATIO    = 0.1  %10              % (If SST used) ratio of turbulent to laminar viscosity
%FREESTREAM_NU_FACTOR            = 3                 % (For SA) initial turbulent viscosity ratio (default 3)
% The above turbulence freestream settings are not all used for SA, but included for completeness.

REF_ORIGIN_MOMENT_X             = 0.0
REF_ORIGIN_MOMENT_Y             = 0.0
REF_ORIGIN_MOMENT_Z             = 0.0
REF_LENGTH                      = {axial_chord}
REF_AREA                        = 0.0
REF_DIMENSIONALIZATION          = DIMENSIONAL


%-------------------------- GAS & VISCOSITY MODEL -----------------------------%
FLUID_MODEL             = IDEAL_GAS
GAMMA_VALUE             = {gamma}
GAS_CONSTANT            = {R}
VISCOSITY_MODEL         = SUTHERLAND
MU_REF                  = 1.716E-5
MU_T_REF                = 273.15
SUTHERLAND_CONSTANT     = 110.4


% -------------------- BOUNDARY CONDITION DEFINITION --------------------------%
INLET_TYPE              = TOTAL_CONDITIONS
MARKER_HEATFLUX         = ( blade1, 0.0, blade2, 0.0  )

MARKER_PLOTTING         = ( blade1 )                                        % Marker(s) of the surface in the surface flow solution file
MARKER_MONITORING       = ( blade1 )                                        % Marker(s) of the surface where the non-dimensional coefficients are evaluated.
MARKER_ANALYZE          = ( inlet, outlet, blade1 )                         % Marker(s) of the surface that is going to be analyzed in detail (massflow, average pressure, distortion, etc)

MARKER_INLET            = ( inlet, {T01}, {P01}, {np.cos(alpha1 * np.pi / 180)}, {np.sin(alpha1 * np.pi / 180)}, 0)
MARKER_OUTLET           = ( outlet,  {P2})
MARKER_PERIODIC         = ( simmetricWallsBOTTOM, simmetricWallsTOP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {pitch}, 0.0 )
%MARKER_INLET_TURBULENT = ( inlet, TI2, nu_factor2 )          %SST Model
%MARKER_INLET_TURBULENT = ( inlet,  nu_factor2 )                %SA Model


%-------------------------- NUMERICAL METHODS SETTINGS ------------------------%

% -------------------- FLOW NUMERICAL METHOD DEFINITION 
CONV_NUM_METHOD_FLOW    = JST                      %Can try SLAU, ROE or AUSMPLUSUP2, but if so must define other ROE parameters such as ROE_KAPPA
ENTROPY_FIX_COEFF       = 0.05
TIME_DISCRE_FLOW        = EULER_IMPLICIT

% -------------------- TURBULENT NUMERICAL METHOD DEFINITION 
CONV_NUM_METHOD_TURB    = SCALAR_UPWIND
TIME_DISCRE_TURB        = EULER_IMPLICIT
CFL_REDUCTION_TURB      = 0.8                       %Can try other values

% ----------- SLOPE LIMITER AND DISSIPATION SENSOR DEFINITION 
MUSCL_FLOW              = NO
MUSCL_TURB              = NO                        %Can try YES
SLOPE_LIMITER_FLOW      = BARTH_JESPERSEN           %Can try VAN_ALBADA_EDGE
SLOPE_LIMITER_TURB      = VENKATAKRISHNAN           %Should be same as SLOPE_LIMITER_FLOW unless VAN_ALBADA_EDGE
VENKAT_LIMITER_COEFF    = 0.01                      %Can ty 0.05 default
LIMITER_ITER            = 999999
JST_SENSOR_COEFF        = ( 0.5, 0.02 )             %Can try other values but rather unnecessary

% ------------- COMMON PARAMETERS DEFINING THE NUMERICAL METHOD 
NUM_METHOD_GRAD_RECON   = WEIGHTED_LEAST_SQUARES
CFL_NUMBER              = 12 %0.1                   %original 20 
CFL_ADAPT               = YES
CFL_ADAPT_PARAM         = ( 0.1, 1.2, 0.1, 100.0)    %Structure (factor-down, factor-up, CFL min value, CFL max value, acceptable linear solver convergence, starting iteration)
                      % = ( 0.1, 1.2, 0.1, 75) 
% ------------------------ LINEAR SOLVER DEFINITION 
LINEAR_SOLVER                       = FGMRES        %Can try BGSTAB or RESTARTED_FGMRES
LINEAR_SOLVER_PREC                  = LU_SGS        %Can try ILU, LU_SGS (Lower-Upper Symmetric Gauss-Seidel)
%LINEAR_SOLVER_ILU_FILL_IN          = 0             %Can try 1-2 or even 3 to test convergence speed
LINEAR_SOLVER_ERROR                 = 1E-4
LINEAR_SOLVER_ITER                  = 10
%LINEAR_SOLVER_RESTART_FREQUENCY    = 10            %Can try if Linear Solver = RESTARTED_FGMRES

% -------------------------- MULTIGRID PARAMETERS 
MGLEVEL                 = 3                         % Multi-Grid Levels (0 = no multi-grid) - Can even try 4
MGCYCLE                 = V_CYCLE                   % Can try W-CYCLE but perhaps unnecesary
MG_PRE_SMOOTH           = ( 1, 2, 3, 3, 3, 3 )      % Multigrid pre-smoothing level
MG_POST_SMOOTH          = ( 1, 2, 3, 3, 3, 3 )      % Multigrid post-smoothing level
MG_CORRECTION_SMOOTH    = ( 1, 2, 3, 3, 3, 3 )      % Jacobi implicit smoothing of the correction
MG_DAMP_RESTRICTION     = 0.75                      % Damping factor for the residual restriction
MG_DAMP_PROLONGATION    = 0.75                      % Damping factor for the correction prolongation


% ------------------------------- SOLVER CONTROL ------------------------------%
CONV_RESIDUAL_MINVAL    = -8                        % Can try lower but unnecessary
CONV_FIELD              = RMS_DENSITY               % Can also try MASSFLOW
CONV_STARTITER          = 500                       % Original 500
ITER                    = 6000


% ------------------------- SCREEN/HISTORY VOLUME OUTPUT --------------------------%
SCREEN_OUTPUT           = (INNER_ITER, WALL_TIME, RMS_DENSITY, LINSOL_ITER_TRANS, LINSOL_RESIDUAL_TRANS)
HISTORY_OUTPUT          = (INNER_ITER, WALL_TIME, RMS_DENSITY , RMS_MOMENTUM-X , RMS_ENERGY, RMS_TKE, RMS_DISSIPATION, LINSOL, CFL_NUMBER, FLOW_COEFF_SURF, AERO_COEFF_SURF)
VOLUME_OUTPUT           = (COORDINATES, SOLUTION, PRIMITIVE, RESIDUAL, TIMESTEP, MESH_QUALITY, VORTEX_IDENTIFICATION)
SCREEN_WRT_FREQ_INNER   = 10
OUTPUT_WRT_FREQ         = 50
WRT_PERFORMANCE         = YES
WRT_RESTART_OVERWRITE   = YES
WRT_SURFACE_OVERWRITE   = YES
WRT_VOLUME_OVERWRITE    = YES
WRT_FORCES_BREAKDOWN    = NO
%COMM_LEVEL             = FULL %Can try to optimize or test MPI runN_ing


% ------------------------- INPUT/OUTPUT FILE INFORMATION --------------------------%
% INPUTS
MESH_FILENAME           = cascade2D{string}_{bladeName}.su2
MESH_FORMAT             = SU2
MESH_OUT_FILENAME       = cascade2D{string}_out_{bladeName}.su2
SOLUTION_FILENAME       = restart_flow{string}_{bladeName}.dat

% OUTPUTS
OUTPUT_FILES            = (RESTART, PARAVIEW, SURFACE_PARAVIEW, CSV, SURFACE_CSV)
CONV_FILENAME           = history_{string}_{bladeName}
RESTART_FILENAME        = restart_flow{string}_{bladeName}.dat
VOLUME_FILENAME         = volume_flow{string}_{bladeName}
SURFACE_FILENAME        = surface_flow{string}_{bladeName}
%GRAD_OBJFUNC_FILENAME  = of_grad{string}_{bladeName}.dat

'''

    # Write the information to the AIRFOIL file
    with open(f"cascade2D{string}_{bladeName}.cfg", "w") as f:
        f.write(data_airfoil)    

# Run the main function
if __name__ == "__main__":
    configFile()
    print(f"Config file created: cascade2D{string}_{bladeName}.cfg")
    
#####################################################################################
#                                                                                   #
#                                       SU2 RUN                                     #
#                                                                                   #
##################################################################################### 

if __name__ == "__main__":
    
    # Run SU2 simulation using the config file.
    config_file = os.path.join(current_directory, f"cascade2D{string}_{bladeName}.cfg")
    try:
        if os.path.exists(config_file):
            print(f"Config file exists at: {config_file}")
        else:
            print(f"Config file not found at: {config_file}")
        os.system(f'mpiexec -n "{no_cores}" SU2_CFD "{config_file}"')
        print("SU2 Run Initialized!")
    except Exception as e:
        print("Error", e)  

    #################################################################################
    #  HISTORY FILE TRACKING - Residuals, Linear Solvers, CFL, CD, CL           
    #################################################################################
    
    hist = pd.read_csv(f'history_{string}_{bladeName}.csv')
    
    # RMS Tracking
    plt.plot(hist['Inner_Iter'], hist['    "rms[Rho]"    '], label='ρ')                     # Density
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoU]"   '], label='ρu')                    # Momentum-x
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoE]"   '], label='ρE')                    # Energy
    #plt.plot(hist['Inner_Iter'], hist['    "rms[RhoV]"   '], label='ρv')                    # Momentum-y
    #plt.plot(hist['Inner_Iter'], hist['     "rms[nu]"    '], label='v')                     # Viscosity
    #plt.plot(hist['Inner_Iter'], hist['     "rms[k]"    '], label='k')                     # TKE
    #plt.plot(hist['Inner_Iter'], hist['     "rms[w]"    '], label='w')
    plt.grid(alpha=0.3);  plt.legend();  plt.xlabel('Iteration')
    plt.ylabel('RMS residual - Airfoil');  plt.tight_layout();  plt.show()
    
    # Linear Solver Tracking
    plt.plot(hist['Inner_Iter'], hist['    "LinSolRes"   '], label='LSRes')                 # Linear Solver Residual
    plt.plot(hist['Inner_Iter'], hist['  "LinSolResTurb" '], label='LSResTurb')             # Linear Solver Residual
    plt.grid(alpha=0.3);  plt.legend();  plt.xlabel('Iteration')
    plt.ylabel('Linear Solver residual - Airfoil');  plt.tight_layout();  plt.show()
    
    # RMS Tracking
    plt.plot(hist['Inner_Iter'], hist['     "Avg CFL"    '], label='CFL')                   # CFL used per iteration
    plt.grid(alpha=0.3);  plt.legend();  plt.xlabel('Iteration')
    plt.ylabel('Average CFL - Airfoil');  plt.tight_layout();  plt.show()
    
    # Aero Coefficients Tracking
    plt.plot(hist['Inner_Iter'], hist['   "CD(blade1)"   '], label='CD')                    # Drag Coefficient
    plt.plot(hist['Inner_Iter'], hist['   "CL(blade1)"   '], label='CL')                    # Lift Coefficient
    plt.grid(alpha=0.3);  plt.legend();  plt.xlabel('Iteration')
    plt.ylabel('Aerodynamic Coefficients - Airfoil');  plt.tight_layout();  plt.show()

    
#raise SystemExit("Stopping script execution.")

# %%

#####################################################################################
#                                                                                   #
#                                RESULT VISUALIZATION                               #
#                                                                                   #
##################################################################################### 

# ─────────────────────────────────────────────────────────────────────────────
#   BASIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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

def roll_array(arr, shift):
    """
    Rolls the arrays so that idx_maxP is placed at index 0.
    Ensures the suction side starts at the max‐pressure location for easy plotting.
    """
    return np.concatenate([arr[shift:], arr[:shift]])

# ─────────────────────────────────────────────────────────────────────────────
#   SU2 UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def SU2_organize(df):
    """
    Reorganizes the surface CSV data from SU2 to separate
    leading_edge, upper_surface, trailing_edge, lower_surface.
    """
    leading_edge  = df.iloc[0:1].copy()      # row 0
    trailing_edge = df.iloc[1:2].copy()      # row 1

    geo = df.iloc[2:].copy().reset_index(drop=True)
    x, y = geo['x'].values, geo['y'].values

    # find the largest jump between consecutive points = break TE→LE
    dist = np.hypot(np.diff(x), np.diff(y))
    idx_break = np.argmax(dist) + 1          # first point of pressure surface

    upper_surface  = geo.iloc[:idx_break].copy()           # suction side
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
        mirror_PS=False,
        exp_x=None, # optional experimental x array
        exp_mach=None # optional experimental Mach array
    ):
    """
    Plots SU2 results in Non-Norm style (direct values) plus
    optional experimental data for direct comparison.
    """
    fig, ax1 = plt.subplots(figsize=(14, 9))

    # Plot SU2 (suction & pressure side)
    plt.plot(sSSnorm, dataSS, marker='o', markersize=2, linestyle='-', color='darkblue', label='SU2 (SS)')
    
    s_ps = -sPSnorm if mirror_PS else sPSnorm
    plt.plot(s_ps, dataPS, marker='o', markersize=2, linestyle='-', color='lightblue', label='SU2 (PS)')

    # Overlay optional experimental distribution
    if (exp_x is not None) and (exp_mach is not None):
        plt.scatter(exp_x, exp_mach, s=20, color='red', label='Mises Data')

    plt.ylabel(f'{quantity}', size=20)
    plt.tick_params(axis='y', labelcolor='grey')
    plt.grid(visible=True, color='lightgray', linestyle='--')
    if mirror_PS:
        plt.xlim(-1, 1)       # show full mirror
    else:
        plt.xlim(0, 1)
    plt.legend(loc='upper left', prop={'size': 20}, edgecolor='k', fancybox=False)
    plt.savefig(f"non-normalized{quantity}_{string}_{bladeName}.svg", format='svg', bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#   MISES UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def MISES_blDataGather(file_path):
    """
    Reads boundary-layer data from file_path, skipping the first two header lines.
    Blank lines (or short lines) mark the end of a streamtube.
    Returns lists of lists for each variable:
      - x, y geometry
      - delta_star, theta, theta_star, shape_factor, Mach
    """
    all_x_values = []
    all_y_values = []
    all_delta_star = []
    all_theta = []
    all_theta_star = []
    all_shape_factor = []
    all_mach = []

    # Temporary arrays for the current streamtube
    x_tmp, y_tmp = [], []
    ds_tmp, th_tmp, ts_tmp, sf_tmp, M_tmp = [], [], [], [], []

    with open(file_path, 'r') as f:
        # Skip header lines
        for _ in range(2):
            next(f, None)

        for line in f:
            tokens = line.split()
            if len(tokens) < 9:
                # End of one streamtube
                if x_tmp:
                    all_x_values.append(x_tmp)
                    all_y_values.append(y_tmp)
                    all_delta_star.append(ds_tmp)
                    all_theta.append(th_tmp)
                    all_theta_star.append(ts_tmp)
                    all_shape_factor.append(sf_tmp)
                    all_mach.append(M_tmp)
                # Reset for next tube
                x_tmp, y_tmp = [], []
                ds_tmp, th_tmp, ts_tmp, sf_tmp, M_tmp = [], [], [], [], []
                continue

            x_tmp.append(float(tokens[0]))
            y_tmp.append(float(tokens[1]))
            ds_tmp.append(float(tokens[5]))
            th_tmp.append(float(tokens[6]))
            ts_tmp.append(float(tokens[7]))
            sf_tmp.append(float(tokens[8]))
            M_tmp.append(float(tokens[13]))

    # Catch final tube if no trailing blank line
    if x_tmp:
        all_x_values.append(x_tmp)
        all_y_values.append(y_tmp)
        all_delta_star.append(ds_tmp)
        all_theta.append(th_tmp)
        all_theta_star.append(ts_tmp)
        all_shape_factor.append(sf_tmp)
        all_mach.append(M_tmp)

    return (
        all_x_values, 
        all_y_values, 
        all_delta_star, 
        all_theta, 
        all_theta_star, 
        all_shape_factor, 
        all_mach
    )

def MISES_fieldDataGather(file_path):
    """
    Reads field data from file_path, skipping the first two header lines.
    Columns (0..7) => x, y, rho/rho0, p/p0, u/a0, v/a0, q/a0, M
    Blank lines separate the data into streamtubes.
    Returns all_x, all_y, all_rho, all_p, all_u, all_v, all_q, all_m
    """
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

def surfaceFlowAnalysis(extension):
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   SU2 DATA
    # ─────────────────────────────────────────────────────────────────────────────

    su2_file = os.path.join(current_directory, f"surface_flow{string}_{bladeName}.csv")
    df = pd.read_csv(su2_file, sep=',')
    x      = df['x'].values
    y      = df['y'].values
    xNorm  = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    '''
    x      = df['x'].values
    y      = df['y'].values
    s_norm = surface_fraction(x,y)
    pressure        = df['Pressure'].values
    pressure_coeff  = df['Pressure_Coefficient'].values
    friction_coeff  = df['Skin_Friction_Coefficient_x'].values
    yPlus           = df['Y_Plus'].values
    
    temperature     = df['Temperature'].values
    density         = df['Density'].values
    energy          = df['Energy'].values
    laminar_visc    = df['Laminar_Viscosity'].values
    '''
    
    _, _, dataSS, dataPS = SU2_organize(df)
    
    # Suction Side - Upper Surface
    xSS                 = dataSS['x'].values
    ySS                 = dataSS['y'].values
    s_normSS            = surface_fraction(xSS,ySS)
    pressureSS          = dataSS['Pressure'].values
    pressure_coeffSS    = dataSS['Pressure_Coefficient'].values
    friction_coeffSS    = dataSS['Skin_Friction_Coefficient_x'].values
    yPlusSS             = dataSS['Y_Plus'].values
    
    temperatureSS       = dataSS['Temperature'].values
    densitySS           = dataSS['Density'].values
    energySS            = dataSS['Energy'].values
    laminar_viscSS      = dataSS['Laminar_Viscosity'].values
    
    machSS = compute_Mx(P01, pressureSS, gamma)
    
    # Pressure Side - Lower Surface
    xPS                 = dataPS['x'].values
    yPS                 = dataPS['y'].values
    s_normPS            = surface_fraction(xPS,yPS)
    s_normPS_mirr       = -s_normPS
    pressurePS          = dataPS['Pressure'].values
    pressure_coeffPS    = dataPS['Pressure_Coefficient'].values
    friction_coeffPS    = dataPS['Skin_Friction_Coefficient_x'].values
    yPlusPS             = dataPS['Y_Plus'].values
    
    temperaturePS       = dataPS['Temperature'].values
    densityPS           = dataPS['Density'].values
    energyPS            = dataPS['Energy'].values
    laminar_viscPS      = dataPS['Laminar_Viscosity'].values
    
    machPS = compute_Mx(P01, pressurePS, gamma)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   MISES DATA
    # ─────────────────────────────────────────────────────────────────────────────
    
    #Again we read the files in the directory
    resultsFileName = f"machDistribution.{string}"
    resultsFilePath = os.path.join(current_directory, resultsFileName)
    
    #We extract the MISES surface infromation in lists for upper and lower surfaces
    with open(file=resultsFilePath, mode='r') as f:
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

    
    # ─────────────────────────────────────────────────────────────────────────────
    #   RMS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────────
    
    # --------- Linear‑interp SU2 onto those fractions 
    su2_ss = np.interp(ss_frac, s_normSS, machSS)
    su2_ps = np.interp(ps_frac, s_normPS, machPS)
    
    # --------- Combined RMS (%)  
    rel_err_ss = (ss_mach - su2_ss) / su2_ss
    rel_err_ps = (ps_mach - su2_ps) / su2_ps
    rms_pct = np.sqrt(np.mean(np.concatenate([rel_err_ss**2, rel_err_ps**2]))) * 100
    
    print(f"\nCombined RMS error = {rms_pct:.2f}%")
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   PLOTTING
    # ─────────────────────────────────────────────────────────────────────────────
    
    # ---------- Single overlay of Mach
    
    SU2_DataPlotting(
        sSSnorm     = s_normSS,
        sPSnorm     = s_normPS,
        dataSS      = machSS,
        dataPS      = machPS,
        quantity    ="Mach Number",
        string      = string,
        mirror_PS   = False,
        exp_x       = blade_frac,
        exp_mach    = blade_mach
    )
    
    SU2_DataPlotting(s_normSS, s_normPS, yPlusSS, yPlusPS,
                 "Y Plus", string, mirror_PS=True)
    
    SU2_DataPlotting(s_normSS, s_normPS, friction_coeffSS, friction_coeffPS,
                 "Skin Friction Coefficient", string, mirror_PS=True)
    
    
###############################################################################
#                          EXECUTE THE CODE                                   #
###############################################################################

if __name__ == "__main__":
    surfaceFlowAnalysis('databladeVALIDATION')

