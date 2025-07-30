
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
# - SPLEENC1_Geometry_Airfoil_2D_v1.csv
# - Mach Distribution file
# Others to be determined when checking other files

#---- FILE MANAGEMENT AND DIRECTORIES ----
# SU2 related 
no_cores = 12 # Change this to switch the processing power of the computation (numbers of used cores)
string = "databladeVALIDATION" # File names
string2 = "safe_start" # For SU2 optimization

# SPLEEN related 
bladeName = "SPLEEN"
blade_fileName = "SPLEENC1_Geometry_Airfoil_2D_v1"
current_directory = os.path.dirname(os.path.abspath(__file__))
fileExtension = "csv"
airfoil_file_path   = os.path.join(current_directory, f"{blade_fileName}.{fileExtension}")
airfoil_file_path2  = os.path.join(current_directory, f"blade.{string}")
expFilesLocation    = os.path.join(current_directory, "Experimental Data")
filesPL01           = os.path.join(expFilesLocation, "PL01")
filesPL02           = os.path.join(expFilesLocation, "PL02")
filesPL06           = os.path.join(expFilesLocation, "PL06")
filesBlade          = os.path.join(expFilesLocation, "Blade")


#---- BLADE GEOMETRY ----
pitch = 32.950e-3 #[m]
chord_length = 52.285e-3 #[m]
axial_chord = 47.614e-3 #[m]
stagger = 24.40 #[deg]
alpha_m_in = 37.3 #[deg]
alpha_m_out = -53.80 #[deg]


#---- TESTING SETTINGS ----
Re = 70 
M = 70
St_test     = '000'
Re_test     = f'{Re}'
M_test      = f'0{M}'

# Define reference arrays for indexing
Mach_levels = [70, 80, 90, 95]      # Corresponds to M = 0.70, 0.80, etc.
Re_levels   = [65, 70, 100, 120]    # Corresponds to Re = 65k, 70k, etc.

# Get indices
mach_index = Mach_levels.index(M)
re_index   = Re_levels.index(Re)

# Compute flattened index (4 elements per Mach level, in your list layout)
flat_index = mach_index * len(Re_levels) + re_index

# Lookup Tables
#Re         65k     70k     100k    120k
Y_TG_tests = [                                #Ma
            0.0515, 0.0516, 0.0523, 0.0527, #0.70
            0.0567, 0.0567, 0.0570, 0.0572, #0.80
            0.0597, 0.0597, 0.0596, 0.0595, #0.90
            0.0604, 0.0604, 0.0600, 0.0598  #0.95
            ]

#Re         65k    70k    100k   120k                   From Table 5.1 - Measurement Techniques
P01_tests = [                                   #Ma
            10009, 10779, 15399, 18478,   #0.70
            9295,  10010, 14301, 17161,   #0.80
            8821,  9500,  13571, 16285,   #0.90
            8652,  9318,  13311, 15974    #0.95
            ]

#Re         65k   70k   100k   120k                     From Table 5.1 - Measurement Techniques
P6_tests = [                                  #Ma
            7216, 7771, 11101, 13321,   #0.70
            6098, 6567, 9381,  11258,   #0.80
            5216, 5617, 8024,  9629,    #0.90
            4841, 5213, 7447,  8937     #0.95
            ]

# Assign values
Y_TG_test = Y_TG_tests[flat_index]
P01_test  = P01_tests[flat_index]
P6_test   = P6_tests[flat_index]

# Optional display
print(f"Selected Y_TG: {Y_TG_test}, P01: {P01_test}, P6: {P6_test}")

#C1_1, C2_1, C3_1,            = [-0.04183, 0.17898, -3.69520e-09]
#C1_2, C2_2, C3_2, C4_2, C5_2 = [-0.05513, 0.18825, 5.04960e-08, -0.09340, -4.53216e-08] 
#Y_TG1 = C1_1 + C2_1*M1_is + C3_1*Re1_is
#Y_TG6 = C1_2 + C2_2*M6_is + C3_2*Re6_is + C4_2*M6_is**2 + C5_2*M6_is*Re6_is


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

#---- TURBULENCE INLET CONDITIONS ---- (From Table 4.2 - Measurement Techniques)
# Without TG
#TI_iso2 = 0.6 #[%] Isotropic turbulence assumption
#TI2 = 0.7 #[%]
#ILS_1 = 60e-3 #[m] @ 20 - 100 Hz avg
#ILS_2 = 17.5e-3 #[m]  @ 100 - 400 Hz avg
#ILS_avr2 = np.avr(ILS_1, ILS_2)

# With TG
TI_iso2 = 2.4 #[%] Isotropic turbulence assumption
TI2 = 2.2 #[%]
ILS_1 = 13.5e-3 #[m] @ 20 - 100 Hz avg
ILS_2 = 11e-3 #[m]  @ 100 - 400 Hz avg
ILS_avr2 = (ILS_1 + ILS_2)/2


#####################################################################################
#                           Spleen Boundary Conditions                              #
##################################################################################### 
# Fluid properties
R = 287.058 #[J/kg K]
gamma = 1.4
mu = 1.716e-5 # [Pa s] dynamic viscosity
#nu = 1.5e-5 # [m2/s] kinematic viscosity 

# Blade testing conditions (@PL02 & @PL06)
fileExtension           = 'xlsx'

df_dataPath             = os.path.join(filesPL02, f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_PL02_C5HP_pa.{fileExtension}")
df_exp_PL02_pa          = pd.read_excel(df_dataPath, usecols=[2,3,4,5])
df_exp_PL02_pa.columns  = ['i','pitch','P02/P01','Ps2/P01']

df_dataPath             = os.path.join(filesPL06, f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_PL06_L5HP_MeshOD_pa.{fileExtension}")
df_exp_PL06_pa          = pd.read_excel(df_dataPath, usecols=[2,3,4,5,6])
df_exp_PL06_pa.columns  = ['d','pitch','P06/P01','Ps6/P01','ksi']

# Blade testing turbulence conditions (@PL02)
#df_dataPath             = os.path.join(filesPL02, f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_PL02_XW_s5000.{fileExtension}")
#df_exp_PL02_trb         = pd.read_excel(df_dataPath, usecols=[5])
#df_exp_PL02_trb.columns = ['U_mean']

# PL_ref Conditions
T0_ref = 300 #[K]
P_ref = 9310.72429 #[Pa]
#M_ref = compute_Mx(P01, P_ref, gamma)
#T_ref      = compute_Tx(T0_ref, M_ref, gamma)
#V_ref      = compute_Vx(M_ref, gamma, R, T_ref)
#rho_ref    = compute_rhox(P_ref, T_ref, R)
#mu_ref = compute_mux(T_ref)
#Re_ref = compute_Rex(rho_ref, V_ref, axial_chord, mu_ref)

# PL01 Conditions
#P01     = P0_ref * (1 - Y_TG)
P01     = 10779.39 #P01_test  
P1      = P_ref
T01     = T0_ref
M1      = compute_Mx(P01, P1, gamma)
T1      = compute_Tx(T01, M1, gamma)
V1      = compute_Vx(M1, gamma, R, T1)
rho1    = compute_rhox(P1, T1, R)
mu1     = compute_mux(T1)
Re1     = compute_Rex(rho1, V1, axial_chord, mu1)

# PLO2 Conditions
P02     = P01 * df_exp_PL02_pa['P02/P01'].iloc[0]
P2      = P01 * df_exp_PL02_pa['Ps2/P01'].iloc[0]
#Umean2  = np.average(df_exp_PL02_trb['U_mean'].iloc[:])
T02     = T0_ref
M2      = compute_Mx(P02, P2, gamma)
T2      = compute_Tx(T02, M2, gamma)
V2      = compute_Vx(M2, gamma, R, T2)
rho2    = compute_rhox(P2, T2, R)
mu2     = compute_mux(T2)
nu2     = compute_nux(mu2, rho2)
Re2     = compute_Rex(rho2, V2, axial_chord, mu2)
#k2, epsilon2, omega2, nu_factor2 = compute_TurbulentQtys(TI_iso2, Umean2, ILS_avr2, nu2)

# Outlet conditions
M6  = M / 100
Re6 = Re * 1000
P6  = 7771.16 #P6_test 
#P06 = P01 * df_exp_PL06_pa['P06/P01'].iloc[-1]
#P6  = P01 * df_exp_PL06_pa['Ps6/P01'].iloc[-1]

# Inlet flow angle
betta1 = 37.3 #36.69  # From Table 2.2 - Measurement Techniques 


#---- MESH CONSTRUCTION ----
# Boundary parameters
alpha1 = alpha_m_in
alpha2 = alpha_m_out 
dist_PL01 = 1.12
dist_PL02 = 0.50
dist_PL06 = 1.50
dist_inlet = 2          # How many axial chords upstream will the inlet be placed
dist_outlet = 3        # How many axial chords downstream will the outlet be placed 


#---- BC Print ----
print("BOUNDARY CONDITIONS")
print("Inlet Total Temperature [K]:", T02)
print("Inlet Total Pressure [Pa]:", P02)
print("Inlet Static Temperature [K]:", T2)
print("Inlet Static Pressure [Pa]:", P2)
print("Inlet Mach number:", M2)
print("Outlet Mach number:", M6)
print("Inlet Reynolds Number:", Re2)
print("Outlet Reynolds Number:", Re6)
print("Inlet Flow Angle:", betta1)
#print("Inlet NuFactor:", nu_factor2)


#####################################################################################
#                           Spleen Boundary Conditions                              #
##################################################################################### 



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
    P0 = [x_ss0[-1], y_ss0[-1]]    # TE-SS
    P3 = [x_ps0[-1], y_ps0[-1]]    # TE-PS
    dist = np.hypot(P3[0]-P0[0], P3[1]-P0[1])
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

                                                          
# --------------------------- AIRFOIL & PARAMETERS ---------------------------
AIRFOIL_FILE = airfoil_file_path

if AIRFOIL_FILE == airfoil_file_path2 :
    alpha1 = -50
    alpha2 = 50
    axial_chord = 0.999889
    pitch = 0.790391
else:
    alpha1 = 37.3
    alpha2 = -53.80
    axial_chord = 47.614e-3
    pitch = 32.950e-3

# --------------------------- GENERAL MESH PARAMETERS ---------------------------
sizeCellFluid: float = 0.02 * axial_chord       # Fluid related cell size
sizeCellAirfoil: float = 0.02 * axial_chord     # Airfoil related cell size
nCellAirfoil: int = 549 # 525                   # BL lines number of cells in y
nCellPerimeter: int = 183                       # BL and Mesh Boundary number of cells
nBoundaryPoints = 50                            # For boundary number of points selection in the BL grid edge
    
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
    out = process_airfoil_file(AIRFOIL_FILE, n_points=1000, n_te=60, d_factor=0.5)
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
        f.write("Field[1].EdgesList   = {1000, 2000};   // SS & PS splines\n")
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

#%%

#####################################################################################
#                                                                                   #
#                              SU2 CONFIG FILE CREATION                             #
#                                                                                   #
##################################################################################### 


#################################################################################
#  AIRFOIL file creation           
#################################################################################

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
%SA_OPTIONS              = BCM
KIND_TRANS_MODEL        = NONE
MATH_PROBLEM            = DIRECT
RESTART_SOL             = NO


% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%
MACH_NUMBER                     = {M1}              % Inlet Mach number
AOA                             = {betta1}          % Midspan cascade aligned with the flow
FREESTREAM_PRESSURE             = {P01}              % Free-stream static pressure in Pa
FREESTREAM_TEMPERATURE          = {T01}              % Free-stream static temperature
REYNOLDS_NUMBER                 = {Re1}             % Free-stream Reynolds number
REYNOLDS_LENGTH                 = {axial_chord}     % Normalization length
FREESTREAM_TURBULENCEINTENSITY  = 0.001              % (If SST used) freestream turbulence intensity (2% as example)
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

MARKER_INLET            = ( inlet, {T01}, {P01}, {np.cos(betta1 * np.pi / 180)}, {np.sin(betta1 * np.pi / 180)}, 0)
MARKER_OUTLET           = ( outlet,  {P6})
MARKER_PERIODIC         = ( simmetricWallsBOTTOM, simmetricWallsTOP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {pitch}, 0.0 )
%MARKER_INLET_TURBULENT  = ( inlet, TI2, nu_factor2 )          %SST Model
MARKER_INLET_TURBULENT  = ( inlet,  0.1 )                %SA Model


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
CONV_RESIDUAL_MINVAL    = -10                        % Can try lower but unnecessary
CONV_FIELD              = RMS_DENSITY               % Can also try MASSFLOW
CONV_STARTITER          = 500                       % Original 500
ITER                    = 4000


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


% -------------------------- MESH SMOOTHING -----------------------------%
%SMOOTH_GEOMETRY= 0                     % Before each computation, implicitly smooth the nodal coordinates


% ----------------------- GEOMETRY EVALUATION PARAMETERS ----------------------% ???
% ---------------- MESH DEFORMATION PARAMETERS (NEW SOLVER) -------------------% ???
% ---------- NO THERMAL CONDUCTIVITY IS CONSIDERED WITHIN THE AIRFOIL ---------%

'''

    # Write the information to the AIRFOIL file
    with open(f"cascade2D{string}_{bladeName}.cfg", "w") as f:
        f.write(data_airfoil)    

print(f"Config file created: cascade2D{string}_{bladeName}.cfg")


# Run the main function
if __name__ == "__main__":
    configFile()

    
#raise SystemExit("Stopping script execution.")


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
#%%
    #################################################################################
    #  HISTORY FILE TRACKING - Residuals, Linear Solvers, CFL, CD, CL           
    #################################################################################
    
    hist = pd.read_csv(f'history_{string}_{bladeName}.csv')
    
    # RMS Tracking
    plt.plot(hist['Inner_Iter'], hist['    "rms[Rho]"    '], label='ρ')                     # Density
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoU]"   '], label='ρu')                    # Momentum-x
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoE]"   '], label='ρE')                    # Energy
    #plt.plot(hist['Inner_Iter'], hist['     "rms[nu]"    '], label='v')                     # Viscosity
    #plt.plot(hist['Inner_Iter'], hist['     "rms[k]"    '], label='k')                     # TKE
    #plt.plot(hist['Inner_Iter'], hist['     "rms[w]"    '], label='w')                     # Dissipation
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
    '''
    # Total Pressure Tracking
    plt.plot(hist['Inner_Iter'], hist['Avg_TotalPress(blade1)'], label='AvgTotalPress')     # Total Pressure across Blade
    plt.grid(alpha=0.3);  plt.legend();  plt.xlabel('Iteration')
    plt.ylabel('Average Total Pressure - Airfoil');  plt.tight_layout();  plt.show()
    '''
#raise SystemExit("Stopping script execution.")

#%%

#####################################################################################
#                                                                                   #
#                                RESULT VISUALIZATION                               #
#                                                                                   #
##################################################################################### 

# ─────────────────────────────────────────────────────────────────────────────
#   BASIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def surface_fraction(x, y):
    """
    Calculate normalized surface distance from leading edge to trailing edge.
    
    Parameters:
    x, y: arrays of airfoil surface coordinates
    
    Returns:
    s_norm: normalized surface distance (0 to 1)
    """
    # Calculate incremental distances between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Calculate cumulative surface distance
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    
    # Normalize by total surface length
    s_norm = s / s[-1]
    
    return s_norm

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
    
def SU2_extract_plane_data(df, x_plane, pitch, alpha_m, atol=1e-3):
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
    sub_df['total_pressure'] = sub_df['Pressure'] * (1 + 0.5*(gamma-1)*sub_df['Mach']**2)**(gamma/(gamma-1))
    sub_df['y_norm'] = sub_df['y'] / pitch  # Normalize y
    sub_df['flow_angle'] = np.atan2(sub_df['Velocity_y'], sub_df['Velocity_x']) * 180 / np.pi - alpha_m
    
    # Sort by normalized y for clean plots
    sub_df = sub_df.sort_values('y_norm').reset_index(drop=True)
    return sub_df    

def tile_plane_data(sim_df: pd.DataFrame, exp_min: float, exp_max: float, pitch: float) -> pd.DataFrame:
    """Return a tiled & windowed copy of *sim_df* that fully covers the experimental
    span ``[exp_min , exp_max]`` in *y/pitch* coordinates.

    The incoming *sim_df* must be the output from ``SU2_extract_plane_data`` and
    therefore contain a ``y_norm`` column (``y / pitch`` in the original SU2
    plane).  The routine replicates that entire dataframe upward and downward by
    integer multiples of *pitch*, assigns a new absolute ordinate
    ``y_rolled = y_norm + k·pitch`` for each replica, then keeps only the rows
    whose ``y_rolled`` lie inside the experimental window.  Finally, the rows
    are returned **sorted by ``y_rolled``** so they can be plotted directly.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Data returned by ``SU2_extract_plane_data`` — must include ``y_norm``.
    exp_min, exp_max : float
        Minimum and maximum experimental ordinates (already non‑dimensionalised
        as *y/pitch*).
    pitch : float, default 1.0
        Pitch length used to convert shifts; keep at 1.0 if the input ``y_norm``
        is already non‑dimensionalised by pitch.

    Returns
    -------
    pandas.DataFrame
        Copy of *sim_df* with an extra column **``y_rolled``** filtered to the
        experimental window and sorted.
    """
    if 'y_norm' not in sim_df.columns:
        raise KeyError("'y_norm' column required – did you call SU2_extract_plane_data?")

    if exp_max <= exp_min:
        raise ValueError('exp_max must be greater than exp_min')

    sim_min = sim_df['y_norm'].min()
    sim_max = sim_df['y_norm'].max()

    # How many upward shifts are needed so that the *top* of the clone crosses exp_max?
    k_needed = int(np.ceil((exp_max - sim_max) / pitch))

    # Build that single clone
    clone = sim_df.copy()
    clone['y_rolled'] = clone['y_norm'] + k_needed * pitch
    clone['pitch_shift'] = k_needed

    # Trim to the experimental bounds
    window = clone[(clone['y_rolled'] >= exp_min) & (clone['y_rolled'] <= exp_max)].copy()
    return window.sort_values('y_rolled').reset_index(drop=True)

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
        plt.scatter(exp_x, exp_mach, s=20, color='red', label='Exp. Data')

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

def plot_plane_comparison(sim_df: pd.DataFrame, exp_df: pd.DataFrame, *,
                          exp_y_col: str, exp_val_col: str,
                          field: str, plane_str: str, title: str | None = None):
    """Scatter‑plot SU2 vs experimental data for a given plane.

    If *sim_df* contains a ``y_rolled`` column, it is used on the abscissa;
    otherwise the routine falls back to ``y_norm``.  All other columns are
    forwarded untouched.
    """
    x_key = 'y_rolled' if 'y_rolled' in sim_df.columns else 'y_norm'

    plt.figure(figsize=(10, 6))
    plt.scatter(sim_df[x_key], sim_df[field], label='SU2', s=30, zorder=10)
    plt.scatter(exp_df[exp_y_col], exp_df[exp_val_col], label='SPLEEN',
                c='red', s=30, zorder=10)

    plt.xlabel('y / pitch')
    plt.ylabel(field)
    plt.title(title or f'{field} at {plane_str}')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

###############################################################################
#                            MAIN FUNCTION                                    #
###############################################################################


def surfaceFlowAnalysis(string):
    """
    - Loads SU2 surface data (no smoothing).
    - Scans *all* experimental files, finds the one that minimises the combined
      suction‑plus‑pressure RMS (%) error versus SU2 (linear interpolation).
    - Prints that file‑ID and the RMS, then overlays its Mach curve with SU2.
    - All existing plots are preserved.
    """
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   SU2 DATA
    # ─────────────────────────────────────────────────────────────────────────────
    
    # ---------- 1) SU2 Data Extraction -------------------
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
    #   SPLEEN DATA
    # ─────────────────────────────────────────────────────────────────────────────
    
    # ---------- 1) SPLEEN Data Extraction -------------------
    fileExtension = 'xlsx'
    
    Loc_test =  'PL01' #-----------------------
    Inst_test = 'BLT'
    fileIDs = 's5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath              = os.path.join(filesPL01, testRun)
    df_exp_PL01_BLT          = pd.read_excel(df_dataPath, usecols=[1,3])
    df_exp_PL01_BLT.columns  = ['y/g','P0_BL/P01']
        
    Loc_test =  'PL02' #-----------------------
    Inst_test = 'C5HP'
    fileIDs = 's5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath                 = os.path.join(filesPL02, testRun)
    df_exp_PL02_C5HP            = pd.read_excel(df_dataPath, usecols=[1,3,4,5,6,7,8])
    df_exp_PL02_C5HP.columns    = ['y/g','i','pitch','rho','V_ax','P02/P01','Ps2/P01']
    df_exp_PL02_C5HP_mod        = df_exp_PL02_C5HP.copy()
    df_exp_PL02_C5HP_mod['P02'] = df_exp_PL02_C5HP_mod['P02/P01'] * P01
    df_exp_PL02_C5HP_mod['Ps2'] = df_exp_PL02_C5HP_mod['Ps2/P01'] * P01
    '''
    Inst_test = 'FRV4H_Turbulence'
    fileIDs = 'S5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath                    = os.path.join(filesPL02, testRun)
    df_exp_PL02_FRV4H_Turb         = pd.read_excel(df_dataPath, usecols=[1,3,4,5])
    df_exp_PL02_FRV4H_Turb.columns = ['y/g','TKE','TI-mean','ILS']
    
    Inst_test = 'XW'
    fileIDs = 's5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath            = os.path.join(filesPL02, testRun)
    df_exp_PL02_XW         = pd.read_excel(df_dataPath, usecols=[1,3,4,5,6,7])
    df_exp_PL02_XW.columns = ['y/g','TI','TI_ISO','U_mean','ILS_1','ILS_2']
    '''
    Loc_test =  'PL06' #-----------------------
    Inst_test = 'L5HP'
    fileIDs = 's5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath              = os.path.join(filesPL06, testRun)
    df_exp_PL06_L5HP         = pd.read_excel(df_dataPath, usecols=[1,3,4,5,6,7,8,9])
    df_exp_PL06_L5HP.columns = ['y/g','d','pitch','rho','V_ax','P06/P01','Ps6/P01','ksi']
    df_exp_PL06_L5HP_mod        = df_exp_PL06_L5HP.copy()
    df_exp_PL06_L5HP_mod['P06'] = df_exp_PL06_L5HP_mod['P06/P01'] * P01
    df_exp_PL06_L5HP_mod['Ps6'] = df_exp_PL06_L5HP_mod['Ps6/P01'] * P01
    df_exp_PL06_L5HP_mod['Loss'] = (P01 - df_exp_PL06_L5HP_mod['P06']) / P01
    '''
    Inst_test = 'FRV4H_Turbulence'
    fileIDs = 'S5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath                    = os.path.join(filesPL06, testRun)
    df_exp_PL06_FRV4H_Turb         = pd.read_excel(df_dataPath, usecols=[1,3,4,5])
    df_exp_PL06_FRV4H_Turb.columns = ['y/g','TKE','TI-mean','ILS']
    
    Inst_test = 'FRV4H'
    fileIDs = 's5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath                  = os.path.join(filesPL06, testRun)
    df_exp_PL06_FRV4H            = pd.read_excel(df_dataPath, usecols=[1,3,4,5,6,7,8])
    df_exp_PL06_FRV4H.columns    = ['y/g','d','pitch','rho','V_ax','P06/P01','Ps6/P01']
    df_exp_PL06_FRV4H_mod        = df_exp_PL06_FRV4H.copy()
    df_exp_PL06_FRV4H_mod['P06'] = df_exp_PL06_FRV4H_mod['P06/P01'] * P01
    df_exp_PL06_FRV4H_mod['Ps6'] = df_exp_PL06_FRV4H_mod['Ps6/P01'] * P01
    '''
    Loc_test =  'Blade' #-----------------------
    Inst_test = 'PT'
    fileIDs = 's4970'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath             = os.path.join(filesBlade, testRun)
    df_exp_Blade_PT         = pd.read_excel(df_dataPath, usecols=[1,2,3])
    df_exp_Blade_PT.columns = ['x/C_ax','S/S_l','Ps/P01']
    df_exp_Blade_PT_mod     = df_exp_Blade_PT.copy()
    df_exp_Blade_PT_mod     = df_exp_Blade_PT['Ps/P01'] * P01
    
    Inst_test = 'HF'
    fileIDs = 'S5000'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath             = os.path.join(filesBlade, testRun)
    df_exp_Blade_HF         = pd.read_excel(df_dataPath, usecols=[1,2,3,4,5,6,7,8,9,10])
    df_exp_Blade_HF.columns = ['x/C_ax','S/S_l','E','STDE','SKEWE','KURTE','QSS','STDQSS','SKEWQSS','KURTQSS']
    '''
    Inst_test = 'FR'
    fileIDs = 'S4970'
    testRun = f"SPLEEN_C1_NC_St{St_test}_Re{Re_test}_M{M_test}_{Loc_test}_{Inst_test}_{fileIDs}.{fileExtension}"
    df_dataPath             = os.path.join(filesBlade, testRun)
    df_exp_Blade_FR         = pd.read_excel(df_dataPath, usecols=[1,2,3,4,5,6])
    df_exp_Blade_FR.columns = ['x/C_ax','S/S_l','Ps/P01','STDP/P01','SKEWP/P01','KURTP/P01']
    '''
    
    # ---------- 2) SPLEEN DATA HANDLING - we compute other qtys  --------------
    
    inside_exp = (2.0/(gamma-1)) * (df_exp_Blade_PT['Ps/P01']**((1-gamma)/gamma) - 1.0)
    df_exp_Blade_PT['Mach'] = np.sqrt(np.clip(inside_exp, 0.0, None))
    
    # --------- Split EXP into SS / PS fractions 
    ss_mask = df_exp_Blade_PT['S/S_l'] >= 0
    ps_mask = ~ss_mask

    # Suction side
    ss_frac  = df_exp_Blade_PT.loc[ss_mask, 'S/S_l'].to_numpy()          # 0 ➔ 1
    ss_mach  = df_exp_Blade_PT.loc[ss_mask, 'Mach'   ].to_numpy()

    # Pressure side  (use +ve fraction)
    ps_frac  = -df_exp_Blade_PT.loc[ps_mask, 'S/S_l'].to_numpy()         # 0 ➔ 1
    ps_mach  =  df_exp_Blade_PT.loc[ps_mask, 'Mach'   ].to_numpy()
    
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
    
    best_mach   = df_exp_Blade_PT['Mach'].to_numpy()
    best_scurve = df_exp_Blade_PT['S/S_l'].to_numpy()
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   PLOTTING
    # ─────────────────────────────────────────────────────────────────────────────
    
    # ---------- Single overlay of Mach
    exp_x_example    = np.abs(best_scurve)    # <<< flip to +x
    exp_mach_example = best_mach
    
    SU2_DataPlotting(
        sSSnorm     = s_normSS,
        sPSnorm     = s_normPS,
        dataSS      = machSS,
        dataPS      = machPS,
        quantity    ="Mach Number",
        string      = string,
        mirror_PS   = False,
        exp_x       = exp_x_example,
        exp_mach    = exp_mach_example
    )
    
    SU2_DataPlotting(s_normSS, s_normPS, yPlusSS, yPlusPS,
                 "Y Plus", string, mirror_PS=True)
    
    SU2_DataPlotting(s_normSS, s_normPS, friction_coeffSS, friction_coeffPS,
                 "Skin Friction Coefficient", string, mirror_PS=True)
    
    
    def plot_su2_field(fieldSS, fieldPS, field):
        SU2_DataPlotting(
            sSSnorm     = s_normSS,
            sPSnorm     = s_normPS,
            dataSS      = fieldSS,
            dataPS      = fieldPS,
            quantity    = field,
            string      = string            
        )
    
    #plot_su2_field(temperatureSS, temperaturePS, "Temperature")
    #plot_su2_field(densitySS, densityPS, "Density")
    #plot_su2_field(energySS, energyPS, "Energy")
    
    #plot_su2_field(pressure_coeffSS, pressure_coeffPS, "Pressure Coefficient")
    #plot_su2_field(yPlusSS, yPlusPS, "Y Plus", sSSnorm=s_normSS,  sPSnorm=s_normPS_mirr)
    #plot_su2_field(friction_coeffSS, friction_coeffPS, "Skin Friction Coefficient", sSSnorm=s_normSS,  sPSnorm=s_normPS_mirr)
    
    
    # ---------- 7) Compare PL02 - PL06 results ------------------------------
    
    # Plane locations
    xPL02 = -dist_PL02 * axial_chord
    xPL06 =  dist_PL06 * axial_chord
    restart_file = os.path.join(current_directory, 'restart_flowdatabladeVALIDATION_SPLEEN.csv')
    
    # --- SU2 Extraction
    df = pd.read_csv(restart_file, sep=',')
    pl02_sim = SU2_extract_plane_data(df, xPL02, pitch, alpha_m_in)
    pl06_sim = SU2_extract_plane_data(df, xPL06, pitch, alpha_m_out)
    
    '''
    # --- PL02 Plots
    plot_plane_comparison(
        pl02_sim, df_exp_PL02_C5HP_mod, exp_y_col='y/g', exp_val_col='rho',
        field='Density', plane_str='PL02', title='Density at PL02')
    plot_plane_comparison(
        pl02_sim, df_exp_PL02_C5HP_mod, exp_y_col='y/g', exp_val_col='Ps2',
        field='Pressure', plane_str='PL02', title='Static Pressure at PL02')
    plot_plane_comparison(
        pl02_sim, df_exp_PL02_C5HP_mod, exp_y_col='y/g', exp_val_col='i',
        field='flow_angle', plane_str='PL02', title='Incidence Angle at PL02')

    plot_plane_comparison(
        pl02_sim, df_exp_PL02_FRV4H_Turb, exp_y_col='y/g', exp_val_col='TKE',
        field='Turb_Kin_Energy', plane_str='PL02', title='TKE at PL02')
    plot_plane_comparison(
        pl02_sim, df_exp_PL02_FRV4H_Turb, exp_y_col='y/g', exp_val_col='TI-mean',
        field='Turb_index', plane_str='PL02', title='TI at PL02')
    '''
    # --- PL06 Plots
    ymin_pl06, ymax_pl06 = df_exp_PL06_L5HP_mod['y/g'].agg(['min', 'max'])
    pl06_sim_window = tile_plane_data(pl06_sim, exp_min=ymin_pl06, exp_max=ymax_pl06, pitch=pitch)
    pl06_sim_window['Pressure Defect'] = (P01 - pl06_sim_window['total_pressure']) / P01
    
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_L5HP_mod, exp_y_col='y/g', exp_val_col='rho',
        field='Density', plane_str='PL06', title='Density at PL06')
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_L5HP_mod, exp_y_col='y/g', exp_val_col='Ps6',
        field='Pressure', plane_str='PL06', title='Static Pressure at PL06')
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_L5HP_mod, exp_y_col='y/g', exp_val_col='Loss',
        field='Pressure Defect', plane_str='PL06', title='Pressure Defect at PL06')
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_L5HP_mod, exp_y_col='y/g', exp_val_col='d',
        field='flow_angle', plane_str='PL06', title='Incidence Angle at PL06')
    '''
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_FRV4H_Turb, exp_y_col='y/g', exp_val_col='TKE',
        field='Turb_Kin_Energy', plane_str='PL06', title='TKE at PL06')
    plot_plane_comparison(
        pl06_sim_window, df_exp_PL06_FRV4H_Turb, exp_y_col='y/g', exp_val_col='TI-mean',
        field='Turb_index', plane_str='PL06', title='TI at PL06')
    '''

    
    
###############################################################################
#                          EXECUTE THE CODE                                   #
###############################################################################

if __name__ == "__main__":
    surfaceFlowAnalysis('databladeVALIDATION')
