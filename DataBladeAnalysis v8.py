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
import logging
import argparse
import datablade_utils as utils
import datablade_post as post
from pathlib import Path
# pyOCC imports
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.gp import gp_Pnt

# ---------------------------------------------------------------------------
#  CLI and logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_file):
    "Configure the root logger and echo to stdout."
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    "Return parsed command line arguments."
    parser = argparse.ArgumentParser(description="Run DataBlade pipeline")
    parser.add_argument("--blade", default="blade", help="Blade directory name")
    parser.add_argument("--cores", type=int, default=12, help="MPI cores for SU2")
    return parser.parse_args()

# ---------------------------------------------------------------------------
#  INITIALIZATION
# ---------------------------------------------------------------------------



# You will need the following files to run this analysis:
# - Ises file
# - Blade file
# - Gridpar file
# - Mach Distribution file
# Others to be determined when checking other files

# --------------------------- FILE SELECTION ---------------------------

bladeName = "blade"  # Change this name depending on the blade you want to study
no_cores = 12         # MPI cores for SU2
string = "databladeVALIDATION"  # File names suffix
string2 = "safe_start"          # For SU2 optimization
fileExtension = "csv"

# Directory layout


# ────────────────────────────────────────────────────────────────────────────────
# 1) I/O helpers
# ────────────────────────────────────────────────────────────────────────────────



def mesh_datablade():       
    
    # --------------------------- GEOMETRY EXTRACTION ---------------------------
    out = process_airfoil_file(bladeFilePath, n_points=1000, n_te=60, d_factor=d_factor)
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

    geo_file = run_dir / f"cascade2D{string}_{bladeName}.geo"
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
        f.write('Physical Curve("symmetricWallsBOTTOM", 18002) = {150};\n')
        f.write('Physical Curve("symmetricWallsTOP",    18003) = {152};\n')
        f.write('Physical Curve("outlet", 18004) = {151};\n')
        f.write('Physical Curve("blade1", 18005) = {2000, 1000};\n')
        f.write('Physical Surface("fluid", 18008) = {5};\n') 
        
        
    logging.info(f"Geo file written at: {geo_file}")
    
    # Run gmsh to generate the SU2 mesh.
    logging.info("STARTING mesh generation...")
    try:
        if os.path.exists(geo_file):
            logging.info(f"File exists at: {geo_file}")
        else:
            logging.info(f"File not found at: {geo_file}")
            
        os.system(f'gmsh "{geo_file}" -2 -format su2')
        logging.info("Mesh successfully created!")
    except Exception as e:
        logging.error("Error", e)       

    
#%%

#####################################################################################
#                                                                                   #
#                              SU2 CONFIG FILE CREATION                             #
#                                                                                   #
##################################################################################### 


def configSU2_datablade():
 
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
MARKER_PERIODIC         = ( symmetricWallsBOTTOM, symmetricWallsTOP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {pitch}, 0.0 )
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
    with open(run_dir / f"cascade2D{string}_{bladeName}.cfg", "w") as f:
        f.write(data_airfoil)

# Run the main function

    


def run_SU2():
    """Launch SU2 using ``mpiexec`` and log the output."""
    config_file = run_dir / f"cascade2D{string}_{bladeName}.cfg"
    if not config_file.exists():
        logging.warning("Config file not found at: %s", config_file)
        return
    logging.info("Running SU2 with %s cores", no_cores)
    try:
        subprocess.run([
            "mpiexec",
            "-n",
            str(no_cores),
            "SU2_CFD",
            str(config_file),
        ], check=True)
        logging.info("SU2 run finished")
    except subprocess.CalledProcessError as exc:
        logging.error("SU2 failed: %s", exc)

def surfaceFlowAnalysis_datablade():
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   SU2 DATA
    # ─────────────────────────────────────────────────────────────────────────────

    su2_file = run_dir / f"surface_flow{string}_{bladeName}.csv"
    df = pd.read_csv(su2_file, sep=',')
    x      = df['x'].values
    y      = df['y'].values
    xNorm  = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    '''
    x      = df['x'].values
    y      = df['y'].values
    s_norm = post.surface_fraction(x,y)
    pressure        = df['Pressure'].values
    pressure_coeff  = df['Pressure_Coefficient'].values
    friction_coeff  = df['Skin_Friction_Coefficient_x'].values
    yPlus           = df['Y_Plus'].values
    
    temperature     = df['Temperature'].values
    density         = df['Density'].values
    energy          = df['Energy'].values
    laminar_visc    = df['Laminar_Viscosity'].values
    '''
    
    _, _, dataSS, dataPS = post.SU2_organize(df)
    
    # Suction Side - Upper Surface
    xSS                 = dataSS['x'].values
    ySS                 = dataSS['y'].values
    s_normSS            = post.surface_fraction(xSS,ySS)
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
    s_normPS            = post.surface_fraction(xPS,yPS)
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
    resultsFilePath = blade_dir / resultsFileName
    
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
    
    logging.info(f"\nCombined RMS error = {rms_pct:.2f}%")
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   PLOTTING
    # ─────────────────────────────────────────────────────────────────────────────
    
    # ---------- Single overlay of Mach
    
    post.SU2_DataPlotting(
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
    
    post.SU2_DataPlotting(s_normSS, s_normPS, yPlusSS, yPlusPS,
                 "Y Plus", string, mirror_PS=True)
    
    post.SU2_DataPlotting(s_normSS, s_normPS, friction_coeffSS, friction_coeffPS,
                 "Skin Friction Coefficient", string, mirror_PS=True)
    
    

def main():
    args = parse_args()
    global bladeName, no_cores, base_dir, blade_dir, run_dir
    global isesFilePath, bladeFilePath, outletFilePath
    global d_factor, pitch, axial_chord, alpha1, alpha2
    bladeName = args.blade
    no_cores = args.cores
    base_dir = Path(__file__).resolve().parent
    blade_dir = base_dir / "Blades" / bladeName
    run_root = blade_dir / "results"
    run_root.mkdir(exist_ok=True)
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    n = 1
    while (run_root / f"Test_{n}_{date_str}").exists():
        n += 1
    run_dir = run_root / f"Test_{n}_{date_str}"
    run_dir.mkdir()
    setup_logging(run_dir / "pipeline.log")

    isesFilePath = blade_dir / f"ises.{string}"
    bladeFilePath = blade_dir / f"{bladeName}.{string}"
    outletFilePath = blade_dir / f"outlet.{string}"

    alpha1, alpha2, Re = utils.extract_from_ises(isesFilePath)
    pitch = utils.extract_from_blade(bladeFilePath)
    M1, P21_ratio = utils.extract_from_outlet(outletFilePath)

    geom0 = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=0.5)
    d_factor = utils.compute_d_factor(
        wedge_angle_deg=np.degrees(geom0['wedge_angle']),
        axial_chord=geom0['axial_chord'],
        te_thickness=geom0['te_open_thickness'])
    logging.info(f"Updated d_factor = {d_factor:.3f}")
    geom = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=d_factor)
    stagger = geom['stagger_angle']
    axial_chord = geom['axial_chord']
    logging.info("Stagger angle (deg): %s", stagger)
    logging.info("Axial chord: %s", axial_chord)

    R = 287.058
    gamma = 1.4
    mu = 1.716e-5
    T01 = 314.15
    T1 = T01 / (1 + (gamma - 1)/2 * M1**2)
    c1 = np.sqrt(gamma * R * T1)
    u1 = M1 * c1
    rho1 = mu * Re / (u1 * np.cos(stagger))
    P1 = rho1 * R * T1
    P01 = P1 * (1 + (gamma - 1)/2 * M1**2)**(gamma/(gamma - 1))
    P2 = P21_ratio * P1
    logging.info(f"Upstream pressure is : {P01}")
    logging.info(f"Outlet pressure is : {P2}")

    alpha1 = int(np.degrees(np.arctan(alpha1)))
    alpha2 = int(np.degrees(np.arctan(alpha2)))
    dist_inlet = 2
    dist_outlet = 3
    TI2 = 2.2

    global sizeCellFluid, sizeCellAirfoil, nCellAirfoil, nCellPerimeter
    global nBoundaryPoints, first_layer_height, bl_growth, bl_thickness
    global size_LE, dist_LE, size_TE, dist_TE
    global VolWAkeIn, VolWAkeOut, WakeXMin, WakeXMax
    sizeCellFluid = 0.02 * axial_chord
    sizeCellAirfoil = 0.02 * axial_chord
    nCellAirfoil = 549
    nCellPerimeter = 183
    nBoundaryPoints = 50
    first_layer_height = 0.01 * sizeCellAirfoil
    bl_growth = 1.17
    bl_thickness = 0.03 * pitch
    size_LE = 0.1 * sizeCellAirfoil
    dist_LE = 0.01 * axial_chord
    size_TE = 0.1 * sizeCellAirfoil
    dist_TE = 0.01 * axial_chord
    VolWAkeIn = 0.35 * sizeCellFluid
    VolWAkeOut = sizeCellFluid
    WakeXMin = -0.1 * axial_chord
    WakeXMax = (dist_outlet - 1.5) * axial_chord

    mesh_datablade()
    configSU2_datablade()
    run_SU2()
    surfaceFlowAnalysis_datablade()

if __name__ == "__main__":
    main()
