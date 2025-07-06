import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


def configSU2_datablade():
 
    data_airfoil = f'''

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                   
%                                                                              %
% SU2 AIRFOIL configuration file                                               %
% Case description: General Airfoil                                            %
% Author: Freddy Chica	                                                       %
% Institution: Université Catholique de Louvain                                %
% Date: 11, Nov 2024                                                           %
% File Version: 9                                                              %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
SOLVER                  = RANS
KIND_TURB_MODEL         = SA
SA_OPTIONS              = BCM
KIND_TRANS_MODEL        = NONE                        % NONE or LM
MATH_PROBLEM            = DIRECT
RESTART_SOL             = NO


% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%
MACH_NUMBER                     = {M1}              % Inlet Mach number
AOA                             = {alpha1}          % Midspan cascade aligned with the flow
FREESTREAM_PRESSURE             = {P01}             % Free-stream static pressure in Pa
FREESTREAM_TEMPERATURE          = {T01}             % Free-stream static temperature
REYNOLDS_NUMBER                 = {Re}              % Free-stream Reynolds number
REYNOLDS_LENGTH                 = {axial_chord}     % Normalization length
FREESTREAM_TURBULENCEINTENSITY  = {TI/100} % 0.001  % (If SST used) freestream turbulence intensity (2% as example)
FREESTREAM_TURB2LAMVISCRATIO    = 0.1  %10          % (If SST used) ratio of turbulent to laminar viscosity
%FREESTREAM_NU_FACTOR            = 3                % (For SA) initial turbulent viscosity ratio (default 3)
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
%MARKER_INLET_TURBULENT = ( inlet, TI2, nu_factor2 )                        %SST Model
%MARKER_INLET_TURBULENT = ( inlet,  nu_factor2 )                            %SA Model


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
CONV_RESIDUAL_MINVAL    = -7                        % Can try lower but unnecessary
CONV_FIELD              = RMS_DENSITY               % Can also try MASSFLOW
CONV_STARTITER          = 500                       % Original 500
ITER                    = 7000


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
MESH_FILENAME           = cascade2D_{string}_{bladeName}.su2
MESH_FORMAT             = SU2
MESH_OUT_FILENAME       = cascade2D_{string}_out_{bladeName}.su2
SOLUTION_FILENAME       = restart_flow_{string}_{bladeName}.dat

% OUTPUTS
OUTPUT_FILES            = (RESTART, PARAVIEW, SURFACE_PARAVIEW, CSV, SURFACE_CSV)
CONV_FILENAME           = history_{string}_{bladeName}
RESTART_FILENAME        = restart_flow_{string}_{bladeName}.dat
VOLUME_FILENAME         = volume_flow_{string}_{bladeName}
SURFACE_FILENAME        = surface_flow{string}_{bladeName}
%GRAD_OBJFUNC_FILENAME  = of_grad{string}_{bladeName}.dat

'''

    # Write the information to the AIRFOIL file
    with open(run_dir / f"cascade2D{string}_{bladeName}.cfg", "w") as f:
        f.write(data_airfoil)

def runSU2_datablade():
    # Run SU2 simulation using the config file.
    config_file = run_dir / f"cascade2D{string}_{bladeName}.cfg"
    try:
        if os.path.exists(config_file):
            print(f"Config file exists at: {config_file}")
        else:
            print(f"Config file not found at: {config_file}")
            return
        
        print("SU2 Run Initialized!")
        # Save current working directory
        orig_dir = os.getcwd()
        # Change to the directory where the config file is located
        os.chdir(run_dir)
        
        # Run SU2 from the config directory and capture output
        log_file = run_dir / "su2.log"
        with open(log_file, "w") as logf:
            subprocess.run(
                ["mpiexec", "-n", str(no_cores), "SU2_CFD", str(config_file)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        
        # Build summary from log
        begin_marker = "------------------------------ Begin Solver -----------------------------"
        exit_markers = [
            "----------------------------- Solver Exit -------------------------------",
            "------------------------------ Error Exit -------------------------------",
        ]
        with open(log_file, "r") as logf:
            log_lines = logf.readlines()

        begin_idx = next((i for i, l in enumerate(log_lines) if begin_marker in l), len(log_lines))

        exit_idx = len(log_lines)
        for marker in exit_markers:
            found = next((i for i, l in enumerate(log_lines) if marker in l), None)
            if found is not None:
                exit_idx = found
                break

        summary_lines = log_lines[:begin_idx]
        summary_lines += log_lines[max(0, exit_idx - 5):]

        summary_file = run_dir / "run_summary.txt"
        with open(summary_file, "w") as f:
            f.writelines(summary_lines)
    except Exception as e:
        print("Error", e)
    finally:
        print("SU2 Run Finalized!")
        # Return to the original working directory
        os.chdir(orig_dir)
        