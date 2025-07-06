import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}"
})


def post_processing_datablade():
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   HISTORY FILE TRACKING - Residuals, Linear Solvers, CFL, CD, CL
    # ─────────────────────────────────────────────────────────────────────────────
    
    residuals_file = run_dir / f'history_{string}_{bladeName}.csv'
    hist = pd.read_csv(residuals_file)

    # RMS Tracking
    plt.plot(hist['Inner_Iter'], hist['    "rms[Rho]"    '], label=r'$\rho$')               # Density
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoU]"   '], label=r'$\rho u$')             # Momentum-x
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoE]"   '], label=r'$\rho E$')             # Energy
    #plt.plot(hist['Inner_Iter'], hist['    "rms[RhoV]"   '], label=r'$\rho v$')            # Momentum-y
    #plt.plot(hist['Inner_Iter'], hist['     "rms[nu]"    '], label='v')                    # Viscosity
    #plt.plot(hist['Inner_Iter'], hist['     "rms[k]"    '], label='k')                     # TKE
    #plt.plot(hist['Inner_Iter'], hist['     "rms[w]"    '], label='w')
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'RMS residual - {bladeName}');
    plt.savefig(run_dir / f'rms_residual_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Linear Solver Tracking
    plt.plot(hist['Inner_Iter'], hist['    "LinSolRes"   '], label='LSRes')                 # Linear Solver Residual
    plt.plot(hist['Inner_Iter'], hist['  "LinSolResTurb" '], label='LSResTurb')             # Linear Solver Residual
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Linear Solver residual - {bladeName}');
    plt.savefig(run_dir / f'linear_solver_residual_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # CFL Tracking
    plt.plot(hist['Inner_Iter'], hist['     "Avg CFL"    '], label='CFL')                   # CFL used per iteration
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Average CFL - {bladeName}');
    plt.savefig(run_dir / f'cfl_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Aero Coefficients Tracking
    plt.plot(hist['Inner_Iter'], hist['   "CD(blade1)"   '], label='CD')                    # Drag Coefficient
    plt.plot(hist['Inner_Iter'], hist['   "CL(blade1)"   '], label='CL')                    # Lift Coefficient
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Aerodynamic Coefficients - {bladeName}');
    plt.savefig(run_dir / f'aero_coefficients_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    
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
    friction_coeffSSx   = dataSS['Skin_Friction_Coefficient_x'].values
    friction_coeffSSy   = dataSS['Skin_Friction_Coefficient_y'].values
    friction_coeffSS    = np.sqrt(friction_coeffSSx**2 + friction_coeffSSy**2)
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
    friction_coeffPSx   = dataPS['Skin_Friction_Coefficient_x'].values
    friction_coeffPSy   = dataPS['Skin_Friction_Coefficient_y'].values
    friction_coeffPS    = np.sqrt(friction_coeffPSx**2 + friction_coeffPSy**2)
    yPlusPS             = dataPS['Y_Plus'].values
    
    temperaturePS       = dataPS['Temperature'].values
    densityPS           = dataPS['Density'].values
    energyPS            = dataPS['Energy'].values
    laminar_viscPS      = dataPS['Laminar_Viscosity'].values
    
    machPS = compute_Mx(P01, pressurePS, gamma)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   MISES DATA
    # ─────────────────────────────────────────────────────────────────────────────
    
    #We read the files in the directory
    mises_machFile  = f"machDistribution.{string}"
    mises_machFile  = blade_dir / mises_machFile
    mises_fieldFile = f"field.{string}"
    mises_fieldFile = blade_dir / mises_fieldFile
    mises_blFile    = f"bl.{string}"
    mises_blFile    = blade_dir / mises_blFile
    
    # machDistribution file data extraction
    if file_nonempty(mises_machFile):
        ps_frac, ss_frac, ps_mach, ss_mach = MISES_machDataGather(mises_machFile)
        if len(ps_frac) and len(ss_frac):
            blade_frac_mach = np.concatenate([ps_frac, ss_frac])
            blade_mach      = np.concatenate([ps_mach, ss_mach])
        else:
            print("[WARNING] MISES mach file contained no valid data; skipping RMS computation")
            blade_frac_mach = None
            blade_mach = None
    else:
        print("[INFO] No MISES mach data found; plotting SU2 results only")
        blade_frac_mach = None
        blade_mach = None
    
    # bl file data extraction
    if file_nonempty(mises_blFile):
        ps_bl, ss_bl = MISES_blDataGather(mises_blFile)
        ps_frac_bl = -ps_bl['s'].values
        ss_frac_bl =  ss_bl['s'].values
        cf_ps = ps_bl['Cf'].values
        cf_ss = ss_bl['Cf'].values
        cf_bl        = np.concatenate([cf_ps, cf_ss])
        blade_frac_bl = np.concatenate([ps_frac_bl, ss_frac_bl])
    else:
        print("[INFO] No MISES boundary layer data found; plotting SU2 results only")
        cf_bl = None
        blade_frac_bl = None
    
    # field file data extraction
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   RMS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────────
    
    if blade_frac_mach is not None:
       # --------- Linear‑interp SU2 onto those fractions
        su2_ss = np.interp(ss_frac, s_normSS, machSS)
        su2_ps = np.interp(ps_frac, s_normPS, machPS)
    
        # --------- Combined RMS
        diff_ss = su2_ss - ss_mach
        diff_ps = su2_ps - ps_mach
        diff_all = np.concatenate([diff_ss, diff_ps])
        rms = np.sqrt(np.nanmean(diff_all**2)) * 100
    
        print(f"\nCombined RMS error = {rms:.4f}%")
        # Record RMS value for later reporting
        summary_file = run_dir / "run_summary.txt"
        try:
            with open(summary_file, "a") as f:
                f.write(f"Mach RMS error: {rms:.4f}\n")
        except OSError:
            print(f"[WARNING] Could not append RMS to {summary_file}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   PLOTTING
    # ─────────────────────────────────────────────────────────────────────────────
    
    SU2_DataPlotting(s_normSS, s_normPS, machSS, machPS,
                 "Mach Number", string, run_dir, bladeName, mirror_PS=False, 
                 exp_s=blade_frac_mach, exp_data=blade_mach)
    
    SU2_DataPlotting(s_normSS, s_normPS, yPlusSS, yPlusPS,
                 "Y Plus", string, run_dir, bladeName, mirror_PS=True)
    
    SU2_DataPlotting(s_normSS, s_normPS, friction_coeffSS, friction_coeffPS,
                 "Skin Friction Coefficient", string, run_dir, bladeName, mirror_PS=True,
                 exp_s=blade_frac_bl, exp_data=cf_bl)

