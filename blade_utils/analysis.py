"""Post-processing utilities for surface flow analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def compute_Mx(P0x, Px, gamma):
    """Return Mach number from total and static pressure."""
    return np.sqrt((2 / (gamma - 1)) * ((P0x / Px) ** ((gamma - 1) / gamma) - 1))


def surface_fraction(xvals, yvals):
    """Normalize arc lengths for plotting."""
    dx = np.diff(xvals)
    dy = np.diff(yvals)
    seg = np.sqrt(dx ** 2 + dy ** 2)
    arc = np.cumsum(seg)
    arc = np.insert(arc, 0, 0.0)
    return arc / arc[-1] if arc[-1] != 0 else arc


def SU2_organize(df):
    """Split SU2 surface CSV into ordered segments."""
    leading_edge = df.iloc[0:1].copy()
    trailing_edge = df.iloc[1:2].copy()
    geo = df.iloc[2:].copy().reset_index(drop=True)
    x, y = geo['x'].values, geo['y'].values
    dist = np.hypot(np.diff(x), np.diff(y))
    idx_break = np.argmax(dist) + 1
    upper_surface = geo.iloc[:idx_break].copy()
    lower_surface = geo.iloc[idx_break:].copy()
    return leading_edge, trailing_edge, upper_surface, lower_surface


def SU2_DataPlotting(sSSnorm, sPSnorm, dataSS, dataPS,
                     quantity, string, mirror_PS=False,
                     exp_x=None, exp_mach=None):
    """Plot SU2 results and optional experimental data."""
    plt.figure(figsize=(14, 9))
    plt.plot(sSSnorm, dataSS, 'o-', ms=2, color='darkblue', label='SU2 (SS)')
    s_ps = -sPSnorm if mirror_PS else sPSnorm
    plt.plot(s_ps, dataPS, 'o-', ms=2, color='lightblue', label='SU2 (PS)')
    if exp_x is not None and exp_mach is not None:
        plt.scatter(exp_x, exp_mach, s=20, color='red', label='Mises Data')
    plt.ylabel(quantity)
    plt.grid(True, linestyle='--', alpha=0.3)
    if mirror_PS:
        plt.xlim(-1, 1)
    else:
        plt.xlim(0, 1)
    plt.legend(loc='upper left')
    plt.savefig(f"non-normalized{quantity}_{string}.svg", bbox_inches='tight')
    plt.show()


def surfaceFlowAnalysis_datablade(current_directory, string, bladeName,
                                  P01, gamma):
    """Generate comparison plots for SU2 and MISES results."""
    su2_file = os.path.join(current_directory,
                            f"surface_flow{string}_{bladeName}.csv")
    df = pd.read_csv(su2_file, sep=',')
    _, _, dataSS, dataPS = SU2_organize(df)

    xSS = dataSS['x'].values
    ySS = dataSS['y'].values
    s_normSS = surface_fraction(xSS, ySS)
    pressureSS = dataSS['Pressure'].values
    machSS = compute_Mx(P01, pressureSS, gamma)

    xPS = dataPS['x'].values
    yPS = dataPS['y'].values
    s_normPS = surface_fraction(xPS, yPS)
    pressurePS = dataPS['Pressure'].values
    machPS = compute_Mx(P01, pressurePS, gamma)

    SU2_DataPlotting(s_normSS, s_normPS, machSS, machPS,
                     "Mach Number", string)
