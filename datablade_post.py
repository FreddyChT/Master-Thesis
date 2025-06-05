import numpy as np
import matplotlib.pyplot as plt
import logging

"""Post-processing utilities for DataBladeAnalysis.
Each function here is used when visualizing or organizing solver results."""

# ---------------------------------------------------------------------------
#  SU2 and MISES utilities -- used during post-processing
# ---------------------------------------------------------------------------

# 4) SU2 and MISES utilities -- used during post-processing
# ---------------------------------------------------------------------------

def surface_fraction(xvals, yvals):
    """Normalize arc lengths for plotting (post-processing)."""
    dx = np.diff(xvals); dy = np.diff(yvals)
    seg = np.sqrt(dx**2 + dy**2)
    arc = np.cumsum(seg)
    arc = np.insert(arc, 0, 0.0)
    return arc/arc[-1] if arc[-1] != 0 else arc

def roll_array(arr, shift):
    """Roll array so that index `shift` becomes the first (plotting helper)."""
    return np.concatenate([arr[shift:], arr[:shift]])

def SU2_organize(df):
    """Split SU2 surface CSV into leading, trailing, upper and lower surfaces."""
    leading_edge = df.iloc[0:1].copy()
    trailing_edge = df.iloc[1:2].copy()
    geo = df.iloc[2:].copy().reset_index(drop=True)
    x, y = geo['x'].values, geo['y'].values
    dist = np.hypot(np.diff(x), np.diff(y))
    idx_break = np.argmax(dist) + 1
    upper_surface = geo.iloc[:idx_break].copy()
    lower_surface = geo.iloc[idx_break:].copy()
    return leading_edge, trailing_edge, upper_surface, lower_surface

def SU2_extract_plane_data(df, x_plane, pitch, alpha_m, atol=1e-4):
    """Extract data at a constant x-plane from SU2 restart file."""
    columns = ['y', 'Density', 'Pressure', 'Velocity_x', 'Velocity_y', 'Mach']
    mask = np.isclose(df['x'], x_plane, atol=atol)
    if not mask.any():
        print(f"[WARNING] No data found at x = {x_plane} (tol={atol}). Try increasing tolerance.")
        return None
    sub_df = df.loc[mask, columns + ['x']].copy()
    sub_df['y_norm'] = sub_df['y'] / pitch
    sub_df['flow_angle'] = np.atan2(sub_df['Velocity_y'], sub_df['Velocity_x']) * 180 / np.pi - alpha_m
    sub_df = sub_df.sort_values('y_norm').reset_index(drop=True)
    return sub_df

def SU2_DataPlotting(sSSnorm, sPSnorm, dataSS, dataPS, quantity, string,
                     mirror_PS=False, exp_x=None, exp_mach=None):
    """Quick helper to plot surface data and optional experimental values."""
    fig, _ = plt.subplots(figsize=(14, 9))
    plt.plot(sSSnorm, dataSS, marker='o', ms=2, linestyle='-', color='darkblue', label='SU2 (SS)')
    s_ps = -sPSnorm if mirror_PS else sPSnorm
    plt.plot(s_ps, dataPS, marker='o', ms=2, linestyle='-', color='lightblue', label='SU2 (PS)')
    if exp_x is not None and exp_mach is not None:
        plt.scatter(exp_x, exp_mach, s=20, color='red', label='Mises Data')
    plt.ylabel(quantity, size=20)
    plt.grid(alpha=0.3)
    plt.xlim(-1 if mirror_PS else 1)
    plt.legend(loc='upper left', prop={'size':20}, edgecolor='k', fancybox=False)
    plt.tight_layout()
    plt.show()

def MISES_blDataGather(file_path):
    """Parse MISES boundary layer output for post-processing."""
    all_x_values = []; all_y_values = []
    all_delta_star = []; all_theta = []
    all_theta_star = []; all_shape_factor = []; all_mach = []
    x_tmp = []; y_tmp = []
    ds_tmp = []; th_tmp = []; ts_tmp = []; sf_tmp = []; M_tmp = []
    with open(file_path, 'r') as f:
        for _ in range(2):
            next(f, None)
        for line in f:
            tokens = line.split()
            if len(tokens) < 9:
                if x_tmp:
                    all_x_values.append(x_tmp); all_y_values.append(y_tmp)
                    all_delta_star.append(ds_tmp); all_theta.append(th_tmp)
                    all_theta_star.append(ts_tmp); all_shape_factor.append(sf_tmp)
                    all_mach.append(M_tmp)
                x_tmp, y_tmp = [], []
                ds_tmp, th_tmp, ts_tmp, sf_tmp, M_tmp = [], [], [], [], []
                continue
            x_tmp.append(float(tokens[0])); y_tmp.append(float(tokens[1]))
            ds_tmp.append(float(tokens[5])); th_tmp.append(float(tokens[6]))
            ts_tmp.append(float(tokens[7])); sf_tmp.append(float(tokens[8]))
            M_tmp.append(float(tokens[13]))
    if x_tmp:
        all_x_values.append(x_tmp); all_y_values.append(y_tmp)
        all_delta_star.append(ds_tmp); all_theta.append(th_tmp)
        all_theta_star.append(ts_tmp); all_shape_factor.append(sf_tmp)
        all_mach.append(M_tmp)
    return (all_x_values, all_y_values, all_delta_star, all_theta,
            all_theta_star, all_shape_factor, all_mach)

def MISES_fieldDataGather(file_path):
    """Parse MISES field output (density, pressure, velocity, Mach)."""
    all_x=[]; all_y=[]; all_rho=[]; all_p=[]; all_u=[]; all_v=[]; all_q=[]; all_m=[]
    x_tmp=[]; y_tmp=[]; rho_tmp=[]; p_tmp=[]; u_tmp=[]; v_tmp=[]; q_tmp=[]; m_tmp=[]
    with open(file_path, 'r') as f:
        for _ in range(2):
            next(f, None)
        for line in f:
            tokens = line.split()
            if len(tokens) < 8:
                if x_tmp:
                    all_x.append(x_tmp); all_y.append(y_tmp)
                    all_rho.append(rho_tmp); all_p.append(p_tmp)
                    all_u.append(u_tmp); all_v.append(v_tmp)
                    all_q.append(q_tmp); all_m.append(m_tmp)
                x_tmp, y_tmp = [], []
                rho_tmp, p_tmp = [], []
                u_tmp, v_tmp = [], []
                q_tmp, m_tmp = [], []
                continue
            x_tmp.append(float(tokens[0])); y_tmp.append(float(tokens[1]))
            rho_tmp.append(float(tokens[2])); p_tmp.append(float(tokens[3]))
            u_tmp.append(float(tokens[4])); v_tmp.append(float(tokens[5]))
            q_tmp.append(float(tokens[6])); m_tmp.append(float(tokens[7]))
    if x_tmp:
        all_x.append(x_tmp); all_y.append(y_tmp)
        all_rho.append(rho_tmp); all_p.append(p_tmp)
        all_u.append(u_tmp); all_v.append(v_tmp)
        all_q.append(q_tmp); all_m.append(m_tmp)
    return all_x, all_y, all_rho, all_p, all_u, all_v, all_q, all_m

def MISES_DataGather(data, xNorm, y, n):
    """Organize MISES surface data for comparison plots."""
    index_closest_to_zero = np.abs(data - max(data)).argmin()
    xSS = xNorm[index_closest_to_zero:]
    xSS = np.concatenate((xSS, xNorm[:n*3]))
    ySS = y[index_closest_to_zero:]
    ySS = np.concatenate((ySS, y[:n*3]))
    dataSS = data[index_closest_to_zero:]
    dataSS = np.concatenate((dataSS, data[:n*3]))
    dataSS = savgol_filter(dataSS, window_length=15, polyorder=3)
    xPS = xNorm[index_closest_to_zero:n*7-3:-1]
    xPS = np.concatenate((xPS, xNorm[n*7-3:n*4-2:-1]))
    yPS = y[index_closest_to_zero:n*7-3:-1]
    yPS = np.concatenate((yPS, y[n*7-3:n*4-2:-1]))
    dataPS = data[index_closest_to_zero:n*7-3:-1]
    dataPS = np.concatenate((dataPS, data[n*7-3:n*4-2:-1]))
    dataPS = savgol_filter(dataPS, window_length=15, polyorder=3)
    dxSS = np.diff(xSS); dySS = np.diff(ySS)
    segment_lengthsSS = np.sqrt(dxSS**2 + dySS**2)
    lengths_cumulativeSS = np.cumsum(segment_lengthsSS)
    xSSnorm = lengths_cumulativeSS/lengths_cumulativeSS[-1]
    dxPS = np.diff(xPS); dyPS = np.diff(yPS)
    segment_lengthsPS = np.sqrt(dxPS**2 + dyPS**2)
    lengths_cumulativePS = np.cumsum(segment_lengthsPS)
    xPSnorm = lengths_cumulativePS/lengths_cumulativePS[-1]
    dataSSTrial = data[index_closest_to_zero:]
    dataSSTrial = np.concatenate((dataSSTrial, data[:n*3]))
    dataSSTE = dataSSTrial[-1]
    return xSSnorm, xPSnorm, dataSS, dataPS, dataSSTE
