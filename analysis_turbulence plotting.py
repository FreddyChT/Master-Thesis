# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 21:55:44 2025

@author: fredd
"""

# Matplotlib template to reproduce the look of the provided figure
# - Many curves colored by incidence angle with a top colorbar
# - A highlighted (maroon) reference curve
# - White scatter markers with black edges ("experimental" points)
# - Two dashed rectangular boxes labeled SS and PS
# - Optional right-side inset zooms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# Synthetic data (replace with yours)
# -----------------------------
np.random.seed(2)
x = np.linspace(0.0, 1.0, 400)
angles = np.linspace(-2.0, 2.0, 60)  # incidence angles [deg]

# Two families of curves, loosely shaped like the figure
# "SS" (upper) and "PS" (lower)
def family_upper(x, inc):
    base = 0.72 + 0.22 * (1 - np.exp(-12 * x))
    tweak = 0.015 * inc * (1 - 1.6 * x)  # slight angle dependence
    tail  = -0.05 * np.exp(-2.2 * (1 - x))
    return base + tweak + tail

def family_lower(x, inc):
    base = 0.24 + 0.11 * np.exp(-70 * x) + 0.52 * x**1.2
    tweak = 0.02 * inc * (x - 0.5)
    return base + tweak

# Choose one angle to highlight as the "reference" solution
inc_ref = 1.2

# Experimental-like points (replace with real data)
x_pts_u = np.linspace(0.08, 0.95, 18)
y_pts_u = family_upper(x_pts_u, 0.1) + np.random.normal(0, 0.008, x_pts_u.size)

x_pts_l = np.linspace(0.06, 0.95, 16)
y_pts_l = family_lower(x_pts_l, 0.0) + np.random.normal(0, 0.007, x_pts_l.size)

# -----------------------------
# Plotting
# -----------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
})

fig = plt.figure(figsize=(8.2, 5.6))
# GridSpec to leave room for the top colorbar
from matplotlib.gridspec import GridSpec
G = GridSpec(nrows=2, ncols=2, height_ratios=[0.15, 0.85], width_ratios=[1.0, 0.42], wspace=0.25, hspace=0.0)

# Main axis spans left column, bottom row
ax = fig.add_subplot(G[1,0])

# Colormap and normalization for angles
cmap = cm.get_cmap("turbo")  # 'viridis' also works well
norm = Normalize(vmin=angles.min(), vmax=angles.max())

# Plot the colored families
for a in angles:
    ax.plot(x, family_upper(x, a), color=cmap(norm(a)), lw=2.0, alpha=1.0)
    ax.plot(x, family_lower(x, a), color=cmap(norm(a)), lw=2.0, alpha=1.0)

# Highlight curve
ax.plot(x, family_upper(x, inc_ref), color="#6b0000", lw=3.0, zorder=3)
ax.plot(x, family_lower(x, inc_ref), color="#6b0000", lw=3.0, zorder=3)

# Scatter points (white fill with black edge)
ax.scatter(x_pts_u, y_pts_u, facecolor="white", edgecolor="black", s=50, zorder=4)
ax.scatter(x_pts_l, y_pts_l, facecolor="white", edgecolor="black", s=50, zorder=4)

# Axes limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0.0, 1.02)
ax.set_xlabel(r"$s/S_L\;[-]$")
ax.set_ylabel(r"$M_{is}\;[-]$")

# Dashed focus rectangles and labels
box1 = Rectangle((0.13, 0.70), 0.42, 0.20, fill=False, lw=2.0,
                 linestyle=(0, (7, 7)), color="black")
box2 = Rectangle((0.10, 0.20), 0.28, 0.20, fill=False, lw=2.0,
                 linestyle=(0, (7, 7)), color="black")
ax.add_patch(box1)
ax.add_patch(box2)
ax.text(0.58, 0.73, "SS", transform=ax.transAxes, fontsize=18)
ax.text(0.58, 0.29, "PS", transform=ax.transAxes, fontsize=18)

# Ticks inside and mirrored like the reference figure
ax.tick_params(which="both", top=True, right=True, length=6)

# -----------------------------
# Top colorbar axis
# -----------------------------
cb_ax = fig.add_subplot(G[0,0])
cb = cm.ScalarMappable(norm=norm, cmap=cmap)
# Use an "image" to draw a horizontal colorbar look
cb_img = np.linspace(angles.min(), angles.max(), 256)[None, :]
cb_ax.imshow(cb_img, aspect="auto", extent=[angles.min(), angles.max(), 0, 1], origin="lower", cmap=cmap)
cb_ax.set_yticks([])
cb_ax.set_xlim(angles.min(), angles.max())
cb_ax.set_xlabel(r"$inc\;[\degree]$", labelpad=6)
cb_ax.xaxis.set_ticks([-2, -1, 0, 1, 2])
for spine in cb_ax.spines.values():
    spine.set_visible(False)

# -----------------------------
# Optional right-hand inset zooms
# -----------------------------
ax_ss = fig.add_subplot(G[1,1])
ax_ps = ax_ss.inset_axes([0, -0.50, 1, 0.48])  # a second axis below the first

# SS inset region
xmask1 = (x >= 0.10) & (x <= 0.45)
for a in angles:
    ax_ss.plot(x[xmask1], family_upper(x, a)[xmask1], color=cmap(norm(a)), lw=2)
ax_ss.plot(x[xmask1], family_upper(x, inc_ref)[xmask1], color="#6b0000", lw=3)
ax_ss.scatter(x_pts_u, y_pts_u, facecolor="white", edgecolor="black", s=40)
ax_ss.set_xlim(0.10, 0.45)
ax_ss.set_ylim(0.70, 0.90)
ax_ss.set_yticks([0.7, 0.8, 0.9])
ax_ss.set_xticks([0.1, 0.2, 0.3, 0.4])
ax_ss.text(0.94, 0.92, "SS", transform=ax_ss.transAxes, ha="right")
ax_ss.tick_params(direction="in", top=True, right=True, length=5)

# PS inset region
xmask2 = (x >= 0.10) & (x <= 0.40)
for a in angles:
    ax_ps.plot(x[xmask2], family_lower(x, a)[xmask2], color=cmap(norm(a)), lw=2)
ax_ps.plot(x[xmask2], family_lower(x, inc_ref)[xmask2], color="#6b0000", lw=3)
ax_ps.scatter(x_pts_l, y_pts_l, facecolor="white", edgecolor="black", s=40)
ax_ps.set_xlim(0.10, 0.40)
ax_ps.set_ylim(0.20, 0.40)
ax_ps.set_yticks([0.2, 0.3, 0.4])
ax_ps.text(0.94, 0.90, "PS", transform=ax_ps.transAxes, ha="right")
ax_ps.tick_params(direction="in", top=True, right=True, length=5)

for a in [ax_ss, ax_ps]:
    for spine in a.spines.values():
        spine.set_linewidth(1.4)

fig.tight_layout()
plt.show()
