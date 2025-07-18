# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:03:36 2025

@author: fredd
"""

"""
mesh_cell_counter.py  –  quick estimator of the final element count

USAGE
-----
>>> from mesh_cell_counter import estimate_cells, polygon_area
>>> A_core = polygon_area(outer_boundary_pts) - polygon_area(airfoil_pts)
>>> N = estimate_cells(
...     nCellAirfoil      = 549,
...     sizeCellFluid     = 0.004,      # m
...     sizeCellWake      = 0.001,      # VolWAkeIn
...     first_layer_height= 1.2e-5,     # m
...     bl_growth         = 1.17,
...     bl_thickness      = 0.003,      # m
...     A_core            = A_core,     # m²
...     WakeXMin          = -0.1,       # m
...     WakeXMax          =  1.6,       # m
...     WakeYMin          = -0.5,       # m
...     WakeYMax          =  0.5        # m
... )
>>> print(f"estimated cells: {N:,.0f}")
"""

import math
import numpy as np

_TRI_AREA = 0.4330127018922193          # √3 / 4


def n_boundary_layers(h1: float, r: float, H: float) -> int:
    if r <= 1.0:
        raise ValueError("bl_growth must be > 1.0")
    return math.ceil(math.log(1 + (r - 1) * H / h1, r))


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1)))


def estimate_cells(
    *,
    nCellAirfoil: int,
    sizeCellFluid: float,
    sizeCellWake: float,
    first_layer_height: float,
    bl_growth: float,
    bl_thickness: float,
    A_core: float,                     # outer-boundary area minus blade area
    WakeXMin: float,
    WakeXMax: float,
    WakeYMin: float,
    WakeYMax: float,
    TRI_AREA: float = 0.48,            # tweak if you have reference meshes
) -> int:
    # 1) boundary-layer belt
    N_BL = n_boundary_layers(first_layer_height, bl_growth, bl_thickness)
    cells_bl = nCellAirfoil * N_BL

    # 2) areas
    A_wake = (WakeXMax - WakeXMin) * (WakeYMax - WakeYMin)
    if A_core < A_wake:
        print(f"[WARNING] A_core ({A_core:.2f}) < A_wake ({A_wake:.2f}); "
              "treating bulk area as zero.")
    A_bulk = max(A_core - A_wake, 0.0)

    # 3) triangles
    cells_bulk = A_bulk / (TRI_AREA * sizeCellFluid**2)

    f_wake = 0.5 * (1 + sizeCellWake / sizeCellFluid)   # linear size blending
    cells_wake = f_wake * A_wake / (TRI_AREA * sizeCellWake**2)

    return int(round(cells_bl + cells_bulk + cells_wake))


axial_chord = 1.0017255564576113
sizeCellFluid     = 0.04 * axial_chord

# ----------------------------------------------------------------------
# example as self-test
if __name__ == "__main__":
    # fictitious numbers for a quick smoke test
    A_core_demo = 18.6            # m²  (make sure this is really ≥ A_wake!)
    N = estimate_cells(
        nCellAirfoil      = 549,
        sizeCellFluid     = 0.04 * axial_chord,
        sizeCellWake      = 0.35 * sizeCellFluid,
        first_layer_height= 2.3100789650226696e-05,
        bl_growth         = 1.2532596124885966,
        bl_thickness      = 0.02567715897914744,
        A_core            = A_core_demo,
        WakeXMin          = 0.1 * axial_chord ,
        WakeXMax          = 2.50431389 - 0.5 * axial_chord,
        WakeYMin          = -2.48207475,
        WakeYMax          =  2.48207475,
    )
    print(f"≈ {N:,d} cells")      # → 88 600  (-2 % vs the real 90 540)
