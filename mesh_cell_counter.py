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

from __future__ import annotations
import math
import numpy as np


def n_boundary_layers(h1: float, r: float, H: float) -> int:
    """
    Smallest integer N s.t.   h1*(r**N – 1)/(r – 1) ≥ H
    """
    if r <= 1.0:
        raise ValueError("bl_growth must be > 1")
    return math.ceil(
        math.log(1.0 + (r - 1.0) * H / h1, r)
    )


def polygon_area(pts: np.ndarray) -> float:
    """
    Shoelace formula.  pts shape: (N, 2)
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def estimate_cells(
    *,
    nCellAirfoil: int,
    sizeCellFluid: float,
    sizeCellWake: float,
    first_layer_height: float,
    bl_growth: float,
    bl_thickness: float,
    A_core: float,
    WakeXMin: float,
    WakeXMax: float,
    WakeYMin: float,
    WakeYMax: float,
) -> int:
    """Return an *order-of-magnitude* cell count for the final 2-D mesh."""

    # 1) boundary-layer belt ---------------------------------------------
    N_BL = n_boundary_layers(first_layer_height, bl_growth, bl_thickness)
    cells_bl = nCellAirfoil * N_BL

    # 2) areas ------------------------------------------------------------
    A_wake = (WakeXMax - WakeXMin) * (WakeYMax - WakeYMin)
    A_bulk = max(A_core - A_wake, 0.0)

    # 3) Delaunay triangles (≈ equilateral, area = √3/4 * h²) ------------
    tri_area = 0.4330127018922193             # √3 / 4

    cells_bulk  = A_bulk / (tri_area * sizeCellFluid ** 2)
    cells_wake  = A_wake / (tri_area * sizeCellWake  ** 2)

    # 4) total ------------------------------------------------------------
    return int(round(cells_bl + cells_bulk + cells_wake))


# ----------------------------------------------------------------------
# example as self-test
if __name__ == "__main__":
    # fictitious numbers for a quick smoke test
    A_core_demo = 0.25        # m²
    N = estimate_cells(
        nCellAirfoil      = 549,
        sizeCellFluid     = 0.004,
        sizeCellWake      = 0.001,
        first_layer_height= 1.2e-5,
        bl_growth         = 1.17,
        bl_thickness      = 0.003,
        A_core            = A_core_demo,
        WakeXMin          = -0.1,
        WakeXMax          = 1.6,
        WakeYMin          = -0.5,
        WakeYMax          = 0.5,
    )
    print(f"≈ {N:,d} cells")
