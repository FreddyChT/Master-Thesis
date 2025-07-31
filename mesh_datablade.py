import numpy as np
from utils import *

'''
bladeFilePath, run_dir, string, bladeName,                          # File management                
dist_inlet, dist_outlet, x_plane                                    # Inlet / outlet definition
axial_chord, pitch, d_factor,                                       # Geometric parameters          
sizeCellAirfoil, sizeCellFluid,                                     # Grid element sizes
nCellAirfoil, nCellPerimeter,                                       # Grid no. elements
nBoundaryPoints,                                                    # Smoothness of airfoil-like top & bottom boundary lines
first_layer_height, bl_growth, bl_thickness,                        # BL definition
size_LE, dist_LE, size_TE, dist_TE,                                 # LE & TE refinement
VolWAkeIn, VolWAkeOut, WakeXMin, WakeXMax                           # Wake / airfoil refinement
'''

def mesh_datablade():
    # --------------------------- GEOMETRY EXTRACTION ---------------------------
    # read_spleen_airfoil returns: PS then SS.
    out = process_airfoil_file(bladeFilePath, n_points=1000, n_te=60, d_factor=0.5)
    xSS, ySS, _, _ = out['ss']
    xPS, yPS, _, _ = out['ps']
    
    # --------------------------- BOUNDARY POINTS ---------------------------
    L1x = dist_inlet * axial_chord
    #L1 = L1x / abs(np.cos(alpha1 * np.pi/180))
    #L1y = L1 * abs(np.sin(alpha1 * np.pi/180))
    L2x = (dist_outlet + 1) * axial_chord                     # distance from leading edge is 1 axial chord
    #L2 = L2x / abs(np.cos(alpha2 * np.pi/180))
    #L2y = L2 * abs(np.sin(alpha2 * np.pi/180))
    
    m1 = np.tan(alpha1*np.pi/180)
    m2 = np.tan(alpha2*np.pi/180)
    
    geo_file = run_dir / f"cascade2D_{string}_{bladeName}.geo"
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

        # ---- Partition airfoil into 4 regions (LE, TE, upper mid, lower mid) ----
        le_frac = 0.015
        te_frac = 0.015
        
        n_ss = len(xSS)
        n_le = max(1, int(n_ss * le_frac))
        n_te = max(1, int(n_ss * te_frac))
        te_start = n_ss - n_te
        
        def write_bspline(cid, ids):
            f.write(f"BSpline({cid}) = {{{', '.join(map(str, ids))}}};\n")

        # suction side segments
        ids_ss_le = list(range(0, n_le + 1))
        ids_ss_mid = list(range(n_le, te_start + 1))
        ids_ss_te = list(range(te_start, n_ss))

        write_bspline(1100, ids_ss_le)
        write_bspline(1101, ids_ss_mid)
        write_bspline(1102, ids_ss_te)

        # pressure side uses same LE/TE points as suction side
        all_ps = [0] + bottomPts[1:-1] + [TE_ID]
        ids_ps_le = all_ps[: n_le + 1]
        ids_ps_mid = all_ps[n_le : te_start + 1]
        ids_ps_te = all_ps[te_start:]

        write_bspline(2100, ids_ps_le)
        write_bspline(2101, ids_ps_mid)
        write_bspline(2102, ids_ps_te)
        
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
        
        # ------------------ OUTER PERIMETER (node‑to‑node periodic) ------------------
        xMean = (np.array(xSS) + np.array(xPS)) / 2
        yMean = (np.array(ySS) + np.array(yPS)) / 2
        
        f.write("\n// Bottom boundary polyline \n")
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
        bottom_ids.append('15001')
        f.write(f"Line(150) = {{{', '.join(map(str, bottom_ids))}}};\n")
        
        f.write("\n// Top boundary polyline (translate bottom_ids by +pitch) \n")
        top_ids = [15003]
        for ii, idx in enumerate(bottom_idxs):
            tpid = 15100 + ii + 100
            xt, yt = xMean[idx], yMean[idx] + pitch/2
            f.write(f"Point({tpid}) = {{{xt:.16e}, {yt:.16e}, 0, k}};\n")
            top_ids.append(tpid)
        top_ids.append('15002')
        f.write(f"Line(152) = {{{', '.join(map(str, top_ids))}}};\n")
        
        f.write("\n// Single inlet/outlet lines \n")
        f.write("Line(153) = {15000, 15003};   // inlet\n")
        f.write("Line(151) = {15001, 15002};   // outlet\n")
        
        # ------------------ CURVE LOOPS ------------------
        f.write("\n// Curve Loop airfoil & boundary edge\n")
        f.write("Curve Loop(10) = {1100, 1101, 1102, -2102, -2101, -2100};\n")
        f.write("Curve Loop(50) = {150, 151, -152, -153};\n\n")
        
        # ------------------ PLANE SURFACES ------------------
        # Now define plane surfaces from the curve loops.
        f.write("Plane Surface(5) = {50, 10}; \n") # Fluid subdomain
    
        # ------------------ TRANSFINITE MESH DEFINITIONS ------------------
        f.write("\n// Transfinite definitions for connector lines\n")
        # Airfoil and Boundary layer curves
        f.write(f"Transfinite Curve {{1100}} = {nCellAirfoil*(4*le_frac)} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{1101}} = {nCellAirfoil*(1-le_frac-te_frac)} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{1102}} = {nCellAirfoil*(4*te_frac)} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{2100}} = {nCellAirfoil*(4*le_frac)} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{2101}} = {nCellAirfoil*(1-le_frac-te_frac)} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{2102}} = {nCellAirfoil*(4*te_frac)} Using Progression 1; \n")

        # Airfoil and Mesh boundary curves
        f.write(f"Transfinite Curve {{10}} = {nCellPerimeter} Using Progression 1; \n")
        f.write(f"Transfinite Curve {{50}} = {nCellPerimeter} Using Progression 1; \n")
    
        # --------------------------------------------------------------------- #
        #  NEW 1  ─ Boundary‑Layer field (curved, orthogonal grid lines)        #
        # --------------------------------------------------------------------- #
        f.write("\n// --- BOUNDARY‑LAYER FIELD (curved normals) ---------------\n")
        f.write("Field[1] = BoundaryLayer;\n")
        f.write("Field[1].EdgesList   = {1100, 1101, 1102, 2102, 2101, 2100};   // SS & PS splines\n")
        f.write(f"Field[1].hwall_n     = {first_layer_height};\n")
        f.write(f"Field[1].ratio       = {bl_growth};\n")
        f.write(f"Field[1].thickness   = {bl_thickness};\n")
        f.write(f"Field[1].hfar        = {sizeCellFluid};\n")
        f.write(f"Field[1].Quads       = 1;          // keep quads after recombine\n")
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
        f.write(f"Field[7].YMin  = { -max(abs(y15001), abs(y15002)) };\n")
        f.write(f"Field[7].YMax  = { max(abs(y15001), abs(y15002)) };\n")
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
        f.write('Physical Curve("blade1", 18005) = {1100, 1101, 1102, 2102, 2101, 2100};\n')
        f.write('Physical Surface("fluid", 18008) = {5};\n') 
        
        
    print(f"Geo file written at: {geo_file}")
    
    # Run gmsh to generate the SU2 mesh.
    print("STARTING mesh generation...")
    import gmsh
    
    gmsh.initialize()
    gmsh.open(str(geo_file))
    gmsh.model.mesh.generate(2)
    gmsh.write(str(run_dir / f"cascade2D_{string}_{bladeName}.msh"))
    gmsh.write(str(run_dir / f"cascade2D_{string}_{bladeName}.su2"))
    gmsh.finalize()
      