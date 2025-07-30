[Data description: Data measurements]
=====================================

Project name : Secondary and Leakage Flow Effects in High-Speed Low-Pressure Turbines {SPLEEN}
               
Main authors of the database : Sergio Lavagnoli, Gustavo Lopes, Loris Simonassi

The authors gratefully acknowledge funding of the SPLEEN
project by the Clean Sky 2 Joint Undertaking under the European
Unions Horizon 2020 research and innovation program under the
grant agreement 820883

Date : 27-10-2022
Version : 3

VKI facility name     : Continuous High Speed Cascade Wind Tunnel S-1
Measurement technique : Miniature pneumatic 5h probe upstream measurements
Data structure        : Matrix
Data format           : Excel, MatLab, any recent version
Data processing level : processed



Instrumentation chain :
-----------------------
   - Probe              : P-C5HP-01
   - Sensor             : Scanivalve MPS4264 
   - DAS                : Scanivalve MPS4264
   - Sampling frequency : 300 Hz
   - Sampling time      : 3 s

Reference system :
------------------

-  z/H    : Normalized spanwise distance from the endwall
-  y/g    : Normalized pitchwise location
-  x/C_ax : Normalized axial location

Reference :
-----------

- i 		: incidence angle to cascade in deg (definition of angles in "SPLEEN-HSTC-DB-MeasurementTechniques_vX")
- Pitch 	: Pitch angle at inlet of cascade in deg (definition of angles in "SPLEEN-HSTC-DB-MeasurementTechniques_vX")
- rho 		: Local density measured by C5HP in Kg m-3
- V_ax 		: Local axial velocity measured by C5HP in m s-1
- P02/P01	: Local total pressure measured by C5HP at Plane 02 normalized by the freestream total pressure at plane 01 obtained with correlation
- Ps2/P01	: Local static pressure measured by C5HP at Plane 02 normalized by the freestream total pressure at plane 01 obtained with correlation

Comments :
----------
- Computation of i : Incidence [deg] = Flow angle [deg] - Metal angle at the inlet [deg]

- Metal angle at the inlet [deg] = 37.3






