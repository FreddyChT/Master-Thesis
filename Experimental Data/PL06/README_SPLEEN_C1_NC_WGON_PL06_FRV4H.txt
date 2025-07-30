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
   - Probe              : P-FR-4H-01 
   - Sensor             : Kulite sensor
   - DAS                : NI6253
   - Sampling frequency : 1.2 MHz
   - Sampling time      : 3 s

Reference system :
------------------

-  z/H    : Normalized spanwise distance from the endwall
-  y/g    : Normalized pitchwise location
-  x/C_ax : Normalized axial location

Reference :
-----------

- d 		: outlet flow deviation in deg (definition of angles in "SPLEEN-HSTC-DB-MeasurementTechniques_vX")
- Pitch 	: Pitch angle at inlet of cascade in deg (definition of angles in "SPLEEN-HSTC-DB-MeasurementTechniques_vX")
- rho 		: Local density measured by L5HP in Kg m-3
- V_ax 		: Local axial velocity measured by L5HP in m s-1
- P06/P01	: Local total pressure measured by L5HP at Plane 06 normalized by the freestream total pressure at plane 01 obtained with correlation
- Ps6/P01	: Local static pressure measured by L5HP at Plane 06 normalized by the freestream total pressure at plane 01 obtained with correlation

Comments :
----------
- Computation of d : Deviation[deg] = Flow angle [deg] - Metal angle at the outlet [deg]

- Metal angle at the outlet [deg] = 53.8






