[Data description: Data measurements]
=====================================

Project name : Secondary and Leakage Flow Effects in High-Speed Low-Pressure Turbines {SPLEEN}
               
Main authors of the database : Sergio Lavagnoli, Gustavo Lopes, Loris Simonassi

The authors gratefully acknowledge funding of the SPLEEN
project by the Clean Sky 2 Joint Undertaking under the European
Unions Horizon 2020 research and innovation program under the
grant agreement 820883

Date : 26-10-2023
Version : 7

VKI facility name     : Continuous High Speed Cascade Wind Tunnel S-1
Measurement technique : Fast response transducers
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

- k 		: Turbulent kinetic energy
- TI_mean       : Turbulence intensity computed with FRV4HP, using time-averaged local absolute velocity
- ILS           : Integral length scale estimated with FRV4HP

Comments:
---------

- Users of the database are warned about the measurements performed with the FRV4HP in the absence of the WG. The measurements display a behavor that is not physical and therefore should be used with caution. 
The turbulent quantities should be normalized by the midspan value and adimensionalized with the correct value of turbulence intensity measured with a hot-wire probe (~2.40%).
For more details on the scaling of the inlet turbulence profiles, go to Section 5 of the file "SPLEENC1_MeasurementTechniques"

Definitions:
------------

- To get additional explanation on computation of turbulent quantities with FRV4HP, go to Section 4.5 of the file "SPLEENC1_MeasurementTechniques"

