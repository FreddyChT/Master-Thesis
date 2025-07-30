[Data description: Data measurements]
=====================================

Project name : Secondary and Leakage Flow Effects in High-Speed Low-Pressure Turbines {SPLEEN}
               
Main authors of the database : Sergio Lavagnoli, Gustavo Lopes, Loris Simonassi

The authors gratefully acknowledge funding of the SPLEEN
project by the Clean Sky 2 Joint Undertaking under the European
Unions Horizon 2020 research and innovation program under the
grant agreement 820883

Date : 25-03-2024
Version : 4

VKI facility name     : Continuous High Speed Cascade Wind Tunnel S-1
Measurement technique : Single hot wire measurement
Data structure        : Matrix
Data format           : Excel, MatLab, any recent version
Data processing level : processed


Instrumentation chain :
-----------------------
   - Probe              : XW
   - Sensor             : Anemometer ( Dantec Dynamic Streamline Pro CTA)
   - DAS                : NI6253-USB
   - Sampling frequency : 40 kHz
   - Sampling time      : 2 s

Reference system :
------------------

-  z/H    : Normalized spanwise distance from the endwall
-  y/g    : Normalized pitchwise location
-  x/C_ax : Normalized axial location

Reference :
-----------

- TI		: Turbulence intensity accounting for streamwise and crosswise components in %
- TI_ISO	: Turbulence intensity computed with isotropic assumption in %
- U_mean	: mean local velocity measured from 5HP at the testing location
- ILS_1		: Integral length scale, from spectrum method, 20-100 Hz avg, in mm
- ILS_2		: Integral length scale, from spectrum method, 100-400 Hz avg, in mm

Comments : The assumption that the fluctuations of total temperature are negligible is made.
---------- So the voltage fluctuations depends only to the sensitivity relative to the velocity and density.
           The sensitivity method is based on the works of Cukurel et al. and Boufidi and Fontaneto.

	     The post-processing methodology is detailed in the work of Pastorino et al. (2024):
		"MEASUREMENTS OF TURBULENCE IN COMPRESSIBLE LOW-DENSITY FLOWS AT THE INLET OF A TRANSONIC LINEAR CASCADE WITH
 		AND WITHOUT UNSTEADY WAKES"

           Additional informations can be found in the word file "SPLEENC1_MeasurementTechniques".



