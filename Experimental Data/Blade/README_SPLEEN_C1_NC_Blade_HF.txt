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
Measurement technique : Hot-Films on the blade
Data structure        : Matrix
Data format           : Excel, MatLab, any recent version
Data processing level : processed



Instrumentation chain :
-----------------------
   - Probe              : B-HF
   - Sensor             : Hot-films / Anemometer (Dantec Dynamic Streamline Pro CTA)
   - DAS                : NI6253-USB
   - Sampling frequency : 1.2 MHz
   - Sampling time      : 3 s

Reference system :
------------------

-  z/H    : Normalized spanwise distance from the endwall
-  x/Ca_x : Normalized axial location (PS coordinate is negative to distinguish from SS)
-  S/S_l  : Normalized location along the PS or SS surface length (PS coordinate is negative to distinguish from SS)

Comments :  Negative values correspond to position on the pressure side
----------

Reference :
-----------

- E 		: time-averaged Bridge voltage measured by sensor in V
- STDE [V] 	: standard deviation of bridge voltage in V
- SKEWE [V] 	: skewness of bridge voltage in V
- KURTE [V] 	: kurtosis of bridge voltage in V
- QSS [-]	: time-averaged quasi-wall shear stress
- STDQSS [-]	: standard deviation of quasi-wall shear stress in V
- SKEWQSS [-]	: skewness  of quasi-wall shear stress in V
- KURTQSS [-]	: kurtosis  of quasi-wall shear stress in V

Definitions:
------------

- Standard deviation : To get additional explanation on the functionning of measures of standard deviation,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"

- Skewness           : To get additional explanation on the functionning of measures of skewness,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"

- Kurtosis           : To get additional explanation on the functionning of measures of kurtosis,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"