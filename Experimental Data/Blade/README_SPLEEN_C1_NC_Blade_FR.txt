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
Measurement technique : Fast response transducers
Data structure        : Matrix
Data format           : Excel, MatLab, any recent version
Data processing level : processed


Instrumentation chain :
-----------------------
   - Probe              : B-FR
   - Sensor             : Kulite sensor
   - DAS                : NI6253
   - Sampling frequency : 1.2 MHz
   - Sampling time      : 3 s

Reference system :
------------------

-  z/H    : Normalized spanwise distance from the endwall
-  x/C_ax : Normalized axial location (PS coordinate is negative to distinguish from SS)
-  S/S_l  : Normalized location along the PS or SS surface length (PS coordinate is negative to distinguish from SS)

Reference :
-----------

- Ps/P01 	: time-averaged static pressure measured by the sensor normalized by the freestream total pressure at plane 01 obtained with correlation
- STDP/P01  	: standard deviation of static pressure signal measured by the sensor normalized by the freestream total pressure at plane 01 obtained with correlation
- SKEWP/P01  	: skewness of static pressure signal  measured by the sensor normalized by the freestream total pressure at plane 01 obtained with correlation
- KURTP/P01  	: kurtosis of static pressure signal  measured by the sensor normalized by the freestream total pressure at plane 01 obtained with correlation


Definitions:
------------

- Standard deviation : To get additional explanation on the functionning of measures of standard deviation,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"

- Skewness           : To get additional explanation on the functionning of measures of skewness,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"

- Kurtosis           : To get additional explanation on the functionning of measures of kurtosis,
                       go to section 4.9 of the file "SPLEENC1_MeasurementTechniques"







