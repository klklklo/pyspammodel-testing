# pyspammodel-testing

This repository contains the source code for tests to check the original pyspammodel (https://github.com/klklklo/pyspammodel). 

Original article of the SPAM model: 
Nikolaeva, V.; Gordeev, E. SPAM: Solar Spectrum Prediction for Applications and Modeling. Atmosphere 2023, 14, 226. https://doi.org/10.3390/atmos14020226

The repository contains the following folders:
1. models - contains the source code of the original models and coefficients for them;
2. tests - folder with tests.

Inside the tests folder there are two folders with tests of the SolarSPAM and AeroSPAM models.
- aero_spam_test:
  - octave_data - folder with data calculated by Matlab implementation of AeroSPAM model with input values  65. < F<sub>10.7</sub> < 200. with a step of 1;
  - _aero_coeffs.csv - file with model coefficients;
  - _aero_spam.py - file with test source code.

- solar_spam_test:
  - octave_data - folder with data calculated by Matlab implementation of SolarSPAM model with input values  65. < F<sub>10.7</sub> < 200. with a step of 1;
  - table_A1_data.csv - file with the coefficient table from the original article;
  - table_A1_with_05.csv - updating the coefficient table with the addition of coefficients for λ = 0.5 nm;
  - table_A1_corrected.csv - final corrected version of the coefficient table. Corrected coefficients P<sub>3</sub> at λ = 8.5 nm and λ = 9.5 nm;
  - _solar_spam.py - file with test source code.
   
