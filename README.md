# analysis_temp-asym
Code and data for 'Asymmetry in the Asymmetry in the seasonal cycle of zonal-mean surface air temperature'

Journal article: 'Asymmetry in the Asymmetry in the seasonal cycle of zonal-mean surface air temperature' (2023).

Lettie Roach, Ian Eisenman, Till Wagner and Aaron Donohoe

Accepted at Geophysical Rearch Letters

This repository contains all code and processed data to reproduce the figures in the paper.

- asym_funcs.py - key functions, including the simple model

- obsdata/ - contains processed observtaional data, created from full data by notebook analysis/__process_ERA5_and_JRA55.ipynb)

- simplemodel/ - contains

  - scripts to run the model
  
  - forcingdata/ containining netcdf files created by notebook analysis/_generate_model_forcing.ipynb
  
  - output/ containing all model output, except the zonal variations output shown in Fig. S8. This is too large for Github, but can be easily recreated
  
- analysis/  - contains notebooks to create all the figures. These should work out of the box without running the model, except for plot_deltaT_zonalvariations.ipynb

