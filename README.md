# SIC-probability-daily
Calibrating daily sea-ice concentration forecasts with NCGR-sic (Non-homogeneous Censored Gaussian Regression for Sea Ice Concentration)

Code structure:

`/code`

Scripts with classes needed to for performing NCGR-sic:
`/code/ncgr-sic-package/`
* ncgr_sic.py
* dcnorm.py

Example script to show how to perform NCGR-sic on a sample 33-day forecast
`/code/example/`
* seas5_0601_HB.py

usind model and observation data in `/code/example/data`

Example script to show how to fit observation and its uncertainty value to
the DCNORM distribution
`/code/example/`
* obs_uncertainty_distribution.py

the outputs of this script are already in `/code/example/data`