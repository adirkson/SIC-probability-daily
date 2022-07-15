#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:47:36 2022

@author: ard000

This script shows how to calibrate a 33-day SIC forecast 
at a single grid point in the western Hudson Bay
Model data: SEAS5
Observation data: OSI-SAF
"""
### temporary for testing ####
import sys
sys.path.append('/fs/homeu2/eccc/cmd/cmdx/ard000/work/sea-ice/geps-s2s-calibration/code/Github/SIC-probability-daily/code/ncgr-sic-package')

from dcnorm import dcnorm_gen
import numpy as np

a = 0.0 # minimum SIC
b = 1.0 # maximum SIC

##############################
####### Time variables #######
##############################
# all years
yr_s = 1998 # first year in hindcast record
yr_f = 2017 # last year in hindcast record
years = np.arange(1998, 2017+1, 1)
nyears = len(years)

# lead time
lead_total = 33 # total number of days for the forecast
lead = np.arange(lead_total) # array for each lead time

##############################
######### Load Data ##########
##############################
# current work directory should be ~[branch-name]/code/examples/
path = 'data/'
#33 days of observed sic for all years at the target grid point
Y_tar = np.load(path+'SIC_obs_target.npy') # shape=(years,lead)
#33 days of observed sic standard deviations for all years at the target grid point
Y_std = np.load(path+'SIC_obs_std_target.npy') # shape=(years,lead)

#######################################
####### Distribution fitting ##########
#######################################

# instantiate DCNORM distrbution class
dcnorm = dcnorm_gen(a=a, b=b)

mu_obs = np.zeros((nyears,lead_total))
sigma_obs = np.zeros((nyears,lead_total))
for yy in range(nyears):
    for jj in lead:
        mu_obs[yy,jj], sigma_obs[yy,jj] = dcnorm.fit_moments(Y_tar[yy,jj], Y_std[yy,jj]) # estimate mu and sigma
        
# save parameters in /data
np.save(path+'mu_obs', mu_obs)
np.save(path+'sigma_obs', sigma_obs)