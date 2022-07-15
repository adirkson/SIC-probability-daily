#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:47:36 2022

@author: ard000

This script shows how to calibrate a 33-day SIC forecast 
at a single grid point in the western Hudson Bay
Model data: SEAS5
Observation data: OSI-SAF
The script takes about 9 seconds to run
"""
### temporary for testing ####
import sys
sys.path.append('../ncgr-sic-package')

import ncgr_sic 
from dcnorm import dcnorm_gen
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gfilt
from scipy.interpolate import interp1d

a = 0.0 # minimum possible SIC
b = 1.0 # maximum possible SIC
x = np.linspace(a, b, 101) # the discretrized range for SIC
N_ens = 20 # ensemble size for the model

##############################
####### Time variables #######
##############################
t = 2016 # year of the forecast to be calibrated (feel free to explore other years!!)
# all years
yr_s = 1998 # first year in hindcast record
yr_f = 2017 # last year in hindcast record
years = np.arange(1998, 2017+1, 1)

t_ind = np.where(years==t)[0][0] # index of years array where the forecast year occurs
# training years (leave-one-out cross-validation)
tau_t = years[years != t] # years used for training (all but the forecast year)
tau_t_inds = np.where(years != t)[0] # indices of years array used for training
# lead time
lead_total = 33 # total number of days for the forecast
lead = np.arange(lead_total) # array for each lead time
lead_coarse = lead[::2] # the "coarsened" lead times; calibration is only done explicitly every other day

##############################
######### Load Data ##########
##############################
# current work directory should be ~[branch-name]/code/examples/
path = 'data/'
# 33-days of hindcast sic for all years within 150 km radius of target grid point
SIC_hc_pred_allyears = np.load(path+'SIC_hc_pred.npy') # shape=(years,lead,ensemble,space)
# 33 days of observed sic for all years within a 50 km radius of target grid point
SIC_obs_pint_allyears = np.load(path+'SIC_obs_pint.npy') # shape=(years,lead,space)
# 33 days of observed sic for all years at the target grid point
SIC_obs_target_allyears = np.load(path+'SIC_obs_target.npy') # shape=(years,lead)

# 33 days of DCNORM distribution parameters for the observations 
# for all years at the target grid point; obtained from 
# running `obs_uncertainty_distribution.py`
mu_obs = np.load(path+'mu_obs.npy') # shape=(years,lead)
sigma_obs = np.load(path+'sigma_obs.npy') # shape=(years,lead)

# distance arrays for each grid point in `space` dimension
dist_pred = np.load(path+'distance_pred.npy') # within 150 km of target grid point
dist_pint = np.load(path+'distance_pint.npy') # within 50 km of target grid point

##################################
#### Extract from data arrays ####
#### the necessary variables  ####
#### for performing NCGR-sic  ####
##################################
X_t_pred = SIC_hc_pred_allyears[t_ind] # extract forecast data for year to be calibrated within 150 km
X_pred = SIC_hc_pred_allyears[tau_t_inds] # extract training forecast data within 150 km
Y_pint = SIC_obs_pint_allyears[tau_t_inds] # extract training observations within 50 km
Y_tar = SIC_obs_target_allyears[tau_t_inds] # extract training observations for target grid point

#######################################
#### Actual code for calibrating ######
#######################################

# instantiate the NCGR calibration class
ncgr_sic_gridpoint = ncgr_sic.ncgr_sic_gridpoint(a=0.0, b=1.0) #inputs are min and max SIC possible for input data

# instantiate DCNORM distrbution class
dcnorm = dcnorm_gen(a=a, b=b)

# first define weights for space based on distance
r_pred = 150e3 # the neighborhood radius size for building ensemble forecast predictors
weights_pred = 1 - (dist_pred/r_pred)**2.
weights_pred = weights_pred/np.sum(weights_pred) 

r_pint = 50e3 # the neighborhood radius size for building observation-based pseudo-intercept predictors
weights_pint = 1 - (dist_pint/r_pint)**2.
weights_pint = weights_pint/np.sum(weights_pint) 


##### Section 1: Loop over lead times, build regression equations from training data,
##### and estimate calibration coefficients.
# initialize array to store calibration coefficients for every other forecast day
params_coarse_store = np.zeros((len(lead_coarse), 4))
# initialize array to store predictors for the forecast to be calibrated
predictors_t = np.zeros((lead_total, 2, 2))

# Loop over all lead times
params = None # initialize the calibration coefficients to None
counter_coarse = 0 # index to keep track of the coarsend lead times
for jj in lead:
    # Build predictors from forecasts and observations 
    # for the current lead time jj
    result_predictors = ncgr_sic_gridpoint.build_predictors(X_t_pred[jj], X_pred[:,jj], Y_pint[:,jj],
                                                   tau_t, t,
                                                   weights_pred, weights_pint)
    
    # store the real-time predictors
    predictors_t[jj] = result_predictors.predictors_t

    # If the loop index jj matches a coarsened lead time
    # fit calibration coefficients
    if np.any(lead_coarse==jj):
        # Compute CDF for the observation based on its DCNORM distribution parameters (parameters were computed using `dcnorm.fit_moments`)
        F_obs = np.zeros((len(tau_t), len(x))) # to be filled with the observed CDF for each year over the training period
        for yy in range(len(tau_t)):
            rv_obs = dcnorm(mu_obs[yy,jj], sigma_obs[yy,jj]) # instantiate dcnorm distribution with these parameters
            F_obs[yy] = rv_obs.cdf(x) # compute CDF
                  
        # Estimate calibration coefficients taking observed uncertainty into account
        if params is None:
            # for first forecast day (params=None) don't provide initial guesses for the parameters
            result_params = ncgr_sic_gridpoint.compute_coeffs_obsunc(result_predictors.predictors_tau, 
                                                            F_obs, x)
        else:
            # for remaining forecast days, provide initial guess as previous day's estimate;
            # termininate mimization after 3 iterations by setting maxiter
            result_params = ncgr_sic_gridpoint.compute_coeffs_obsunc(result_predictors.predictors_tau, 
                                                            F_obs, x, 
                                                            params0=result_params.params,
                                                            maxiter=3)
            
        
        # store the calibration coefficients in the prepped array
        params_coarse_store[counter_coarse] = result_params.params
        
        counter_coarse += 1 # update counter
    
####### Section 2: interpolate and smooth the calibration coefficients    
# linearly interpolate the calibration coefficients over the remaining lead times
p_interp = interp1d(lead_coarse, params_coarse_store, fill_value='extrapolate', axis=0)
params_store = p_interp(lead)     

# smooth the parameters using a Gaussianfilter
params_smooth_store = gfilt(params_store, 2, axis=0)  

####### Section 3: Loop over all lead times and apply 
####### regression equations to the forecast to be calibrated
mu_cal = np.zeros(lead_total)
sigma_cal = np.zeros(lead_total)
sip_cal = np.zeros((lead_total,3)) # three sip values for sic=15%, 45%, and 75%
# for visualizing the calibrated forecast, we're going to store distribution percentiles
pcs = np.arange(0.1, 0.9+0.1, 0.1)
percentiles_cal = np.zeros((lead_total, len(pcs)))
# loop over all lead times
for jj in lead:
    # calibrate forecast; note that `sic_thresh` is an optional argument here
    # which determines for which SIC thresholds that SIP is calculated; by default 
    # sic_thresh=np.array([0.15, 0.45, 0.75])
    result_calibrate = ncgr_sic_gridpoint.calibrate(predictors_t[jj], params_smooth_store[jj])
    # unpack the result
    mu_cal[jj], sigma_cal[jj], sip_cal[jj] = result_calibrate
    rv_cal = dcnorm(mu_cal[jj], sigma_cal[jj])
    percentiles_cal[jj] = rv_cal.ppf(pcs)
    

### Plot the result
fig = plt.figure(num=1, figsize=(8,8))
ax = fig.add_subplot(211)
# plot the raw forecast
X_t_target = X_t_pred[:,:,dist_pred==0.0][:,:,0]
ax.plot(lead, np.percentile(X_t_target, pcs*100, axis=-1).T, lw=1.0)
ax.legend(np.around(pcs,2), title='percentiles', ncol=2)
ax.plot(lead, Y_tar[t_ind], lw=2.0, color='k', label='obs')
ax.set_title('Raw Forecast Percentiles')

ax = fig.add_subplot(212)
ax.plot(lead, percentiles_cal, lw=1.0)
ax.plot(lead, Y_tar[t_ind], lw=2.0, color='k', label='obs')
ax.set_title('Calibrated Forecast Percentiles')





