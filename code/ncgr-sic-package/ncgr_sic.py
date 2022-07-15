#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:55:31 2020

@author: arlan
"""
from dcnorm import dcnorm_gen
#from NCGR_sic.dcnorm import dcnorm_gen

import numpy as np
from scipy.stats import norm
from scipy.stats.mstats import hdquantiles
from scipy.optimize import minimize
import sys
from collections import namedtuple
from scipy.stats import t as t_dist



# Function for progress bar
def update_progress(progress, barLength=20):
    ''' (found on forum)
    Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%.
    
    Parameters
    ----------
    progress : float
        A float between 0 and 1. Any int will be converted to a float.
    
    barLength: int (optional)
        The length of the status bar; default of 20 units.

    Returns
    -------
        None
    '''

    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent Completed: [{0}] {1}% {2}".format( "#"*block + " "*(barLength-block), np.around(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def linregress(x, y, axis):
    '''
    A generalization of `scipy.stats.linregress` that works on arrays
    with multiple dimensions.
    
    Parameters
    ----------
    x : ndarray
        The independent variable that y is regressed onto. Same shape as y.
    y : ndarray
        The dependent variable that is regressed on x. Same shape as x.
    axis : int
        The axis along which to regress y on x.

    Returns
    -------
    result: LinregressResult
        An object with the following attributes:
        
            slope : ndarray, shape=(x.shape)
                Slope of the regression line
            int : ndarray, shape=(x.shape)
                Intercept of the regression line
            rval : ndarray, shape=(x.shape)
                Pearson correlation coefficient.
            pval : ndarray, shape=(x.shape)
                The p-value for a hypothesis test whose null hypothesis
                is that the slope is zero, using Wald Test with t-distribution 
                of the test statistic. The test is two-sided.

    '''
    LinregressResult = namedtuple('LinregressResult', ('slope', 'int',
                                               'rval', 'pval'))
    TINY = 1e-20
    df = x.shape[axis]-2
    
    x_a = x - np.ma.mean(x,axis=axis)
    y_a = y - np.ma.mean(y,axis=axis)
    
    var_x = np.ma.sum(x_a**2., axis=axis)
    var_y = np.ma.sum(y_a**2., axis=axis)
    cov_xy = np.ma.sum(x_a*y_a,axis=axis)
    r_xy = cov_xy/np.sqrt(var_x*var_y)
    r_xy[np.sqrt(var_x*var_y)==0.0] = 0.0

    beta = cov_xy/var_x
    beta[var_x==0.0] = 0.0
    
    alpha = np.ma.mean(y, axis=axis) - beta*np.ma.mean(x, axis=axis)
    tval = r_xy * np.sqrt(df / ((1.0 - r_xy + TINY)*(1.0 + r_xy + TINY)))
        
    pval = 2*t_dist.sf(np.ma.abs(tval), df)
    
    if np.ma.is_masked(y):
        beta = np.ma.array(beta, mask=y[0].mask)
        alpha = np.ma.array(alpha, mask=y[0].mask)
        r_xy = np.ma.array(r_xy, mask=y[0].mask)
        pval = np.ma.array(pval, mask=y[0].mask)
    
    return LinregressResult(beta, alpha, r_xy, pval)

# fixed parameters
eps_sigma = 1e-6 # to constrain \sigma>=eps_sigma during minimization
eps_mu = 0.05 # to constrain a-eps_mu<=\mu<=b+eps_mu during minimization
   
class ncgr_sic_gridpoint:
    def __init__(self, a=0.0, b=1.0):
        '''
        A class for performing NCGR-sic.

        Parameters
        ----------
        a : float, optional
            The minimum value for SIC. The default is 0.0.
        b : float, optional
            The maximum value for SIC. The default is 1.0.

        Methods:
            get_model (creates regression predictors from training forecast ensemble and observations)
            compute_coeffs (computes the calibration coefficients)
            calibrate (applies the regression equations to the ensemble forecast)
        

        '''
        self.a = a
        self.b = b
        self.dcnorm = dcnorm_gen(a=a, b=b)

    def build_predictors(self, X_t_pred, X_pred, Y_pint, 
                  tau_t, t, 
                  weights_pred, weights_pint,
                  pval_crit=0.05):
        '''
        Builds the training predictors over training years `tau_t`
        as well as the predictors for the forecast to be calibrated in year `t`.

        description of dimensions used when defining the input `Parameters`:
            N_ens: number of ensemble members
            
            N_gp: number of grid points; can be 1 if no neighboring grid points 
            are used to formulate the ensemble predictors.
            
            N_taut: number of years used for training

        Parameters
        ----------
        X_t_pred : ndarray, shape=(N_ens, N_gp)
            The ensemble forecast(s) used to compute the predictors for the forecast to be calibrated.
        X_pred : ndarray, shape=(N_taut, N_ens, N_gp)
            The ensemble forecasts used to compute the set of predictors for the forecasts used for training.
        Y_pint : ndarray, shape=(N_taut, N_gp)
            The observations used to compute the pseudo-intercept predictors in the regression equations.
        tau_t : ndarray, shape=(N_taut,)
            The years used for training (corresponding to the arguments `X_pred`, `Y_pint`, and `Y_tar`)
        t : float
            The year of the forecast to be calibrated.
        weights_pred : ndarray, shape=(N_gp,)
            A weight applied to each of the pseudo-intercept predictors at the corresponding grid point. Weights must sum to 1.
        weights_pint : ndarray, shape=(N_gp,)
            A weight applied to each of the ensemble-based predictors at the corresponding grid point. Weights must sum to 1.    
        pval_crit : float, optional
            The critical p-value for determining whether trends are statistically significant.
            Default is 0.05.

        Returns
        -------
        result: predictors_result
            An object with the following attibutes:
                predictors_tau : ndarray, shape=(2,2,N_taut)
                    The four regression predictors for each
                    year in the training period. The values in 
                    `predictors_tau[0]` are for the equation for :math:`\mu`. 
                    The values in `predictors_tau[1]` are for
                    the equation for :math:`\sigma`. The first predictor is
                    the pseudo-intercept, the second predictor is the ensemble-based 
                    statistic.
                    
                predictors_t : ndarray, shape=(2,2)
                    The four regression predictors for forecast
                    to be calibrated. The values in 
                    `predictors_tau[0]` are for the equation for :math:`\mu`. 
                    The values in `predictors_tau[1]` are for
                    the equation for :math:`\sigma`. The first predictor is
                    the pseudo-intercept, the second predictor is the ensemble-based 
                    statistic.
                

        '''
        
        N_tau = len(tau_t)
        
        # Perform linear regression of X_pred and Y_pint on tau_t to determine if trends are 
        # statistically significant.
        YEARS_curr_pred = tau_t[:,np.newaxis] * np.ones(X_pred.mean(axis=1).shape)
        YEARS_curr_pint = tau_t[:,np.newaxis] * np.ones(Y_pint.shape)
        
        result_x_pred = linregress(YEARS_curr_pred,X_pred.mean(axis=1),axis=0)
        result_y_pint = linregress(YEARS_curr_pint,Y_pint,axis=0)        
        
        slope_y_pint, int_y_pint, pval_y_pint = result_y_pint.slope, result_y_pint.int, result_y_pint.pval
        slope_x_pred, int_x_pred, pval_x_pred = result_x_pred.slope, result_x_pred.int, result_x_pred.pval
        
        y_coeffs_pint = np.array([slope_y_pint,int_y_pint])
        x_coeffs_pred = np.array([slope_x_pred,int_x_pred])
        
        ########################################
        ######### mu predictors ###############
        ########################################
          
        ######## mu clim predictor ################
        # mu climatology for the obs (this is a weighted trend)
        mu_clim_tau = np.zeros((N_tau,len(weights_pint)))
        mu_clim_tau[:,pval_y_pint<=pval_crit] = (y_coeffs_pint[0][pval_y_pint<=pval_crit,np.newaxis]*tau_t + y_coeffs_pint[1][pval_y_pint<=pval_crit][:,np.newaxis]).T
        mu_clim_tau[:,pval_y_pint>pval_crit] = Y_pint.mean(axis=0)[pval_y_pint>pval_crit]    
        mu_clim_tau = np.sum(weights_pint[np.newaxis,:]*mu_clim_tau,axis=-1)
        
        mu_clim_t = np.zeros((len(weights_pint)))
        mu_clim_t[pval_y_pint<=pval_crit] = (y_coeffs_pint[0][pval_y_pint<=pval_crit]*t + y_coeffs_pint[1][pval_y_pint<=pval_crit]).T
        mu_clim_t[pval_y_pint>pval_crit] = Y_pint.mean(axis=0)[pval_y_pint>pval_crit]    
        mu_clim_t = np.sum(weights_pint*mu_clim_t)    
        
        # mu_clim += TINY
        
        ##### X_d predictor #################
        # de-trended hindcast ensemble mean
        X_d_tau = np.zeros(X_pred.mean(axis=1).shape)
        X_d_tau[:,pval_x_pred<pval_crit] = X_pred.mean(axis=1)[:,pval_x_pred<pval_crit] - (x_coeffs_pred[0][pval_x_pred<pval_crit,np.newaxis]*tau_t + x_coeffs_pred[1][pval_x_pred<pval_crit][:,np.newaxis]).T
        X_d_tau[:,pval_x_pred>=pval_crit] = X_pred.mean(axis=1)[:,pval_x_pred>=pval_crit] - X_pred.mean(axis=(0,1))[pval_x_pred>=pval_crit]    
        X_d_tau = np.sum(weights_pred[np.newaxis,:]*X_d_tau,axis=-1)
        X_d_tau[mu_clim_tau+X_d_tau<self.a] = self.a - mu_clim_tau[mu_clim_tau+X_d_tau<self.a]
        X_d_tau[mu_clim_tau+X_d_tau>self.b] = self.b - mu_clim_tau[mu_clim_tau+X_d_tau>self.b]
        
        X_d_t = np.zeros(X_t_pred.mean(axis=0).shape)
        X_d_t[pval_x_pred<pval_crit] = X_t_pred.mean(axis=0)[pval_x_pred<pval_crit] - (x_coeffs_pred[0][pval_x_pred<pval_crit]*t + x_coeffs_pred[1][pval_x_pred<pval_crit]).T
        X_d_t[pval_x_pred>=pval_crit] = X_t_pred.mean(axis=0)[pval_x_pred>=pval_crit] - X_pred.mean(axis=(0,1))[pval_x_pred>=pval_crit]    
        X_d_t = np.sum(weights_pred*X_d_t)
        if mu_clim_t+X_d_t<self.a:
            X_d_t = self.a - mu_clim_t
        if mu_clim_t+X_d_t>self.b:
            X_d_t = self.b - mu_clim_t    
        
        predictors_tau_mu = np.array([mu_clim_tau,
                                      X_d_tau])
        predictors_t_mu = np.array([mu_clim_t,
                                      X_d_t])       
    
        ########################################
        ######### sigma predictors #############
        ########################################
        
        # sigma climatology for the obs (this is a weighted standard deviation)
        Y_d_tau = np.zeros(Y_pint.shape)
        Y_d_tau[:,pval_y_pint<=pval_crit] = Y_pint[:,pval_y_pint<=pval_crit] - (y_coeffs_pint[0][pval_y_pint<=pval_crit,np.newaxis]*tau_t + y_coeffs_pint[1][pval_y_pint<=pval_crit][:,np.newaxis]).T
        Y_d_tau[:,pval_y_pint>pval_crit] = Y_pint[:,pval_y_pint>pval_crit] - Y_pint[:,pval_y_pint>pval_crit].mean(axis=0)
        # std_clim_tau = np.std(Y_d_tau, ddof=1, axis=0) + np.mean(Y_pint_std,axis=0)
        std_clim_tau = np.std(Y_d_tau, ddof=1, axis=0)
        std_clim_tau = np.ones(N_tau)*np.sum(weights_pint*std_clim_tau)
        # std_clim+=TINY
     
        std_clim_t = std_clim_tau[0]
        
        ############## s_x predictor ##############
        s_x_tau = np.sum(weights_pred[np.newaxis,:]*np.std(X_pred,ddof=1,axis=1),axis=-1)
        s_x_t = np.sum(weights_pred*np.std(X_t_pred,ddof=1,axis=0))
        
    
        predictors_tau_std = np.array([std_clim_tau,
                                       s_x_tau])
        predictors_t_std = np.array([std_clim_t,
                                     s_x_t])
    
        predictors_tau = np.array([predictors_tau_mu,predictors_tau_std]) 
        predictors_t = np.array([predictors_t_mu,predictors_t_std])     

        model_result = namedtuple('predictors_result', ('predictors_tau', 'predictors_t'))    
        
        return model_result(predictors_tau, predictors_t)
    
    def compute_coeffs(self, predictors_tau, Y_tar, params0=None,
                       es_tol=1e-6, maxiter=None, disp=False):    
        r'''
        Estimae the calibration coefficients :math:`\alpha_1`, :math:`\beta_1`,
        :math:`\alpha_2`, and :math:`\beta_2` for the case in which
        observational uncertainty *IS NOT* included in the CRPS cost function. This
        function minimizes the CRPS over the training set of predictors and observations,
        and outputs the optimized calibration coefficients.
        
        Parameters:        
            predictors_tau : ndarray, shape=(2,2,N_taut)
                One of the returned attributes from 
                `ncgr_sic_gridpoint.get_model`.
                
            Y_tar : ndarray, shape=(N_taut,)
                The observations over the training period `tau_t`.
                
            params0 : ndarray or list, shape=(4,) or if list len(params0)=4
                The initial guesses for the calibration coefficients :math:`\alpha_1`, 
                :math:`\beta_1`, :math:`\alpha_2`, and :math:`\beta_2`. Default is None, in 
                which case they're set to [1.0, 1.0, 0.5, 0.5].
            
            es_tol : float or None, optional
                Early stopping threshold used for minimizing the CRPS. 
                By default ``es_tol=1e-6``. Specifically, this argument
                sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  
                
            maxiter : int, optional
                The maximum number of iterations used for minimizing the CRPS. Specifically, this argument
                sets the ``maxiter`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  
                
            disp : True or False, optional
                If True, the output message of :py:meth:`scipy.optimize.minimize` will be included following
                minimization of the CRPS. By default, ``disp=False``.
     
    
        Returns:
            result : ncgr_result object with attributes
                params : list, length=4:
                    The optimized values of :math:`\alpha_1`, 
                    :math:`\beta_1`, :math:`\alpha_2`, and :math:`\beta_2`.
                    
                message : `OptimizeResult` 
                    `OptimizeResult` object returned
                    from `scipy.stats.minimize`.
        '''
        # set initial parameter guesses            
        crps_funcs_ = crps_funcs(self.a, self.b)
        ############################################################ 
        if maxiter is None:
            res_beinf = minimize(crps_funcs_.crps_ncgr, params0, args=(predictors_tau,Y_tar),
                                 jac=crps_funcs_.crps_ncgr_jac,
                                 tol=es_tol, 
                                 options={'disp':disp}, 
                                 constraints=self.build_cons(predictors_tau,Y_tar))
        else:
            res_beinf = minimize(crps_funcs_.crps_ncgr, params0, args=(predictors_tau,Y_tar),
                                 jac=crps_funcs_.crps_ncgr_jac,
                                 tol=es_tol, 
                                 options={'disp':disp,'maxiter':maxiter}, 
                                 constraints=self.build_cons(predictors_tau,Y_tar))                         
                        
       
        if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or (res_beinf.fun<0.0)&(np.isclose(res_beinf.fun,0)==False):
            # print("trying to re-set params0")
            params0 = np.array([1.0, 1.0, 0.5, 0.5])
            res_beinf = minimize(crps_funcs_.crps_ncgr, params0, args=(predictors_tau,Y_tar),
                                 jac=crps_funcs_.crps_ncgr_jac,
                                 tol=es_tol, 
                                 options={'disp':disp}, 
                                 constraints=self.build_cons(predictors_tau,Y_tar))
    
            if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or (res_beinf.fun<0.0)&(np.isclose(res_beinf.fun,0)==False):
                print("ERROR: Minimization couldn't converge!! This shouldn't happen; contact arlan.dirkson@gmail.com")
                print("params0:",params0)
                print("Predictors_mu",predictors_tau[0])
                print("Predictors_sigma",predictors_tau[1])
                raise(ValueError)
            else:
                params_es = res_beinf.x
                # print("re-setting params0 was successful, minimization converged")
    
        else:
            params_es = res_beinf.x
        
                
        ncgr_result = namedtuple('ncgr_result', ('params','message'))
        
        return ncgr_result(params_es, res_beinf)
    
    
    def compute_coeffs_obsunc(self, predictors_tau, F_obs, x, params0=None,
                       es_tol=1e-6, maxiter=None, disp=False):    
        r'''
        Computes the calibration coefficients :math:`\alpha_1`, :math:`\beta_1`,
        :math:`\alpha_2`, and :math:`\beta_2` for the case in which
        observational uncertainty *IS* included in the CRPS cost function. This
        function minimizes the CRPS over the training set of predictors and observations,
        and outputs the optimized calibration coefficients.
        
        Parameters:        
            predictors_tau : ndarray, shape=(2,2,N_taut)
                One of the returned attributes from 
                `ncgr_sic_gridpoint.get_model`.
                
            F_obs : ndarray, shape=(N_taut,nx)
                The CDF for the observation evaluated at values
                `x` (specified in next argument). nx=len(x).
                
            x : ndarray, shape=(Nx,)
                The values of SIC that were used to compute the observed CDF; 
                a reasonable rane for x would be
                x = np.linspace(0.0, 1.0, 101) if a=0 and b=1.
                
            params0 : ndarray or list, shape=(4,) or if list len(params0)=4
                The initial guesses for the calibration coefficients :math:`\alpha_1`, 
                :math:`\beta_1`, :math:`\alpha_2`, and :math:`\beta_2`. Default is None, in 
                which case they're set to [1.0, 1.0, 0.5, 0.5].
            
            es_tol : float or None, optional
                Early stopping threshold used for minimizing the CRPS. 
                By default ``es_tol=1e-6``. Specifically, this argument
                sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  
                
            maxiter : int, optional
                The maximum number of iterations used for minimizing the CRPS. Specifically, this argument
                sets the ``maxiter`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  
                
            disp : True or False, optional
                If True, the output message of :py:meth:`scipy.optimize.minimize` will be included following
                minimization of the CRPS. By default, ``disp=False``.
     
    
        Returns:
            result : ncgr_result object with attributes
                params : list, length=4:
                    The optimized values of :math:`\alpha_1`, 
                    :math:`\beta_1`, :math:`\alpha_2`, and :math:`\beta_2`.
                    
                message : `OptimizeResult` 
                    `OptimizeResult` object returned
                    from `scipy.stats.minimize`.
                
            
        '''
        if params0 is None:
            params0 = np.array([1.0, 1.0, 0.5, 0.5])
            
        # set initial parameter guesses            
        crps_funcs_ = crps_funcs(self.a, self.b)
        ############################################################ 
        if maxiter is None:
            res_beinf = minimize(crps_funcs_.crps_ncgr_obsunc, params0, args=(predictors_tau,F_obs,x),
                                 tol=es_tol, 
                                 options={'disp':disp}, 
                                 constraints=self.build_cons(predictors_tau,None))
        else:
            res_beinf = minimize(crps_funcs_.crps_ncgr_obsunc, params0, args=(predictors_tau,F_obs,x),
                                 tol=es_tol, 
                                 options={'disp':disp,'maxiter':maxiter}, 
                                 constraints=self.build_cons(predictors_tau,None))                         
       
        if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or (res_beinf.fun<0.0)&(np.isclose(res_beinf.fun,0)==False):
            # print("trying to re-set params0")
            params0 = np.array([1.0, 1.0, 0.5, 0.5])
            res_beinf = minimize(crps_funcs_.crps_ncgr_obsunc, params0, args=(predictors_tau,F_obs),
                                 tol=es_tol, 
                                 options={'disp':disp}, 
                                 constraints=self.build_cons(predictors_tau,None))
    
            if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or (res_beinf.fun<0.0)&(np.isclose(res_beinf.fun,0)==False):
                print("ERROR: Minimization couldn't converge!! This shouldn't happen; contact arlan.dirkson@gmail.com")
                print("params0:",params0)
                print("Predictors_mu",predictors_tau[0])
                print("Predictors_sigma",predictors_tau[1])
                raise(ValueError)
            else:
                params_es = res_beinf.x
                # print("re-setting params0 was successful, minimization converged")
    
        else:
            params_es = res_beinf.x
        
                
        ncgr_result = namedtuple('ncgr_result', ('params','message'))
        
        return ncgr_result(params_es, res_beinf)
    
    
    def calibrate(self, predictors_t, params, 
                      sic_thresh=np.array([0.15,0.45,0.75])):  
        '''
        This function applies the regression equations built using
        the predictors for the forecast to be calibrated
        and optimized calibration coefficients. It outputs the calibrated 
        DCNORM distribution parameters and sea-ice probabilities for specified
        ice concentration thresholds.

        Parameters
        ----------
        predictors_t : ndarray, shape=(2,2)
            The four regression predictors for the forecast
            to be calibrated. The values in 
            `predictors_tau[0]` are for the equation for :math:`\mu`. 
            The values in `predictors_tau[1]` are for
            the equation for :math:`\sigma`. The first predictor is
            the pseudo-intercept, the second predictor is the ensemble-based 
            statistic. The `predictors_t` array is an output of 
            `ncgr_sic_gridpoint.get_model`.
        params : list, length=4:
            The optimized values of :math:`\alpha_1`, 
            :math:`\beta_1`, :math:`\alpha_2`, and :math:`\beta_2`. `params` is
            an output of `ncgr_sic_gridpoint.compute_coeffs` or 
            `ncgr_sic_gridpoint.compute_coeffs_obsunc`.
        sic_thresh : ndarray, shape(N_sip,)
            The sea-ice concentration threshold(s) for which 
            to compute the "sea ice probability"; i.e. the probability that
            SIC>sic_thresh. The default is np.array([0.15,0.45,0.75]),
            in which case N_sip=3.

        Returns
        -------
        result, calibrate_result:
            An object with the following attibutes:
                mu : float
                    The :math:`\mu` parameter for the calibrated forecast DCNORM distribution
                sigma : float
                    The :math:`\sigma` parameter for the calibrated forecast DCNORM distribution
                sip : ndarray, shape (N_sip,)
                    The sea-ice probability; i.e. the probability that
                    SIC>sic_thresh.
        '''
        
        N_pred_mu = predictors_t[0].shape[0]
        N_pred_s = predictors_t[1].shape[0] 
                                            
        params_mu, params_std = params[0:N_pred_mu], params[N_pred_mu:N_pred_mu+N_pred_s]
        
        mu_cal = min(max(self.a-eps_mu, -eps_mu + np.dot(params_mu.T,predictors_t[0])),self.b+eps_mu) # constrain to [a-eps_mu,b+eps_mu]
        sigma_cal = max(eps_sigma, eps_sigma+np.dot(params_std.T,predictors_t[1])) # constrain to [eps_sigma,inf]
        
        fcst_probs = 1.0 - self.dcnorm.cdf(sic_thresh,mu_cal,sigma_cal)
        
        ncgr_fm_result = namedtuple('calibrate_result', ('mu', 'sigma','sip'))
        
        return ncgr_fm_result(mu_cal, sigma_cal, fcst_probs)
    
    def build_cons(self,predictors,y):
        '''
        Builds a dictionary for the constrainst on the DCNORM distribution parameters
        when calling on :py:class:`scipy.optimize.minimize` 
        
        Returns:
            cons : dict 
                Contains the constraint callables used in :py:class:`scipy.optimize.minimize`.
        '''
        
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # number of predictors for sigma
        
        def con_mu1(params, predictors, y):    
            params_mu = params[:N_pred_m]
                
            predictors_mu = predictors[0]
                                   
            mu_hat = -eps_mu + np.dot(params_mu.T,predictors_mu)
            
            return mu_hat - (self.a-eps_mu)
    
        def con_mu2(params, predictors, y):    
            params_mu = params[:N_pred_m]
                
            predictors_mu = predictors[0]
                                   
            mu_hat = -eps_mu + np.dot(params_mu.T,predictors_mu)
            
            return self.b + eps_mu - mu_hat
        
        def con_std(params, predictors, y):
            params_s = params[N_pred_m:N_pred_m+N_pred_s]
            
            predictors_s = predictors[1]    
    
            s_hat = eps_sigma+np.dot(params_s.T,predictors_s)     
            
            return s_hat - eps_sigma
        
        cons = ({'type': 'ineq', 'fun': con_mu1, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_mu2, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_std, 'args':(predictors,y)})
            
        return cons

class crps_funcs():
    
    def __init__(self,a,b):
        '''
        This class contains functions needed to perform CRPS minimization for the DCNORM distribution.
        It also contains a function for computing the CRPS when the forecast distribution
        takes the form of a DCNORM distribution (as it does for NCGR).
        
        Args:
            a (float or int):
                Minimum possible date for the event in non leap year
                day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
                larger than 365 is regarded as a date for the following year.
                
            b (float or int):
                Maximum possible date for the event in non leap year 
                day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
                larger than 365 is regarded as a date for the following year. The 
                ``b`` argument must be larger than the ``a`` argument.
    
        
        The methods contained in this class are:
    
        ``crps_dcnorm()``
            Computes the CRPS for a set of forecsts and observations
            when the predictive distribution takes the form of a 
            DCNORM distribution.
                    
        ``crps_ncgr()``
            The cost function used when executing ``scipy.optimize.mimize``
            in the ``calibrate`` method. Computes the mean CRPS as a function of a set
            of hindcast CDFs (modelled by NCGR) and observed dates.
            
        ``crps_ncgr_jac()``
            Called on in the ``calibrate`` method. 
            Computes the jacobian matrix for the CRPS cost function.
            
        ``crps_singleyear()``
            Called on in the ``calibrate`` method. 
            Computes the CRPS for a single forecast CDF (modelled as a DCNORM
            distribution) and observation.
               
        '''

        self.a = a
        self.b = b   
        self.dcnorm = dcnorm_gen(a=a, b=b)
  

    def crps_dcnorm_single(self, y, mu, sigma):
        '''
        Continuous rank probability score (CRPS) for a single forecast when the forecast distribution
        takes the form of a DCNORM distribution, and the verification is an observed value.

        Args:
            y (float or int):
                Observed value.
            
            mu (float or int):
                DCNORM parameter :math:`\mu`.
                
            sigma (float or int):
                DCNORM parameter :math:`\sigma`
                
        Returns:
            result (float):
                CRPS
                
        '''

        rv = norm()        
        a_star = (self.a-mu)/sigma
        b_star = (self.b-mu)/sigma
        y_star = (y-mu)/sigma
    
        t1 = -sigma*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
        t2 = sigma*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
        t3 = 2*sigma*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
        t4 = sigma*(b_star - y_star)
        
        result = t1 + t2 + t3 + t4
    
        return result[0]     

    def crps_dcnorm(self,y,mu,sigma):
        '''
        Time mean continuous rank probability score (CRPS) when the distribution
        takes the form of a DCNORM distribution.

        Args:
            y (ndarray), shape (`n`,):
                Observed dates, where `n` is the number of
                forecast/observation pairs.
            
            mu (ndarray), shape (`n`,):
                DCNORM parameter :math:`\mu` for each of the `1,...,n` forecast distributions.
                
            sigma (ndarray), shape (`n`,):
                DCNORM parameter :math:`\sigma` for each of the `1,...,n` forecast distributions.
                
        Returns:
            result (float):
                Time mean CRPS.
                
        '''

        N = len(y)
        crps = np.zeros(N)
        rv = norm()
        for ii in np.arange(N):           
            a_star = (self.a-mu[ii])/sigma[ii]
            b_star = (self.b-mu[ii])/sigma[ii]
            y_star = (y[ii]-mu[ii])/sigma[ii]
        
            t1 = -sigma[ii]*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
            t2 = sigma[ii]*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
            t3 = 2*sigma[ii]*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma[ii]*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
            t4 = sigma[ii]*(b_star - y_star)
            
            crps[ii] = t1 + t2 + t3 + t4
        
        result = np.mean(crps)
        return result
  
    
    def crps_obsunc(self,F_obs,mu,sigma,x):
        '''
        Time mean continuous rank probability score (CRPS) when the distribution
        takes the form of a DCNORM distribution.

        Args:
            F_obs (ndarray), shape (`n`,):
                CDF for observation
            
            mu (ndarray), shape (`n`,):
                DCNORM parameter :math:`\mu` for each of the `1,...,n` forecast distributions.
                
            sigma (ndarray), shape (`n`,):
                DCNORM parameter :math:`\sigma` for each of the `1,...,n` forecast distributions.
                
        Returns:
            result (float):
                Time mean CRPS.
                
        '''

        N = len(mu)
        crps = np.zeros(N)
        
        for ii in np.arange(N):       
            cdf_fcst = self.dcnorm.cdf(x,mu[ii],sigma[ii])
            crps[ii] = np.trapz((cdf_fcst-F_obs[ii])**2., x)
        
        result = np.mean(crps)
        return result    
    
    def crps_ncgr(self, coeffs, predictors, y):  
        '''
        Args:            
            coeffs (list), shape (`m`,):
                Coefficients in the NCGR regression equations, 
                where `m` is the total number of coefficients/predictors. The first two values
                correspond to those for :math:`\mu` and the remaining values
                correspond to those for :math:`\sigma`.
                
                
            predictors (object), shape (`n`,):
                Object containing the predictors, where `n=2` is the number of distribution parameters.
                The shape of either predictors[0] or predictors[1] is (`m,p`), where
                `m` is the number of coefficients/predictors for the corresponding parameter, and `p` is the number of
                years in the training period ``self.tau_t``.
                
            y (ndarray), shape (`p`,):
                Array of observed dates, where `p` is the number of
                years in the training period ``self.tau_t``.
                
        Returns:
                The time-averaged continuous rank probability score (CRPS).

        '''
        
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # numebr of preidctors for sigma
        
        # get the coefficients and predictors for the regression equation for mu
        params_m = coeffs[:N_pred_m]     
        predictors_m = predictors[0]
        
        # get the coefficients and predictors for the regression equation for sigma
        params_s = coeffs[N_pred_m:N_pred_m+N_pred_s]
        predictors_s = predictors[1]    
        
        mu = -eps_mu + np.dot(params_m.T,predictors_m) # take linear combination of preidictors and coeffs for mu
        sigma = eps_sigma + np.dot(params_s.T,predictors_s) # "" "" for sigma
    
        return self.crps_dcnorm(y, mu, sigma)
 
    
    def crps_ncgr_obsunc(self, coeffs, predictors, F_obs, x):  
        '''
        Args:            
            coeffs (list), shape (`m`,):
                Coefficients in the NCGR regression equations, 
                where `m` is the total number of coefficients/predictors. The first two values
                correspond to those for :math:`\mu` and the remaining values
                correspond to those for :math:`\sigma`.
                
                
            predictors (object), shape (`n`,):
                Object containing the predictors, where `n=2` is the number of distribution parameters.
                The shape of either predictors[0] or predictors[1] is (`m,p`), where
                `m` is the number of coefficients/predictors for the corresponding parameter, and `p` is the number of
                years in the training period ``self.tau_t``.
                
            y (ndarray), shape (`p`,):
                Array of observed dates, where `p` is the number of
                years in the training period ``self.tau_t``.
                
        Returns:
                The time-averaged continuous rank probability score (CRPS).

        '''
        
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # numebr of preidctors for sigma
        
        # get the coefficients and predictors for the regression equation for mu
        params_m = coeffs[:N_pred_m]     
        predictors_m = predictors[0]
        
        # get the coefficients and predictors for the regression equation for sigma
        params_s = coeffs[N_pred_m:N_pred_m+N_pred_s]
        predictors_s = predictors[1]    
        
        mu = -eps_mu + np.dot(params_m.T,predictors_m) # take linear combination of preidictors and coeffs for mu
        sigma = eps_sigma + np.dot(params_s.T,predictors_s) # "" "" for sigma

        return self.crps_obsunc(F_obs,mu,sigma,x)    
    
    def crps_ncgr_jac(self, coeffs, predictors, y):
        '''
        Args:
            coeffs (list), shape (`n+m`):
                Coefficients in the NCGR regression equations, 
                where `n=2` is the number of distribution parameters 
                and `m` is the number of predictors for a given parameter. The first
                two values are the coefficients for :math:`\mu` and the 
                remaining values are the coefficeints for :math:`\sigma`.
                
                
            predictors (object), shape (`n`,):
                Object containing the predictors, where `n=2` is the number of distribution parameters.
                The shape of either predictors[0] or predictors[1] is (`m,p`), where
                `m` is the number of coefficients/predictors for the corresponding parameter, and `p` is the number of
                years in the training period ``self.tau_t``.
                
            y (ndarray), shape (`p`):
                Array of observed dates, where `p` is the number of
                years in the training period ``self.tau_t``.
                
        Returns:
            (ndarray), shape (m,):
                The jacobian matrix of the time-averaged continuous rank probability score.
        '''

        N = len(y) # number of years the CRPS will averaged over
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # numebr of preidctors for sigma
        
        # get the coefficients and predictors for the regression equation for mu
        params_m = coeffs[:N_pred_m]     
        predictors_m = predictors[0]
        
        # get the coefficients and predictors for the regression equation for sigma
        params_s = coeffs[N_pred_m:N_pred_m+N_pred_s]
        predictors_s = predictors[1]    
        
        mu = -eps_mu + np.dot(params_m.T,predictors_m) # take linear combination of preidictors and coeffs for mu
        sigma = eps_sigma + np.dot(params_s.T,predictors_s) # "" "" for sigma

        def T_mu(z):
            return rv.cdf(z)**2. + 2*rv.pdf(z)**2. 
        
        def T_std(z):
            return z*rv.cdf(z)**2. + 2*z*rv.pdf(z)**2. 

        rv = norm()
        jac = np.zeros((N,N_pred_m+N_pred_s))
    
        for ii in np.arange(N):        
            a_star = (self.a-mu[ii])/sigma[ii]
            b_star = (self.b-mu[ii])/sigma[ii]
            y_star = (y[ii]-mu[ii])/sigma[ii]

        
            jac_mu = T_mu(a_star) - T_mu(b_star) \
                    +np.sqrt(2.)/np.sqrt(np.pi) * (rv.pdf(b_star*np.sqrt(2)) - rv.pdf(a_star*np.sqrt(2))) \
                    +2.*(rv.cdf(b_star) - rv.cdf(y_star))
                    
                    
            jac_std = self.crps_ncgr_sy(np.array([mu[ii],sigma[ii]]), y[ii])/sigma[ii] + T_std(a_star) - T_std(b_star) \
                      +np.sqrt(2.)/np.sqrt(np.pi) * (b_star*rv.pdf(np.sqrt(2)*b_star) - a_star*rv.pdf(np.sqrt(2)*a_star)) \
                      +2.*(b_star*rv.cdf(b_star) - y_star*rv.cdf(y_star)) \
                      + y_star - b_star
                    
            jac[ii,:N_pred_m] = predictors_m[:,ii]*jac_mu
            jac[ii,N_pred_m:] = predictors_s[:,ii]*jac_std
                
    
        return np.mean(jac,axis=0)    
    
    def crps_ncgr_sy(self, params, y):
        '''
        Computes the continuous rank probability score
        for a single forecast with DCNORM distribution.
        
        Args:     
            params (list), shape (2,):
                List containing the two DCNORM distribution parameters 
                :math:`\mu` and :math:`\sigma`.
                
            y (float):
                The observation.
                
        Returns:
            result (float):
                The CRPS.

        '''        
        mu, sigma = params.T
        rv = norm()
        a_star = (self.a-mu)/sigma
        b_star = (self.b-mu)/sigma
        y_star = (y-mu)/sigma
        
        t1 = -sigma*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
        t2 = sigma*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
        t3 = 2*sigma*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
        t4 = sigma*(b_star - y_star)
        
        result = t1 + t2 + t3 + t4    
        return result
    
    
