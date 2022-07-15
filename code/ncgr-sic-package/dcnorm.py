# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rv_continuous, norm
from scipy.optimize import minimize

eps_sigma = 1e-6 # to constrain \sigma>=1e-6 during MLE
eps_mu = 1. # to constrain a-1<=\mu<=b+1 during MLE
class dcnorm_gen(rv_continuous):            
    r"""A doubly-censored normal (DCNORM) random variable, censored numerically below :math:`a` and
    above :math:`b`.

    Notes
    -----
    The probability density function for a DCNORM random variable is:
    
        .. math::            
           f(x;\mu,\sigma)=  \Phi\left( \frac{a-\mu}{\sigma}\right)\delta(x-a) +
                       \frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)1_{(a,b)} + 
                       \left[1 - \Phi\left( \frac{b-\mu}{\sigma}\right)\right]\delta(x-b),
     
    where :math:`1_{(a,b)}=1` when :math:`a<x<b` and :math:`1_{(a,b)}=0` otherwise; :math:`\delta(x)` is the delta function;
    :math:`\phi(\cdot)` and :math:`\Phi(\cdot)` are respectively the PDf and CDF for a standard normal distribution with
    zero mean and unit variance. The support is :math:`a\leq X \leq b`, and requirements are :math:`\infty<\mu<\infty` and
    :math:`\sigma>0`.


    :class:`dcnorm_gen` is an instance of a subclass of :py:class:`~scipy.stats.rv_continuous`, and therefore
    inherits all of the methods within :py:class:`~scipy.stats.rv_continuous`. Some of those methods
    have been subclassed here:
    
    ``_argcheck``

    ``_cdf``    
    
    ``_pdf``

    ``_ppf``
    
    ``_stats``

    
    Additional methods added to :class:`dcnorm` are: 
        
    ``fit_moments(mean, std)``    
        Fit a sample to the DCNORM distribution
        using the method of moments (i.e. based on its sample mean and sample standard deviation).
        
    ``ecdf(x, data)``
        The empirical distribution function for a sample ``data`` at values ``x``.
        
    
    ``fit(data)``
        This replaces the ``fit`` method in :py:class:`~scipy.stats.rv_continuous`. 
        Computes the maximum likelihood estimates for the DCNORM distribution
        parameters following [2].
        
    Examples
    --------
    >>> from dcnorm import dcnorm_gen
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    Set the minimum and maximum values allowed for the DCNORM distribution
    
    >>> a=0.0
    >>> b=1.0
    
    Instantiate the DCNORM distribution class with the given min/max values
    
    >>> dcnorm = dcnorm_gen(a=a,b=b)
            
    Variables for the :math:`\mu` and :math:`\sigma` parameters for the DCNORM distribution 
    
    >>> m=0.15
    >>> s=0.3
    
    Create a distribution object (see :py:class:`~scipy.stats.rv_continuous` for details on available methods)
    
    >>> rv = dcnorm(m,s)
    
    As an example, create a sample of 100 random draws from this distribution
    
    >>> X = rv.rvs(size=100)
    
    Now fit the sample to a DCNORM distribution (this is done using maximum likelihood (ML) estimation) 
    
    >>> m_f, s_f = dcnorm.fit(X)
    
    Make a new distribution object with the ML estimates computed previously
    
    >>> rv_f = dcnorm(m_f, s_f)
    
    Make range of values for plotting
    
    >>> x = np.linspace(a, b, 1000)

    Plot the emperical CDF for the sample `X`, the true distribution defined originally, and the 
    distribution obtained by fitting to X.
    
    >>> plt.figure()    
    >>> plt.plot(x, dcnorm.ecdf(x,X), 'k--', label='empirical CDF')
    >>> plt.plot(x, rv.cdf(x), 'k-', label='true CDF')
    >>> plt.plot(x, rv_f.cdf(x), 'r-', label='fitted CDF')   
    >>> plt.legend()    
    >>> plt.show()
    
    """

#####################################################################    
######################### Subclassed Methods ########################
#####################################################################

    def _pdf(self, x, m, s):
        '''
        Subclass the _pdf method (returns the pdf of the 
        DCNORM distribution at x)
        '''
            
        # the ranges of x that break up the piecewise
        # pdf
        condlist = [x<self.a,
                    x==self.a, 
                    np.logical_and(x>self.a, x<self.b),
                    x==self.b,
                    x>self.b]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [0.0,
                      norm.cdf(self.a,loc=m,scale=s),
                      norm.pdf(x,loc=m,scale=s),
                      1.0-norm.cdf(self.b,loc=m,scale=s),
                      0.0]
            
        return np.select(condlist, choicelist)
        
    def cdf(self, x, m, s):         
        # the ranges of x that break up the piecewise
        # cdf
        condlist = [x<self.a,
                    x==self.a, 
                    np.logical_and(x>self.a, x<self.b),
                    x>=self.b]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [0.0,
                      norm.cdf(self.a,loc=m,scale=s),
                      norm.cdf(x,loc=m,scale=s),
                      1.0]
        
        return np.select(condlist, choicelist)


    def _ppf(self, rho, m, s):
        '''
        Returns the inverse of the cumulative distribution function for
        the DCNORM distribution at probabilities rho.
        '''
        condlist = [np.logical_and(rho>=0,rho<=norm.cdf(self.a,loc=m,scale=s)),
                    np.logical_and(rho>norm.cdf(self.a,loc=m,scale=s),rho<norm.cdf(self.b,loc=m,scale=s)), 
                    rho>=norm.cdf(self.b,loc=m,scale=s)]
        # the piecewise pdf associated with the entries in condlist
        choicelist = [self.a,
                      norm.ppf(rho,loc=m,scale=s),
                      self.b]        
        return np.select(condlist, choicelist)


        
    def fit(self, y):    
        ##########################################################################################
        ################# Performs MLE on the data using the DCNORM distribution #################
        ################# based on closed-form solutions of the gradient of the ##################
        ################# log likelihood function for the DCNORM distribution ####################
        ################# set to 0; when no a's or b's are present in the data ###################
        ################# this problem reduces to the MLE of the normal distribution #############
        ##########################################################################################
        
        if np.any((y<=self.a)|(y>=self.b)):    
            def loglike(params,y):
                m, s = params.T
                
                y_sub = y[(y!=self.a)&(y!=self.b)]
    
                na = float(len(y[y==self.a]))
                nb = float(len(y[y==self.b]))
                
                a_star = (self.a - m)/s
                b_star = (self.b - m)/s
                
                # need these if statements to avoid singularities when na=0 or nb=0
                if na>0.0 and nb==0.0:
                    T1 = na*np.log(norm.cdf(a_star)) 
                    T2 = np.sum(-0.5*np.log(2*np.pi*s**2.) - (y_sub-m)**2./(2*s**2.)) 
                    return -(T1+T2)
                elif nb>0.0 and na==0.0:
                    T2 = np.sum(-0.5*np.log(2*np.pi*s**2.) - (y_sub-m)**2./(2*s**2.))
                    T3 = nb*np.log(1 - norm.cdf(b_star)) 
                    return -(T2+T3)                  
                else:
                    T1 = na*np.log(norm.cdf(a_star)) 
                    T2 = np.sum(-0.5*np.log(2*np.pi*s**2.) - (y_sub-m)**2./(2*s**2.))
                    T3 = nb*np.log(1 - norm.cdf(b_star))                               
                    return -(T1+T2+T3)
                
            params0 = [np.mean(y), np.std(y,ddof=1)]
            
            # minimize the negative of the log-likelihood of the DCNORM distribution
            res = minimize(loglike, params0, bounds=((self.a-eps_mu,self.b+eps_mu),(eps_sigma,np.inf)), jac = self.jac_loglike, args=(y,))
            m_hat, s_hat = res.x
        else:
            m_hat, s_hat = norm.fit(y)
        
        return m_hat, s_hat
    
    def jac_loglike(self, params, y):    
        ##########################################################################################
        ################################ Jacobian of log-likelihood ##############################
        ##########################################################################################
        m, s = params.T
        
        y_sub = y[(y!=self.a)&(y!=self.b)]

        na = float(len(y[y==self.a]))
        nb = float(len(y[y==self.b]))
        n0 = float(len(y_sub))
        
        a_star = (self.a - m)/s
        b_star = (self.b - m)/s
        
        # derivative of the log likelihood with respect to mu
        # need these if statements to avoid singularities when na=0 or nb=0
        if na>0 and nb>0:
            eq1 = -na/s*norm.pdf(a_star)/norm.cdf(a_star) + 1/s**2. * np.sum(y_sub - m) + nb/s*norm.pdf(b_star)/(1-norm.cdf(b_star))
        elif na==0 and nb>0:
            eq1 = 1/s**2. * np.sum(y_sub - m) + nb/s*norm.pdf(b_star)/(1-norm.cdf(b_star))
        elif na>0 and nb==0:
            eq1 = -na/s*norm.pdf(a_star)/norm.cdf(a_star) + 1/s**2. * np.sum(y_sub - m) 
            
        
        # derivative of the log likelihood with respect to sigma
        # need these if statements to avoid singularities when na=0 or nb=0
        if na>0 and nb>0:
            T1 = -na/s*a_star*norm.pdf(a_star)/norm.cdf(a_star)
            T2 = -n0/s + 1/s**3. * np.sum((y_sub-m)**2.)
            T3 = nb/s*b_star*norm.pdf(b_star)/(1-norm.cdf(b_star))
            eq2 = T1 + T2 + T3
        elif na>0 and nb==0.0:
            T1 = -na/s*a_star*norm.pdf(a_star)/norm.cdf(a_star)
            T2 = -n0/s + 1/s**3. * np.sum((y_sub-m)**2.)
            eq2 = T1 + T2
        elif na==0 and nb>0:
            T2 = -n0/s + 1/s**3. * np.sum((y_sub-m)**2.)
            T3 = nb/s*b_star*norm.pdf(b_star)/(1-norm.cdf(b_star))
            eq2 = T2 + T3
        
        return np.array([-eq1, -eq2])
            

                
    def _stats(self,m,s,moments='mvsk'):
        
       a_star = (self.a - m)/s
       b_star = (self.b - m)/s
       
       # mean
       m_t1 = self.a*norm.cdf(a_star)
       m_t2 = m*(norm.cdf(b_star) - norm.cdf(a_star)) + s*(norm.pdf(a_star)-norm.pdf(b_star))
       m_t3 = self.b*(1-norm.cdf(b_star))
       mean = m_t1 + m_t2 + m_t3
       
       # variance
       v_t1 = self.a**2. * norm.cdf(a_star)
       v_t2 = s**2. * (norm.cdf(b_star) - b_star*norm.pdf(b_star) - norm.cdf(a_star) + a_star*norm.pdf(a_star))
       v_t3 = 2*m*s * (norm.pdf(a_star) - norm.pdf(b_star))
       v_t4 = m**2. * (norm.cdf(b_star) - norm.cdf(a_star))
       v_t5 = self.b**2. * (1 - norm.cdf(self.b,loc=m,scale=s))       
       var = v_t1 + v_t2 + v_t3 + v_t4 + v_t5 - mean**2.
       
       return mean, var, None, None
   
    def fit_moments(self, mean, std):
        '''
        Fit data to the DCNORM distribution using the method of moments

        Parameters
        ----------
        mean : float
            Sample mean of the data to be fit.
        std : float
            Sample standard deviation of the data to be fit.

        Returns
        -------
        result : ndarray
            An array with the estimated DCNORM parameters np.array([mu, sigma])

        '''
        def moments_invert_min(p, e, s):
            mu, sigma = p
            f = [self.mean(mu,sigma)-e, self.std(mu,sigma)-s]
            return np.dot(f,f)
        
        def build_cons_moms():
            # constraints on the estimated parameters
            def con_mu1(params):  
                mu, sigma = params
                return mu - (0.0-eps_mu)
        
            def con_mu2(params):    
                mu, sigma = params        
                return 1.0 + eps_mu - mu
            
            def con_std(params):
                mu, sigma = params                
                return sigma - eps_sigma
            
            cons = ({'type': 'ineq', 'fun': con_mu1},
                    {'type': 'ineq', 'fun': con_mu2},
                    {'type': 'ineq', 'fun': con_std})
                
            return cons
        

        params0 = [mean, std]
        
        result = minimize(moments_invert_min, params0,
                  constraints=build_cons_moms(), args=(mean, std))
        
        return result.x
       
    def _argcheck(self,m,s):
        #subclass the argcheck method to ensure parameters
        #are constrained to their bounds
        # check = (m>=self.a-eps_mu)&(m<=self.b+eps_mu)&(s>=eps_sigma)
        check = s > 0.0
        if check==True:
            return True
        else:
            return False

#####################################################################    
######################### Methods Unique ############################
#########################  to the dcnorm  ############################
#########################  distribution  ############################
#####################################################################
            
    def ecdf(self, x, data):
        r'''
        For computing the empirical cumulative distribution function (ecdf) of a
        given sample.
        
        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated
               
            data (float or ndarray):
                A sample for which to compute the ecdf.
                
        Returns: ecdf_vals (ndarray):            
            The ecdf for X_samp, evaluated at x.
            
        '''
        
        if isinstance(x,np.float):
            #if x comes in as float, turn it into a numpy array
            x = np.array([x])

        if isinstance(data,np.float):
            #if X_samp comes in as float, turn it into a numpy array
            data = np.array([data])  
    
       
        # sort the values of X_samp from smallest to largest
        xs = np.sort(data)
        
        # get the sample size of xs satisfying xs<=x for each x
        def func(vals):
            return len(xs[xs<=vals])
        
        ys = [len(xs[xs<=vals]) for vals in x]
        

        return np.array(ys)/float(len(xs))



dcnorm = dcnorm_gen(name='dcnorm',shapes='m,s')
