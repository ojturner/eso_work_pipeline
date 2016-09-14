# 2d gaussina fitting
# given a set of input data, fit with a 2d gaussian
# and return the evaluated model
# O.J.Turner 2016

# import the relevant modules
import scipy.optimize as opt
import pylab as pyplt
import numpy as np
import matplotlib.pyplot as plt
from numpy import poly1d
import scipy
import numpy.ma as ma
from lmfit.models import GaussianModel, ConstantModel
from lmfit import Model
from astropy.io import fits
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from numpy import *

def ped_gauss_fit(fit_wl,
                  fit_flux):

    """
    Def:
    Performs simple gaussian fit, guessing initial parameters from the data
    and given input wavelength and input flux values - this time the base
    of the gaussian is left as a free parameter

    Input:
            fit_wl - wavelength of spectrum to fit
            fit_flux - flux of spectrum to fitsWavelength

    Output:
            fit_params - dictionary containing the best fit parameters
                        for each of the spectra
    """

    # construct gaussian model using lmfit

    gmod = GaussianModel()

    # construct constant model using lmfit

    cmod = ConstantModel()

    # set the initial parameter values

    pars = gmod.guess(fit_flux, x=fit_wl)

    # add an initial guess at the constant (0)

    pars += cmod.make_params(c=0)

    mod = cmod + gmod

    # perform the fit
    out = mod.fit(fit_flux, pars, x=fit_wl)

    # print the fit report
    # print out.fit_report()

    # plot to make sure things are working
#        fig, ax = plt.subplots(figsize=(14, 6))
#        ax.plot(fit_wl, fit_flux, color='blue')
#        ax.plot(fit_wl, out.best_fit, color='red')
#        plt.show()
#        plt.close('all')

    return out, out.best_values, out.covar