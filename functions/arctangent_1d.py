# Very Simple - Take an array of data and 
# associated distance measurements, fit an arctangent 
# function in one dimension and extrapolate across a 
# much wider area. Really I should be doing this rather
# than fitting a 2D gaussian function - a better way to
# get robust parameters out, which I'm not really using
# anyway.


# import the relevant modules
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyraf
import numpy.ma as ma
import pickle
from scipy.spatial import distance
from scipy import ndimage
from copy import copy
from lmfit import Model
from itertools import cycle as cycle
from lmfit.models import GaussianModel, PolynomialModel, ConstantModel
from scipy.optimize import minimize
from astropy.io import fits, ascii
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import poly1d
from sys import stdout
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry

def arctan(r, const, vasy, rt):

    """
    Def:
    return the result of evaluating an arctangent function
    """
    return const + (2 / np.pi) * vasy * np.arctan(r / float(rt))

def model_fit(data,
              r,
              weights,
              guess_v,
              guess_rt):

    """
    Def:
    Construct an arctangent model function
    and fit to the data and return the model fit object
    """

    mod = Model(arctan,
                independent_vars=['r'],
                param_names=['vasy', 'rt', 'const'],
                missing='drop')

    mod.set_param_hint('rt', value=guess_rt, min=0, max=40)

    mod.set_param_hint('vasy', value=guess_v, min=-400, max=400)

    mod.set_param_hint('const', value=0)

    fit_pars = mod.make_params()

    mod_fit = mod.fit(data, r=r, params=fit_pars, weights=weights)

    return mod_fit




    