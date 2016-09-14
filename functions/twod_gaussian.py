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
from lmfit.models import GaussianModel
from lmfit import Model
from astropy.io import fits
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from numpy import *


def moments_better(data,
                   circle=0,
                   rotate=1,
                   vheight=1,
                   estimator=np.nanmedian,
                   **kwargs):

    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above.

    If using masked arrays, pass estimator=np.ma.median
    """

    total = np.nansum(np.abs(data))

    Y, X = np.indices(data.shape) # python convention: reverse x,y np.indices
    y = np.argmax(np.nansum(X*np.abs(data),axis=1)/total)
    x = np.argmax(np.nansum(Y*np.abs(data),axis=0)/total)
    col = data[int(y),:]

    # FIRST moment, not second!
    width_x = np.sqrt(np.nansum(np.abs((np.arange(col.size)-y)*col))/np.nansum(np.abs(col)))
    row = data[:, int(x)]
    width_y = np.sqrt(np.nansum(np.abs((np.arange(row.size)-x)*row))/np.nansum(np.abs(row)))
    width = ( width_x + width_y ) / 2.
    height = estimator(data.ravel())
    amplitude = np.nanmax(data)-height
    # print width_x, width_y, height, amplitude
    mylist = [amplitude,x,y]
    if np.isnan(width_y) or np.isnan(width_x) or np.isnan(height) or np.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
        if rotate==1:
            mylist = mylist + [0.] #rotation "moment" is just zero...
            # also, circles don't rotate.
    else:
        mylist = mylist + [width]
    return mylist


def twoD_Gaussian((x, y),
                  offset,
                  amplitude,
                  xo,
                  yo,
                  sigma_x,
                  sigma_y,
                  theta):
    """
    Def: Define the 2D gaussian model
    """

    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))

    return g.ravel()

def fit_gaussian(data):

    """
    Def: Bring the above 2 methods together to fit a gaussian
    """
    # mask out the nan values using np.ma
    data_masked = np.ma.masked_invalid(data)

    # indices of the full array
    y_full, x_full = np.indices(data_masked.shape)

    data_masked_cut = copy(data_masked)

    # create the grid over which to evaluate the gaussian
    y, x = np.indices(data_masked_cut.shape)

    # find the moments of the data and use these as the initial
    # guesses for the gaussian

    list_of_moments = moments_better(data_masked_cut)

    # very important - have to set the nan values equal to the
    # evaluated height parameter

    data_masked_cut[np.isnan(data_masked_cut)] = list_of_moments[0]

    # fit the model
    popt, pcov = opt.curve_fit(twoD_Gaussian,
                               (x, y),
                               data_masked_cut.ravel(),
                               p0=list_of_moments,
                               bounds=([-np.inf,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        -np.pi/2.0],
                                        [np.inf,
                                         np.inf,
                                         data_masked_cut.shape[1],
                                         data_masked_cut.shape[0],
                                         data_masked_cut.shape[1],
                                         data_masked_cut.shape[0],
                                         np.pi/2.0]))

    # alter the gaussian centroid positions by the number of
    # cut pixels at the beginning
#    popt[2] = popt[2] + 2
#    popt[3] = popt[3] + 2
    
    # evaluate the fitted model
    data_fitted = twoD_Gaussian((x_full, y_full), *popt)

    # reshape to original data size
    data_fitted = data_fitted.reshape(data_masked.shape[0],
                                      data_masked.shape[1])


    return [data_fitted, popt]
