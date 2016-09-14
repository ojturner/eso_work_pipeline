# Function to prepare the image which will be
# galfit to create the PSF for future galfitting



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

# add the class file to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import cube_background_subtract as b_s

from cubeClass import cubeOps


def psf_create(star_cube):

    """
    Def:
    Take the input star cube and spit out a 2D image of that
    to be fit by galfit to recover the psf for future fitting
    """

    cube = cubeOps(star_cube)

    # extract up until thermal part starts to dominate

    star_data = np.nanmedian(cube.data[200:2000, :,:], axis=0)

    # noise data for the rms map

    noise_data = np.nanmedian(cube.Table[2].data[200:2000, :,:], axis=0)

    # save both to new fits files
    star_name = '/disk1/turner/DATA/Victoria_galfit/star_image.fits'

    star_rms = '/disk1/turner/DATA/Victoria_galfit/star_image_rms.fits'

    # write out to fits files
    o_hdu = fits.PrimaryHDU(star_data)

    o_hdu.writeto(star_name,
                  clobber=True)

    n_hdu = fits.PrimaryHDU(noise_data)

    n_hdu.writeto(star_rms,
                  clobber=True)

star_cube = '/disk1/turner/DATA/new_comb_calibrated/uncalibrated_goods_p1_0.8_10_better/Science/combine_sci_reconstructed_c_stars_7656.fits'
psf_create(star_cube)