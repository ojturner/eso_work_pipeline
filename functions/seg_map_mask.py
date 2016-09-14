#DATE: 9-9-16
#Takes the segmentation map produced by sextractor, keeps the zeros the same
#and sets everything to 1 that is not equal to the number of the object you are
#attempting to fit

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
import astropy.wcs.utils as autils
from astropy.wcs import WCS
from PIL import Image

def create_galfit_mask(obj_number,
                       seg_map):

    """
    Reads the data in a segmentation map and sets everything to 1 that is
    not 0 or the number of the object which is being fit. 

    Inputs:
            obj_number - the number of the object which will translate
                         to the image regions which aren't masked during
                         the fitting process
            seg_map - the segmentation map to read in

    Output:
            seg_map_treated.fits
    """

    # read in the data from the segmentation map

    data = fits.open(seg_map)[0].data

    header = fits.open(seg_map)[0].header

    # loop through and change all entries which aren't 0 or equal to
    # the object number

    for i in range(0, len(data[0])):

        for j in range(0, len(data[1])):

            if data[i,j] == 0:

                data[i,j] = data[i,j]

            elif data[i,j] == obj_number:

                data[i,j] = 0

            else:

                data[i,j] = 1

    # save the new segmentation map

    new_seg_name = seg_map[:-5] + '_treated.fits'

    seg_hdu = fits.PrimaryHDU(header=header, data = data)

    seg_hdu.writeto(new_seg_name, clobber=True)

create_galfit_mask(1, '/home/oturner/disk1/turner/DATA/Sextractor/SSA_F160W/m38_seg.fits')

