# read in the axis ratios and plot the distribution 
# from the output of galfit files 

# Houses the methods required to analytically
# determine the half-light radii of a galaxy
# taking into account any neighbours which
# must be masked (in a very simple way)


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
from glob import glob
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

def read_ratios(directory):

    """
    Def:
    read in from directory and plot distribution
    """

    axis_list = []

    dir_name = directory + '*_output.fits'

    for entry in glob(dir_name):

        table = fits.open(entry)

        header = table[2].header

        # Get thhe galfit axis ratio

        axis_r_str = header['1_AR']

        axis_r = axis_r_str[:len(axis_r_str) -
                        axis_r_str[::-1].find("+") - 2]

        if axis_r[0] == '[':

            axis_r = axis_r[1:]

        axis_r = float(axis_r)

        axis_list.append(axis_r)

    print axis_list

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.hist(axis_list, bins=8)

    plt.show()

read_ratios('/disk1/turner/DATA/Victoria_galfit/n_band_outputs_flatfield/')

