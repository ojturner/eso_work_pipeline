# function to take as input an 
# array of names and an array
# of corresponding values, tabulating
# them both and also their averages



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


# add the functions folder to the PYTHONPATH
sys.path.append('/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import flatfield_cube as f_f
import psf_blurring as psf
import twod_gaussian as g2d
import rotate_pa as rt_pa

def table_create(column_names,
                 data_values,
                 save_dir):

    """
    Def:
    Tabulates the column names and corresponding values, 
    saves the created table in the save_dir.

    Input:
            column_names - simply the titles for each of the columns
                            input as a list of strings. The galaxy names
                            must be the first column name!!!!!!
            data_values - list of lists of numerical values (apart from 
                            the galaxy names which are strings and treated
                             separately). Important that this is a list of 
                              lists because the functions will attempt to 
                               interpret it as so.
    Output:
            table - saved in the save_dir with the name shown in the first 
                    line of this function definition
    """

    # build the save file name
    file_name = save_dir + 'properties_table.txt'

    if os.path.isfile(file_name):

        os.system('rm %s' % file_name)

    # write all of these values to file

    with open(file_name, 'a') as f:

        f.write('#\t')

        for item in column_names:

            f.write('%s\t' % item)

        f.write('\n')

        # loop round each galaxy
        for gal in data_values:
            # loop round all of the parameters for each galaxy
            for prop in gal:

                f.write('%s\t' % prop)

            # new line before writing to file the next galaxy
            f.write('\n')

        # now compute the averages across each column
        # assume that the first entry is the galaxy name
        f.write('Averages:\t')
        data_without_names = []

        for gal in data_values:
            data_without_names.append(gal[1:])

        averages_array = np.nanmean(np.array(data_without_names), axis=0)

        for item in averages_array:

            f.write('%s\t' % item)

        f.close()







