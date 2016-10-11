# This class houses the methods which are relevant to performing manipulations
# on the reconstructed and combined data cubes.
# As such, the class will be a datacube object


# import the relevant modules

import scipy.optimize as opt
import pylab as pyplt
import numpy as np
import matplotlib.pyplot as plt
from numpy import poly1d
import scipy
import numpy.ma as ma
from copy import copy
from lmfit.models import GaussianModel
from lmfit import Model
from astropy.io import fits
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from numpy import *


class cubeOps(object):
    """
    Def:
    Class mainly for combined data cubes output from the
    kmo_sci_red recipe. Contains a series of functions and
    definitions useful for manipulating the datacubes.
    Input:
    sci_combined datacube
    """
    # Initialiser creates an instance of the cube object
    # Input must be a combined data cube with two extensions - data and noise
    def __init__(self, fileName):

        self.self = self

        self.fileName = fileName

        # Variable containing all the fits extensions
        self.Table = fits.open(fileName)

        # Variable housing the primary data cube
        self.data = self.Table[1].data

        # Primary Header
        self.primHeader = self.Table[0].header

        # set what the combined name extension is going to be. 
        # All information is in the header - can be set of permutations
        # for what this is.

        header_string = str(self.primHeader)

        illum_var = header_string.find('ILLUM_CORR')

        skytweak_var = header_string.find('sky_tweak')

        telluric_var = header_string.find('TELLURIC')

        if illum_var == -1 and skytweak_var == -1 and telluric_var == -1:

            comb_ext = '.fits'

        elif illum_var != -1 and skytweak_var != -1 and telluric_var != -1:

            comb_ext = '__telluric_illum_skytweak.fits'

        elif illum_var != -1 and skytweak_var != -1 and telluric_var == -1:

            comb_ext = '__illum_skytweak.fits'

        elif illum_var != -1 and skytweak_var == -1 and telluric_var == -1:

            comb_ext = '__illum.fits'

        elif illum_var == -1 and skytweak_var != -1 and telluric_var == -1:

            comb_ext = '__skytweak.fits'

        elif illum_var == -1 and skytweak_var == -1 and telluric_var != -1:

            comb_ext = '__telluric.fits'

        elif illum_var != -1 and skytweak_var != -1 and telluric_var != -1:

            comb_ext = '__telluric_illum.fits'

        elif illum_var == -1 and skytweak_var != -1 and telluric_var != -1:

            comb_ext = '__telluric_skytweak.fits'

        else:

            comb_ext = '.fits'

        # define the galaxy name from the full file path

        if fileName.find("/") == -1:

            self.gal_name = copy(fileName)

        # Otherwise the directory structure is included and have to
        # search for the backslash and omit up to the last one

        else:

            self.gal_name = fileName[len(fileName) - fileName[::-1].find("/"):]

        self.gal_name = str(self.gal_name)

        # Collapse over the wavelength axis to get an image
        self.imData = np.nanmedian(self.data, axis=0)

        try:

            self.total_spec = np.nanmedian(np.nanmedian
                                           (self.data[:, 4:len(self.data[0]),
                                            4:len(self.data[1])],
                                            axis=1),
                                           axis=1)
        except:

            print 'Cannot extract the total spectrum'

        # data Header
        self.dataHeader = self.Table[1].header

        # noise Header
        # self.noiseHeader = self.Table[2].header
        # Extract the wavelength calibration info
        try:

            self.start_L = self.dataHeader["CRVAL3"]

            self.dL = self.dataHeader["CDELT3"]

        except KeyError:

            print '[INFO]: This is a raw image'

        # Extract the IFU number from the data
        # Cube may not have this keyword, try statement
        # Not sure if this is good practice - conditional attribute.
        try:

            self.IFUNR = self.dataHeader["HIERARCH ESO PRO IFUNR"]

            key = 'HIERARCH ESO OCS ARM' + str(self.IFUNR) + ' NAME'

            # print key
            self.IFUName = self.dataHeader[key]

        except KeyError:

            try:

                self.noise_header = self.Table[2].header

                self.IFUNR = self.noise_header["HIERARCH ESO PRO IFUNR"]

                key = 'HIERARCH ESO OCS ARM' + str(self.IFUNR) + ' NAME'

                # print key
                self.IFUName = self.noise_header[key]

            except:

                print '[INFO]: This is not a combined Frame' \
                    + ', setting arm name...'

                try:

                    ext_name = self.dataHeader['EXTNAME']

                    num_string = ''

                    for s in ext_name:

                        if s.isdigit():

                            num_string += s

                    num_string = int(num_string)

                    self.IFUNR = copy(num_string)

                    self.IFUName = self.primHeader["HIERARCH ESO OCS ARM"
                                                   + str(self.IFUNR)
                                                   + " NAME"]

                    print '[INFO]: You have specified a reconstructed type'

                except KeyError:

                    print "[Warning]: not a datacube"

        # Set the RA and DEC positions of all the arms. These are in
        # sexagesimal format - convert to degrees for the plot
        self.raDict = {}
        self.decDict = {}
        self.combDict = {}
        self.offList = []

        for i in range(1, 25):

            # if statement to cover the possibility of operational IFUs
            # assigned to sky
            if (self.primHeader["HIERARCH ESO OCS ARM" + str(i) + " NAME"]) \
                    == ("ARM" + str(i) + "_SCI"):

                # Add the IFU to the offlist
                print 'ARM%s is operational but on sky' % str(i)

                self.offList.append(i)

            else:

                # Construct the list of combined science frames that will
                # be thrown out by the science pipeline
                try:

                    nuName = "HIERARCH ESO OCS ARM" + str(i) + " NOTUSED"
                    temp = self.primHeader[nuName]

                except KeyError:

                    combKey = "HIERARCH ESO OCS ARM" + str(i) + " ORIGARM"
                    combName = "HIERARCH ESO OCS ARM" + str(i) + " NAME"

                    self.combDict[self.primHeader[combName]] = \
                        self.primHeader[combKey]

                try:

                    raName = "HIERARCH ESO OCS ARM" + str(i) + " ALPHA"

                    DictName = 'Arm' + str(i)

                    decName = "HIERARCH ESO OCS ARM" + str(i) + " DELTA"

                    # print raName, decName
                    self.raDict[DictName] = \
                        self.raToDeg(self.primHeader[raName])

                    self.decDict[DictName] = \
                        self.decToDeg(self.primHeader[decName])

                except KeyError:
                    print DictName
                    print '[INFO]: IFU %s Not in Use,' % DictName \
                          + ' or not pointing at an object'

                    self.offList.append(i)

        # Construct the list of combined science names separately
        # This is now in order of the IFU
        self.combNames = []

        # Have to hardwire what the fits extension is for now - may come
        # up with a cleverer way of doing this in the future.


        for entry in self.combDict.keys():

            combinedName = 'SCI_COMBINED_' + entry + comb_ext
            self.combNames.append(combinedName)

        # Also construct the list of kmo_combine recipe combined names
        self.rec_combNames = []

        for entry in self.combDict.keys():

            combinedName = 'COMBINE_SCI_RECONSTRUCTED_' + entry + '.fits'
            self.rec_combNames.append(combinedName)

        self.offList = np.array(self.offList)
        self.offList = self.offList - 1
        self.raArray = self.raDict.values()
        self.decArray = self.decDict.values()
        self.IFUArms = self.raDict.keys()
        self.xDit = self.primHeader['HIERARCH ESO OCS TARG DITHA']
        self.yDit = self.primHeader['HIERARCH ESO OCS TARG DITHD']

        # Find the pixel scale if this is a combined cube
        try:

            self.pix_scale = \
                self.primHeader['HIERARCH ESO PRO REC1 PARAM8 VALUE']

        except KeyError:

            print '[INFO]: Could not set pixel scale - not a datacube'

            self.pix_scale = 0

        # Create the wavelength array if this is a combined data type
        try:

            self.wave_array = self.start_L \
                + np.arange(0, 2048 * (self.dL), self.dL)
        except:

            print '[INFO]: cannot set wavelength array'

        # Can define all kinds of statistics
        # from the data common to the methods
        # Find the brightest median pixel in the array
        self.med_array = np.nanmedian(self.data, axis=0)
        self.num = np.nanargmax(self.med_array)
        self.ind1 = self.num / len(self.med_array)
        self.ind2 = self.num - (len(self.med_array) * self.ind1)
        self.filter = self.primHeader['HIERARCH ESO INS FILT1 ID']

    def raToDeg(self, ra):
        """
        Def:
        Helper function - convert sexagesimal RA to degrees.
        Needs to check number digits before the decimal point,
        because the fits files doesn't always give 6

        Input: Sexagesimal RA (HHMMSS.SS)
        Output: Ra in degrees

        """
        ra = str(ra)

        # Figure out how many characters before the decimal point
        i = 0

        for char in ra:

            if char == '.':

                break

            else:

                i += 1

        # Conditionally convert, depending on whether i is 4 or 6
        # Also conditionally execute depending on number decimal places
        if i == 6:

            hours = int(ra[0:2])
            mins = int(ra[2:4])
            secs = float(ra[4:])
            raDeg = (hours * 15) + (mins * 0.25) + (secs * 1.0 / 240)

        elif i == 4:

            mins = int(ra[0:2])
            secs = float(ra[2:])
            raDeg = (mins * 0.25) + (secs * 1.0 / 240)

        else:

            secs = float(ra)
            raDeg = (secs * 1.0 / 240)

        return raDeg

    def decToDeg(self, dec):
        """
        Def:
        Helper function - convert sexagesimal dec to degrees.
        Needs to check number digits before the decimal point,
        because the fits files doesn't always give 6

        Input: Sexagesimal dec (DDMMSS.SS)
        Output: Dec in degrees

        """
        dec = str(dec)
        # Figure out how many characters before the decimal point
        i = 0
        if dec[0] == '-':

            for char in dec:

                if char == '.':

                    break

                else:

                    i += 1

            # Conditionally convert, depending on whether i is 4 or 6
            # Also conditionally execute depending on number decimal places
            if i == 7:

                deg = float(dec[1:3])
                mins = float(dec[3:5])
                secs = float(dec[5:])

                # Careful whether deg is negative or not
                # Becoming more positive if > 0
                decDeg = (deg * -1) - (mins * 1.0 / 60) - (secs * 1.0 / 3600)

            elif i == 5:

                mins = float(dec[1:3])
                secs = float(dec[3:])
                decDeg = (mins * -1.0 / 60) - (secs * 1.0 / 3600)

            else:

                secs = float(dec)
                decDeg = (secs * 1.0 / 3600)

            return decDeg

        else:

            for char in dec:
                if char == '.':
                    break
                else:
                    i += 1

            # Conditionally convert, depending on whether i is 4 or 6
            # Also conditionally execute depending on number decimal places
            # print i
            if i == 6:
                deg = float(dec[0:2])
                mins = float(dec[2:4])
                secs = float(dec[4:])

                # Careful whether deg is negative or not
                # Becoming more positive if > 0
                decDeg = deg + (mins * 1.0 / 60) + (secs * 1.0 / 3600)

            elif i == 4:

                mins = float(dec[0:2])
                secs = float(dec[2:])
                decDeg = (mins * 1.0 / 60) + (secs * 1.0 / 3600)

            else:

                secs = float(dec)
                decDeg = (secs * 1.0 / 3600)

            return decDeg

    def ifuCoordsPlot(self):
        """
        Def:
        Plots the already recorded IFU Positions on the sky

        Inputs: None
        Output: Matplotlib plot of coordinates

        """

        plt.scatter(self.raArray, self.decArray)
        plt.show()
        plt.close('all')

    def specPlot(self,
                 gridSize):
        """
        Def:
        Takes a data cube and creates a median stacked 1-D
        spectrum around the brightest pixel, with the size
        of the median stack defined by gridSize

        Input:
        gridSize - must be a positive int less than the number of
        spatial pixels in the cube.

        Output:
        matplotlib plot of the 1D stacked spectrum


        """
        # If choosing to construct the 1D spectrum from a single pixel:
        print '[INFO]: The Brightest Pixel is at: (%s, %s)' \
              % (self.ind1, self.ind2)

        print self.IFUName

        print self.IFUNR

        if gridSize == 1:

            flux_array = self.data[:, self.ind1, self.ind2]

        else:

            lst = []

            for i in range(self.ind1 - (gridSize / 2),
                           self.ind1 + (gridSize / 2)):

                for j in range(self.ind2 - (gridSize / 2),
                               self.ind2 + (gridSize / 2)):

                    lst.append(self.data[:, i, j])

            flux_array = np.nanmedian(lst, axis=0)

        # Now make very basic plot at the moment
        fig, ax = plt.subplots(1, 1, figsize=(18.0, 12.0))
        ax.plot(self.wave_array, flux_array)
        ax.set_title('Flux vs. Wavelength')
        ax.set_xlabel('Wavelength ($\mu m$)')
        ax.set_ylim(0, 100)

        saveName = (self.fileName)[:-5] + '.png'
        fig.savefig(saveName)
        # plt.show()
        plt.close('all')
        return flux_array

    def centralSpec(self):
        """
        Def:
        Takes a data cube and creates a median stacked 1-D
        spectrum in a 3x3 grid around the central pixel

        Output:
        matplotlib plot of the 1D stacked spectrum


        """
        lst = []

        for i in range((len(self.data[0]) / 2) - 1,
                       (len(self.data[0]) / 2) + 1):

            for j in range((len(self.data[0]) / 2) - 1,
                           (len(self.data[0]) / 2) + 1):

                lst.append(self.data[:, i, j])

        flux_array = np.nanmedian(lst, axis=0)

        return flux_array

    def singlePixelExtract(self,
                           centre_x,
                           centre_y):
        """
        Def:
        Extracts a 1-D spectrum at the central x and y locations provided
        Input - centre_x: the central location of the
                    galaxy on the 2D image in the x direction
                centre_y: the central location of the
                    galaxy on the 2D image in the y direction
        Output - FluxArray: Extracted 1D flux spectrum
                    from the object at the chosen location
        """

        # Already have Data defined - want to collapse this
        # down to a 1D array at the chosen x-y location
        return self.data[:, centre_y, centre_x]

    def specPlot2D(self,
                   orientation):
        """
        Def:
        Takes a data cube and creates a median stacked 2-D
        spectrum across either the horizontal row or vertical column

        Input:
        orientation - either 'vertical' or 'horizontal' (default down)

        Output:
        2D array specifying the 2D spectrum


        """

        # Check the orientation input to see what has been specified#
        try:

            # If 'up' median stack across the rows
            if orientation == 'vertical':

                plot_vec = np.nanmedian(self.data, axis=1)

            elif orientation == 'horizontal':

                plot_vec = np.nanmedian(self.data, axis=2)

            else:

                raise ValueError('Choose either vertical'
                                 + ' or horizontal for the orientation')

        except ValueError:

            print 'check input for Orientation'

            raise

        # Now have the plot vector, plot it.
        plt.imshow(plot_vec)
        plt.savefig('test.png')
        plt.close('all')

    def optimalSpec(self):

        """
        Def:
        Optimally extract the spectrum of the object from the whole image.
        Use the PSF of the object to get the weighting for the extraction.
        Do I just sum over the axis?
        Input: None - everything already defined
        """
        # Fit a gaussian to the fully combined science cube
        # to determine the optimal extraction profile
        print '[INFO]: Fitting the optimal' \
            + ' spectrum for object: %s' % self.IFUName

        params, psfMask, fwhm, offList = self.psfMask()

        # Multiply the cube data by the psfMask
        modCube = psfMask * self.data

        # Recover the width of the gaussian
        width = fwhm / 2.3548
        width = int(np.round(width))

        # Recover the central value
        x = params[2]
        y = params[1]

        # Set the upper and lower limits for optimal spectrum extraction
        x_upper = int(x + (1.5 * width))

        if x_upper > len(self.data[0]):

            x_upper = len(self.data[0])

        x_lower = int(x - (1.5 * width))

        if x_lower < 0:

            x_lower = 0

        y_upper = int(y + (1.5 * width))

        if y_upper > len(self.data[0]):

            y_upper = len(self.data[0])

        y_lower = int(y - (1.5 * width))

        if y_lower < 0:

            y_lower = 0

        print x_lower, x_upper, y_lower, y_upper

        # Set all values greater than 2sigma from the centre = 0
        # Don't want to include these in the final spectrum
        modCube[:, 0:y_lower, :] = 0
        modCube[:, y_upper:len(self.data[0]), :] = 0
        modCube[:, :, 0:x_lower] = 0
        modCube[:, :, x_upper: len(self.data[0])] = 0

        # Sum over each spatial dimension to get the spectrum
        first_sum = np.nansum(modCube, axis=1)
        spectrum = np.nansum(first_sum, axis=1)
        return spectrum

    def optimalSpecFromProfile(self,
                               profile,
                               fwhm,
                               centre_x,
                               centre_y):

        """
        Def:
        Optimally extract the spectrum of the object from the whole image.
        Use the PSF of the object to get the weighting for the extraction.
        Input: Profile - a specified 2D normalised profile for extraction
               fwhm - the fwhm of the tracked star
        """
        try:
            print '[INFO]: Fitting the optimal ' \
                + 'spectrum for object: %s' % self.IFUName

        except AttributeError:

            print '[INFO]: Fitting the optimal spectrum'

        # Multiply the cube data by the psfMask
        modCube = profile * self.data

        # Recover the width of the gaussian
        width = fwhm / 2.3548
        # Recover the central value
        x = copy(centre_x)
        y = copy(centre_y)
        print '[INFO]: The central values of the Gaussian are: %s %s' % (x, y)
        print '[INFO]: And the width is: %s' % width

        # Set the upper and lower limits for optimal spectrum extraction
        x_upper = int(np.round((x + (2.0 * width))))

        if x_upper > len(self.data[0]):

            x_upper = len(self.data[0])

        x_lower = int(np.round((x - (2.0 * width))))

        if x_lower < 0:

            x_lower = 0

        y_upper = int(np.round((y + (2.0 * width))))

        if y_upper > len(self.data[0]):

            y_upper = len(self.data[0])

        y_lower = int(np.round((y - (2.0 * width))))

        if y_lower < 0:

            y_lower = 0

        print x_lower, x_upper, y_lower, y_upper

        # Set all values greater than 2sigma from the centre = 0
        # Don't want to include these in the final spectrum
        modCube[:, 0:y_lower, :] = 0
        modCube[:, y_upper:len(self.data[0]), :] = 0
        modCube[:, :, 0:x_lower] = 0
        modCube[:, :, x_upper: len(self.data[0])] = 0

        imModCube = copy(modCube)
        imModCube = np.nanmedian(modCube, axis=0)

        # Check to see that the gaussian and shifted profile align
        colFig, colAx = plt.subplots(1, 1, figsize=(12.0, 12.0))
        colCax = colAx.imshow(imModCube, interpolation='bicubic')
        colFig.colorbar(colCax)

        # plt.show()
        plt.close('all')

        # Sum over each spatial dimension to get the spectrum
        first_sum = np.nansum(modCube, axis=1)
        spectrum = np.nansum(first_sum, axis=1)

        # for the purposes of the velocity map it will be useful
        # to extract the gaussian parameters from this 1D spectrum
        # first need to check if we're K or HK

        return spectrum

    # Create a gaussian function for use with lmfit
    def gaussian(self,
                 x1,
                 x2,
                 pedastal,
                 height,
                 center_x,
                 center_y,
                 width_x,
                 width_y):
        """
        Def: Return a two dimensional gaussian function
        """

        # make sure we have floating point values
        width_x = float(width_x)
        width_y = float(width_y)

        # Specify the gaussian function here
        func = pedastal + height * exp(
            -(((center_x - x1) / width_x)
                ** 2 + ((center_y - x2) / width_y) ** 2) / 2)
        return func

    # Create a gaussian function for use with the integral
    def gaussianLam(self,
                    pedastal,
                    height,
                    center_x,
                    center_y,
                    width_x,
                    width_y):

        """
        Def: Returns a gaussian function with the given parameters
        """
        # make sure we have floating point values
        width_x = float(width_x)
        width_y = float(width_y)

        return lambda x, y: pedastal + height * exp(
            -(((center_x - x) / width_x)
                ** 2 + ((center_y - y) / width_y) ** 2) / 2)

    def gauss2dMod(self):

        mod = Model(self.gaussian,
                    independent_vars=['x1', 'x2'],
                    param_names=['pedastal',
                                 'height',
                                 'center_x',
                                 'center_y',
                                 'width_x',
                                 'width_y'],
                    missing='drop')

        # print mod.independent_vars
        # print mod.param_names

        return mod

    def moments_better(self,
                       data,
                       circle=0,
                       rotate=1,
                       vheight=1,
                       estimator=np.ma.median,
                       **kwargs):

        """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
        the gaussian parameters of a 2D distribution by calculating its
        moments.  Depending on the input parameters, will only output 
        a subset of the above.

        If using masked arrays, pass estimator=np.ma.median
        """

        total = np.abs(data).sum()

        Y, X = np.indices(data.shape) # python convention: reverse x,y np.indices
        y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
        x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
        col = data[int(y),:]

        # FIRST moment, not second!
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())
        row = data[:, int(x)]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())
        width = ( width_x + width_y ) / 2.
        height = estimator(data.ravel())
        amplitude = data.max()-height
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

    def twoD_Gaussian(self,
                      (x, y),
                      offset,
                      amplitude,
                      xo,
                      yo,
                      sigma_x,
                      sigma_y,
                      theta):

        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))

        return g.ravel()


    def moments(self,
                data):

        """
        Def: Returns (pedastal, height, center_x, center_y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments
        """

        pedastal = np.nanmedian(data)

        height = np.nanmax(data)

        # data[np.isnan(data)] = 0

        # data[data < 0] = 0

        total = np.nansum(data)

        # print 'The sum over the data is: %s' % total
        X, Y = indices(data.shape)

        # print 'The Indices are: %s, %s' % (X, Y)
        center_x = np.nansum((X * data)) / total
        center_y = np.nansum((Y * data)) / total
        width_x = 2.0
        width_y = 2.0

        print [height, center_x, center_y, width_x, width_y, pedastal]
        return [height, center_x, center_y, width_x, width_y, pedastal]

    def fitgaussian(self,
                    data):

        """
        Def: Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit
        """

        # At the moment will assume that the data
        # is imagedata which needs flattened
        # first step is to normalise data by the sum across the spaxels
        # this is to help the gaussian fitting for the flux 
        # calibrated data. also clip the edges for a better fit.

        data = data[3:-3, 3:-3] / np.nanmax(data)

        pars = self.moments(data)

        flat_data = np.ndarray.flatten(data)

        # print 'This is the flattened data: %s' % flat_data

        # Get the gaussian model
        mod = self.gauss2dMod()

        # Set the parameter hints from the initialPars method
        mod.set_param_hint('height', value=pars[0], min=0)

        mod.set_param_hint('center_x', value=pars[1], min=0, max=data.shape[0])

        mod.set_param_hint('center_y', value=pars[2], min=0, max=data.shape[0])

        mod.set_param_hint('width_x', value=pars[3], min=0)

        mod.set_param_hint('width_y', value=pars[4], min=0)

        mod.set_param_hint('pedastal', value=pars[5])

        # Initialise a parameters object to use in the fit
        fit_pars = mod.make_params()

        # Guess isn't implemented for this model
        # Need to pass independent variables for the fit. these come from
        # flattening the indices of data.shape
        x1 = np.ndarray.flatten(indices(data.shape)[0])

        # print 'The first independent variable: %s %s' % (x1, type(x1))
        x2 = np.ndarray.flatten(indices(data.shape)[1])

        # print 'The second independent variable: %s' % x2
        mod_fit = mod.fit(flat_data, x1=x1, x2=x2, params=fit_pars)

        # need to change the center_x and center_y params
        # because of the data clipping

        new_x = mod_fit.best_values['center_x'] + 3

        new_y = mod_fit.best_values['center_y'] + 3

        mod_fit.best_values['center_x'] = new_x

        mod_fit.best_values['center_y'] = new_y

        return mod_fit, x1, x2

    def psfMask(self):

        """Returns (FWHM, psfMask) which are the FWHM of the 2D gaussian
        fit to the collapsed object image and the mask of values found
        after integrating the gaussian function over all the pixels and
        normalising by this value.
        """
        # Find the FWHM and the masking profile of a given datacube

        # Step 1 - perform least squares minimisation to find the parameters

        # instead of using the full self.imData, only use the chunk in the
        # middle of the wavelength range

        if self.filter == 'K':

            lower = 268

            upper = 1336

        elif self.filter == 'H':

            lower = 349

            upper = 1739

        elif self.filter == 'HK':

            lower = 87

            upper = 734

        else:

            lower = 200

            upper = 1800

        clip_value = 3

        # range from which to get the image data now defined
        star_data = np.nanmedian(self.data[lower:upper], axis=0)

        # this doesn't work so smoothly for the tiny numbers. 
        # if the median star_data value is < 10-10, divide through 
        # by 1E-18

        if np.nanmedian(star_data) < 1E-10:

            star_data = star_data / 1E-18


        # mask out the nan values using np.ma
        data_masked = np.ma.masked_invalid(star_data)

        # indices of the full array
        y_full, x_full = np.indices(data_masked.shape)

        # now bring in the sides of data_masked
        data_masked_cut = data_masked[clip_value:-clip_value,
                                      clip_value:-clip_value]

        # create the grid over which to evaluate the gaussian
        y, x = np.indices(data_masked_cut.shape)

        # find the moments of the data and use these as the initial
        # guesses for the gaussian

        list_of_moments = self.moments_better(data_masked_cut)

        # very important - have to set the nan values equal to the
        # evaluated height parameter

        data_masked_cut[isnan(data_masked_cut)] = list_of_moments[0]

        # fit the model
        popt, pcov = opt.curve_fit(self.twoD_Gaussian,
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
        popt[2] = popt[2] + clip_value
        popt[3] = popt[3] + clip_value

        # evaluate the fitted model

        data_fitted = self.twoD_Gaussian((x_full, y_full), *popt)

        # reshape to original data size
        data_fitted = data_fitted.reshape(data_masked.shape[0],
                                          data_masked.shape[1])

        # make dictionary with same parameters as before

        params = {'pedastal' : popt[0],
                  'height' : popt[1],
                  'center_x' : popt[3],
                  'center_y' : popt[2],
                  'width_x' : popt[5],
                  'width_y' : popt[4],
                  'rotation' : popt[6]}

        # divide by the evaluated amplitude to normalise
        data_fitted = data_fitted / params['height']


        # get the seeing
        sigma = 0.5 * (params['width_x'] + params['width_y'])
        FWHM = 2.3548 * sigma

        try:

            print '[INFO]: The FWHM of object %s is: %s' % (self.IFUName, FWHM)

        except AttributeError:

            print '[INFO]: The FWHM is: %s' % FWHM

        # Plot the image and the fit
        colFig, colAx = plt.subplots(1, 1, figsize=(14.0, 14.0))

        colCax = colAx.imshow(self.imData, interpolation='bicubic')

        colAx.contour(x_full,
                      y_full,
                      data_fitted,
                      8,
                      colors='w')

        colFig.colorbar(colCax)

        saveName = (self.fileName)[:-5] + '_gauss.png'

        colFig.savefig(saveName)

        # plt.show()
        plt.close('all')

        # return the FWHM and the masked profile
        return params, data_fitted, FWHM, self.offList

    def plot_HK_sn_map(self,
                       redshift,
                       savefig=False):
        """
        Def:
        Check the signal to noise of the emission lines over the face of a cube
        with known redshift
        Input: redshift - redshift of the galaxy in the cube
               savefig - whether or not to save the figures
        """

        fig, axes = plt.subplots(figsize=(14, 4), nrows=1, ncols=3)
        fig.subplots_adjust(right=0.83)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        # create a sn dictionary to house the line sn maps
        sn_dict = {}

        for line, ax in zip(['[OII]', 'Hb', '[OIII]5007'], axes.flatten()):

            ax.minorticks_on()

            if line == '[OIII]5007':
                oiii5007_wl = 0.500824 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oiii5007_wl))
            elif line == 'Hb':
                hb_wl = 0.486268 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - hb_wl))
            elif line == '[OII]':
                oii_wl = 0.3729875 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oii_wl))

            # the shape of the data is (spectrum, xpixel, ypixel)
            # loop through each x and y pixel and get the OIII5007 S/N
            xpixs = data.shape[1]
            ypixs = data.shape[2]

            sn_array = np.empty(shape=(xpixs, ypixs))

            for i, xpix in enumerate(np.arange(0, xpixs, 1)):

                for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                    spaxel_spec = data[:, i, j]
                    spaxel_noise = noise[:, i, j]

                    line_counts = np.median(spaxel_spec[line_idx - 3:
                                                        line_idx + 3])

                    line_noise = np.median(spaxel_noise[line_idx - 3:
                                                        line_idx + 3])

                    line_sn = line_counts / line_noise

                    if np.isnan(line_sn):
                        sn_array[i, j] = -99.
                    else:
                        sn_array[i, j] = line_sn

            # print max(sn_array.flatten())
            # add the result to the sn_dict
            sn_dict[line] = sn_array

            im = ax.imshow(sn_array, aspect='auto', vmin=0.,
                           vmax=3.,
                           cmap=plt.get_cmap('hot'))

            ax.set_title('%s' % line)

        fig.colorbar(im, cax=cbar_ax)

        # plt.tight_layout()
        # plt.show()

        if savefig:
            fig.savefig('%s_sn_map.pdf' % self.fileName[:-5])
        # return the dictionary containing the noise values
        return sn_dict

    def plot_K_sn_map(self,
                      redshift,
                      savefig=False):

        fig, axes = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
        fig.subplots_adjust(right=0.83)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])
        # Create a dictionary to house the sn_arrays
        sn_dict = {}

        for line, ax in zip(['Hb', '[OIII]5007'], axes.flatten()):

            ax.minorticks_on()

            if line == '[OIII]5007':
                oiii5007_wl = 0.500824 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oiii5007_wl))
            elif line == 'Hb':
                hb_wl = 0.486268 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - hb_wl))

            # the shape of the data is (spectrum, xpixel, ypixel)
            # loop through each x and y pixel and get the OIII5007 S/N
            xpixs = data.shape[1]
            ypixs = data.shape[2]

            sn_array = np.empty(shape=(xpixs, ypixs))
            signal_array = np.empty(shape=(xpixs, ypixs))
            noise_array = np.empty(shape=(xpixs, ypixs))

            for i, xpix in enumerate(np.arange(0, xpixs, 1)):

                for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                    spaxel_spec = data[:, i, j]
                    spaxel_noise = noise[:, i, j]

                    line_counts = np.median(spaxel_spec[line_idx - 3:
                                                        line_idx + 3])
                    if np.isnan(line_counts):
                        signal_array[i, j] = 0
                    else:
                        signal_array[i, j] = line_counts

                    line_noise = np.median(spaxel_noise[line_idx - 3:
                                                        line_idx + 3])

                    if np.isnan(line_noise):
                        noise_array[i, j] = 0
                    else:
                        noise_array[i, j] = line_noise

                    line_sn = line_counts / line_noise

                    if np.isnan(line_sn):
                        sn_array[i, j] = -99.
                    else:
                        sn_array[i, j] = line_sn

            # print max(sn_array.flatten())
            # add the result to the sn dictionary
            sn_dict[line] = sn_array

            im = ax.imshow(sn_array, aspect='auto', vmin=0.,
                           vmax=3.,
                           cmap=plt.get_cmap('hot'))

            ax.set_title('%s' % line)

        fig.colorbar(im, cax=cbar_ax)

        # plt.tight_layout()
        # noise plt.show()

        if savefig:
            fig.savefig('%s_sn_map.pdf' % self.fileName[:-5])

        hdu = fits.PrimaryHDU(noise_array)
        hdu.writeto('%s_noise_map.fits' % self.fileName[:-5], clobber=True)
        hdu = fits.PrimaryHDU(signal_array)
        hdu.writeto('%s_signal_map.fits' % self.fileName[:-5], clobber=True)

        return sn_dict

    def plot_HK_image(self,
                      redshift,
                      savefig=False):

        """
        Def:
        Check the signal to noise of the emission lines over the face of a cube
        with known redshift
        Input: redshift - redshift of the galaxy in the cube
               savefig - whether or not to save the figures
        """

        fig, axes = plt.subplots(figsize=(14, 4), nrows=1, ncols=3)
        fig.subplots_adjust(right=0.83)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        for line, ax in zip(['[OII]', 'Hb', '[OIII]5007'], axes.flatten()):

            ax.minorticks_on()

            # the shape of the data is (spectrum, xpixel, ypixel)
            # loop through each x and y pixel and get the OIII5007 S/N
            xpixs = data.shape[1]
            ypixs = data.shape[2]

            if line == '[OIII]5007':
                oiii5007_wl = 0.500824 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oiii5007_wl))
                met_array_OIII = np.empty(shape=(xpixs, ypixs))

            elif line == 'Hb':
                hb_wl = 0.486268 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - hb_wl))
                met_array_Hb = np.empty(shape=(xpixs, ypixs))

            elif line == '[OII]':
                oii_wl = 0.3729875 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oii_wl))
                met_array_OII = np.empty(shape=(xpixs, ypixs))

            sn_array = np.empty(shape=(xpixs, ypixs))

            for i, xpix in enumerate(np.arange(0, xpixs, 1)):

                for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                    spaxel_spec = data[:, i, j]
                    spaxel_noise = noise[:, i, j]

                    line_counts = np.median(spaxel_spec[line_idx - 3:
                                                        line_idx + 3])

                    line_noise = np.median(spaxel_noise[line_idx - 3:
                                                        line_idx + 3])

                    line_sn = line_counts / line_noise

                    if np.isnan(line_sn):
                        sn_array[i, j] = -99
                    else:
                        sn_array[i, j] = line_counts

                    if line == '[OIII]5007':
                        if line_sn < 0.8:
                            met_array_OIII[i, j] = np.nan
                        else:
                            met_array_OIII[i, j] = line_counts

                    if line == 'Hb':
                        if line_sn < 0.8:
                            met_array_Hb[i, j] = np.nan
                        else:
                            met_array_Hb[i, j] = line_counts

                    if line == '[OII]':
                        if line_sn < 0.8:
                            met_array_OII[i, j] = np.nan
                        else:
                            met_array_OII[i, j] = line_counts

            # print max(sn_array.flatten())

            im = ax.imshow(sn_array, aspect='auto', vmin=0.,
                           vmax=3.,
                           cmap=plt.get_cmap('hot'))

            ax.set_title('%s' % line)

        fig.colorbar(im, cax=cbar_ax)

        # plt.tight_layout()
        # plt.show()
        if savefig:
            fig.savefig('%s_images.pdf' % self.fileName[:-5])
        plt.close('all')

        # now should also have the Hb and OIII metallicity maps
        # divide the two and plot the result
        overall_met = met_array_OIII / met_array_Hb
        overall_met_OII = met_array_OIII / met_array_OII

        # now for each of these convert to metallicity using the Maiolino
        # relations. The problem here is with which root of the polynomial
        # to take. Different roots should be applicable in the high and low
        # metallicity intervals
        # First the Hb ratio, set up a new array to house the results

        x_shape = overall_met.shape[0]
        y_shape = overall_met.shape[1]

        Hb_met_array = np.empty(shape=(x_shape, y_shape))

        # initialise the coefficients, given in Maiolino 2008
        c_0_Hb = 0.1549
        c_1_Hb = -1.5031
        c_2_Hb = -0.9790
        c_3_Hb = -0.0297

        for i, xpix in enumerate(np.arange(0, x_shape, 1)):

            for j, ypix in enumerate(np.arange(0, y_shape, 1)):
                # print 'This is the number: %s' % overall_met[i, j]

                # if the number is nan, leave it as nan

                if np.isnan(overall_met[i, j]) \
                   or np.isinf(overall_met[i, j]) \
                   or (overall_met[i, j]) < 0:

                    Hb_met_array[i, j] = np.nan

                # else subtract the log10(number) from
                # c_0_Hb and set up the polynomial from poly1D

                else:

                    c_0_Hb_new = c_0_Hb - np.log10(overall_met[i, j])

                    p = poly1d([c_3_Hb, c_2_Hb, c_1_Hb, c_0_Hb_new])
                    # print p.r
                    # the roots of the polynomial are given in units
                    # of metallicity relative to solar. add 8.69
                    # met_value = p.r[0] + 8.69
                    # if the root has an imaginary component, just take
                    # the real part
                    Hb_met_array[i, j] = p.r[2].real + 8.69

        # Next the OII ratio, set up a new array to house the results

        x_shape = overall_met_OII.shape[0]
        y_shape = overall_met_OII.shape[1]

        OII_met_array = np.empty(shape=(x_shape, y_shape))

        # initialise the coefficients, given in Maiolino 2008
        c_0_OII = -0.2839
        c_1_OII = -1.3881
        c_2_OII = -0.3172

        for i, xpix in enumerate(np.arange(0, x_shape, 1)):

            for j, ypix in enumerate(np.arange(0, y_shape, 1)):

                # if the number is nan, leave it as nan
                if np.isnan(overall_met_OII[i, j]) \
                   or np.isinf(overall_met_OII[i, j]) \
                   or (overall_met_OII[i, j]) < 0:

                    OII_met_array[i, j] = np.nan

                # else subtract the number from
                # c_0_OII and set up the polynomial from poly1D

                else:
                    # print 'This is the number: %s' % overall_met_OII[i, j]
                    c_0_OII_new = c_0_OII - np.log10(overall_met_OII[i, j])

                    p = poly1d([c_2_OII, c_1_OII, c_0_OII_new])
                    # print p.r
                    # the roots of the polynomial are given in units
                    # of metallicity relative to solar. add 8.69
                    # met_value = p.r[0] + 8.69
                    # if the root has an imaginary component, just take
                    # the real part
                    if np.isreal(p.r[1]):
                        OII_met_array[i, j] = p.r[1] + 8.69
                    else:
                        OII_met_array[i, j] = -100

        fig, ax = plt.subplots(1, figsize=(10, 10))

        im = ax.imshow(Hb_met_array, aspect='auto',
                       vmin=7.5, vmax=9.0, cmap=plt.get_cmap('jet'))

        ax.set_title('[OIII] / Hb')

        fig.colorbar(im)

        # plt.show()
        if savefig:
            fig.savefig('%s_OIII_Hb.pdf' % self.fileName[:-5])
        plt.close('all')

        fig, ax = plt.subplots(1, figsize=(10, 10))

        im = ax.imshow(OII_met_array, aspect='auto',
                       vmin=7.5, vmax=9.0, cmap=plt.get_cmap('jet'))

        ax.set_title('[OIII] / [OII]')

        fig.colorbar(im)
        # plt.show()
        if savefig:
            fig.savefig('%s_OIII_OII.pdf' % self.fileName[:-5])
        plt.close('all')
        return Hb_met_array, OII_met_array

    def plot_K_image(self,
                     redshift,
                     savefig=False):

        fig, axes = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
        fig.subplots_adjust(right=0.83)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        for line, ax in zip(['Hb', '[OIII]5007'], axes.flatten()):

            ax.minorticks_on()

            # the shape of the data is (spectrum, xpixel, ypixel)
            # loop through each x and y pixel and get the OIII5007 S/N
            xpixs = data.shape[1]
            ypixs = data.shape[2]

            if line == '[OIII]5007':
                oiii5007_wl = 0.500824 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - oiii5007_wl))
                met_array_OIII = np.empty(shape=(xpixs, ypixs))
            elif line == 'Hb':
                hb_wl = 0.486268 * (1. + redshift)
                line_idx = np.argmin(np.abs(wl - hb_wl))
                met_array_Hb = np.empty(shape=(xpixs, ypixs))

            sn_array = np.empty(shape=(xpixs, ypixs))

            for i, xpix in enumerate(np.arange(0, xpixs, 1)):

                for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                    spaxel_spec = data[:, i, j]
                    spaxel_noise = noise[:, i, j]

                    line_counts = np.median(spaxel_spec[line_idx - 3:
                                                        line_idx + 3])

                    line_noise = np.median(spaxel_noise[line_idx - 3:
                                                        line_idx + 3])

                    line_sn = line_counts / line_noise

                    if np.isnan(line_sn):
                        sn_array[i, j] = -99
                    else:
                        sn_array[i, j] = line_counts

                    if line == '[OIII]5007':
                        if line_sn < 1.0:
                            met_array_OIII[i, j] = np.nan
                        else:
                            met_array_OIII[i, j] = line_counts

                    if line == 'Hb':
                        if line_sn < 1.0:
                            met_array_Hb[i, j] = np.nan
                        else:
                            met_array_Hb[i, j] = line_counts

            # print max(sn_array.flatten())

            im = ax.imshow(sn_array, aspect='auto', vmin=0.,
                           vmax=3.,
                           cmap=plt.get_cmap('hot'))

            ax.set_title('%s' % line)

        fig.colorbar(im, cax=cbar_ax)

        # plt.tight_layout()
        # plt.show()
        if savefig:
            fig.savefig('%s_images.pdf' % self.fileName[:-5])
        plt.close('all')

        # now should also have the Hb and OIII metallicity maps
        # divide the two and plot the result
        overall_met = met_array_OIII / met_array_Hb

        # now for each of these convert to metallicity using the Maiolino
        # relations. The problem here is with which root of the polynomial
        # to take. Different roots should be applicable in the high and low
        # metallicity intervals
        # First the Hb ratio, set up a new array to house the results

        x_shape = overall_met.shape[0]
        y_shape = overall_met.shape[1]

        Hb_met_array = np.empty(shape=(x_shape, y_shape))

        # initialise the coefficients, given in Maiolino 2008
        c_0_Hb = 0.1549
        c_1_Hb = -1.5031
        c_2_Hb = -0.9790
        c_3_Hb = -0.0297

        for i, xpix in enumerate(np.arange(0, x_shape, 1)):

            for j, ypix in enumerate(np.arange(0, y_shape, 1)):
                # print 'This is the number: %s' % overall_met[i, j]

                # if the number is nan, leave it as nan

                if np.isnan(overall_met[i, j]) \
                   or np.isinf(overall_met[i, j]) \
                   or (overall_met[i, j]) < 0:

                    Hb_met_array[i, j] = np.nan

                # else subtract the log10(number) from
                # c_0_Hb and set up the polynomial from poly1D

                else:

                    c_0_Hb_new = c_0_Hb - np.log10(overall_met[i, j])

                    p = poly1d([c_3_Hb, c_2_Hb, c_1_Hb, c_0_Hb_new])
                    # print p.r
                    # the roots of the polynomial are given in units
                    # of metallicity relative to solar. add 8.69
                    # met_value = p.r[0] + 8.69
                    # if the root has an imaginary component, just take
                    # the real part
                    Hb_met_array[i, j] = p.r[2].real + 8.69

        fig, ax = plt.subplots(1, figsize=(14, 14))

        im = ax.imshow(Hb_met_array, aspect='auto',
                       vmin=7.5, vmax=9.0, cmap=plt.get_cmap('jet'))

        ax.set_title('log([OIII] / Hb)')

        fig.colorbar(im)
        # plt.show()
        if savefig:
            fig.savefig('%s_OIII_Hb.pdf' % self.fileName[:-5])
        plt.close('all')
        return Hb_met_array

    def spaxel_binning(self,
                       data,
                       xbin,
                       ybin,
                       interp='median'):

        """
        Def: bins spaxels in xbin and ybin shaped chunks. Sticking with the
        convention that xbin will always refer to the first index.
        Input: xbin - number of spaxels to go into 1 along x direction
               ybin - number of spaxels to go into 1 along y direction
        """
        # the data is a 3D cube - need to preserve this
        # use a for loop with a step size

        # initially set the dimensions of the data cube
        xlength = data.shape[1]
        ylength = data.shape[2]

        print 'The original cube dimensions are: (%s, %s)' % (xlength, ylength)

        # calculate what the shape of the final array will be
        # this is tricky if the bin sizes do not match the shape of
        # the array. In this case take the modulus result and use that
        # as the final bin width (more often than not this will be 1)
        # for the shape of the final array this means that it is given
        # by the initial shape / binsize + 1 (if % != 0)

        if xlength % xbin == 0:
            new_xlength = xlength / xbin

        else:
            # account for the additional bin at the edge
            new_xlength = (xlength / xbin) + 1

        if ylength % ybin == 0:
            new_ylength = ylength / ybin

        else:
            # account for the additional bin at the edge
            new_ylength = (ylength / ybin) + 1

        # create the new array
        new_data = np.empty(shape=(data.shape[0], new_xlength, new_ylength))

        # loop round and create the new spaxels
        # during each loop component need to check
        # if the indices are reaching the original data size
        # and if so create the final bin using modulus
        # then save each 1D array at the appropriate location
        # in the new_data cube

        # set counters to record the position to store the new spaxel
        # in the new_data array
        x_cube_counter = 0

        for x in range(0, xlength, xbin):
            y_cube_counter = 0
            # check if the xlength has been reached or exceeded
            if x + xbin >= xlength:

                # need a different y for loop that uses the end point
                # limits in the x-direction. First find what these are
                modulus_x = xlength % xbin
                start_x = (xlength - modulus_x) + 1

                # initiate for loop for this scenario
                for y in range(0, ylength, ybin):

                    # this configuration means we will first
                    # be looping down the way, for each row
                    if y + ybin >= ylength:

                        # we've exceeded the original spaxel limit
                        # meaning that indicing will fail. create the final bin
                        modulus_y = ylength % ybin
                        start_y = (ylength - modulus_y) + 1

                        # note the + 1 is required for proper indexing

                        # now take into account the chosen interpolation type
                        if interp == 'sum':
                            # print 'Sum interpolation chosen'
                            new_spaxel = np.nansum(
                                np.nansum(data[:, start_x:xlength - 1,
                                          start_y:ylength - 1],
                                          axis=1), axis=1)

                        elif interp == 'mean':
                            # print 'Mean interpolation chosen'
                            new_spaxel = np.nanmean(
                                np.nanmean(data[:, start_x:xlength - 1,
                                           start_y:ylength - 1],
                                           axis=1), axis=1)

                        # default value of median
                        else:
                            new_spaxel = np.nanmedian(
                                np.nanmedian(data[:, start_x:xlength - 1,
                                             start_y:ylength - 1],
                                             axis=1), axis=1)

                    # everything is okay, limit not exceeded
                    else:

                        if interp == 'sum':
                            # print 'Sum interpolation chosen'
                            new_spaxel = np.nansum(
                                np.nansum(data[:, start_x:xlength - 1,
                                          y:y + ybin],
                                          axis=1), axis=1)

                        elif interp == 'mean':
                            # print 'Mean interpolation chosen'
                            new_spaxel = np.nanmean(
                                np.nanmean(data[:, start_x:xlength - 1,
                                           y:y + ybin],
                                           axis=1), axis=1)

                        # default value of median
                        else:
                            new_spaxel = np.nanmedian(
                                np.nanmedian(data[:, start_x:xlength - 1,
                                             y:y + ybin],
                                             axis=1), axis=1)

                    # add the new spaxel to the new_data
                    # cube in the correct position
                    new_data[:, x_cube_counter, y_cube_counter] = new_spaxel

                    # increment both the x and y cube counters
                    y_cube_counter += 1

            else:

                # everything is okay and the xlimit has not been reached
                for y in range(0, ylength, ybin):

                    # this configuration means we will first
                    # be looping down the way, for each row
                    if y + ybin >= ylength:

                        # we've exceeded the original spaxel limit
                        # meaning that indicing will fail. create the final bin
                        modulus_y = ylength % ybin
                        start_y = (ylength - modulus_y) + 1

                        if interp == 'sum':
                            # print 'Sum interpolation chosen'
                            new_spaxel = np.nansum(
                                np.nansum(data[:, x:x + xbin,
                                          start_y:ylength - 1],
                                          axis=1), axis=1)

                        elif interp == 'mean':
                            # print 'Mean interpolation chosen'
                            new_spaxel = np.nanmean(
                                np.nanmean(data[:, x:x + xbin,
                                           start_y:ylength - 1],
                                           axis=1), axis=1)

                        # default value of median
                        else:
                            new_spaxel = np.nanmedian(
                                np.nanmedian(data[:, x:x + xbin,
                                             start_y:ylength - 1],
                                             axis=1), axis=1)

                    # everything is okay, limit not exceeded
                    else:

                        if interp == 'sum':
                            # print 'Sum interpolation chosen'
                            new_spaxel = np.nansum(
                                np.nansum(data[:, x:x + xbin,
                                          y:y + ybin],
                                          axis=1), axis=1)

                        elif interp == 'mean':
                            # print 'Mean interpolation chosen'
                            new_spaxel = np.nanmean(
                                np.nanmean(data[:, x:x + xbin,
                                           y:y + ybin],
                                           axis=1), axis=1)

                        # default value of median
                        else:
                            new_spaxel = np.nanmedian(
                                np.nanmedian(data[:, x:x + xbin,
                                             y:y + ybin],
                                             axis=1), axis=1)

                    # add the new spaxel to the new_data
                    # cube in the correct position
                    new_data[:, x_cube_counter, y_cube_counter] = new_spaxel

                    # increment both the x and y cube counters
                    y_cube_counter += 1
            x_cube_counter += 1

        # return the new_data
        return new_data

    def OIII_vel_map(self,
                     redshift,
                     savefig=False,
                     binning=False,
                     **kwargs):

        """
        Def:
        given the redshift of the galaxy, compute the associated
        velocity field, taking into account which pixels
        have appropriate signal to noise for the measurements.

        Guess the initial parameters using a gaussian fit to the
        integrated spectrum.

        Input: redshift - redshift of the galaxy
               savefig - option to save plot or not
               binning - choosing to combine the 0.1'' pixels or not
               **kwargs
                      xbin - must be specified if binning
                        this is the number of xpixels to combine together
                      ybin - must be specified if binning
                        this is the number of ypixels to combine together
                      interp - type of interpolation. Either sum, median or
                        mean this is set to median by default
                      params - initial parameters for the gaussian fit. Note
                        it is intended that these paramters should be found
                        by using the galExtact method in pipelineClass. This is
                        a dictionary containing the entries centre, sigma and
                        amplitude

        Output: arrays containing both the OIII velocity measured in each
        (possibly binned) spaxel and the OIII velocity dispersion
        """

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        # if binning is true, take the median of adjacent spaxels
        # this uses the spaxel_binning method which can bin in any
        # different combination of shapes
        if binning:

            # check to see if the bins have been defined
            try:
                kwargs['xbin']
            except KeyError:
                raise KeyError('xbin argument not supplied to function')

            try:
                kwargs['ybin']
            except KeyError:
                raise KeyError('ybin argument not supplied to function')

            # check that both bins are integers less than 10

            if (np.equal(np.mod(kwargs['xbin'], 1), 0)
                    and kwargs['xbin'] < 10.0
                    and np.equal(np.mod(kwargs['ybin'], 1), 0)
                    and kwargs['ybin'] < 10.0):

                xbin = kwargs['xbin']
                ybin = kwargs['ybin']

            else:
                raise ValueError("Non-integer or binsize too large")

            # check for an interpolation keyword
            try:
                kwargs['interp']

                if kwargs['interp'] == 'sum':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='sum')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='sum')

                elif kwargs['interp'] == 'mean':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='mean')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='mean')

                # default median value chosen
                else:

                    # important that the data and noise have the same binning
                    data = self.spaxel_binning(data, xbin, ybin)
                    noise = self.spaxel_binning(noise, xbin, ybin)

            # case where no interpolation keyword is supplied
            except KeyError:
                print 'No interpolation keyword - using median'
                data = self.spaxel_binning(data, xbin, ybin)
                noise = self.spaxel_binning(noise, xbin, ybin)

        # the shape of the data is (spectrum, xpixel, ypixel)
        # loop through each x and y pixel and get the OIII5007 S/N
        xpixs = data.shape[1]
        ypixs = data.shape[2]

        # set the central wavelength of the OIII line
        oiii5007_wl = 0.500824 * (1. + redshift)

        # initialise the empty velocity array
        OIII_vel_array = np.empty(shape=(xpixs, ypixs))
        OIII_sigma_array = np.empty(shape=(xpixs, ypixs))

        # look for the gaussian parameters
        gauss_centre = kwargs.get('centre_oiii', oiii5007_wl)
        gauss_sigma = kwargs.get('sigma_oiii', 0.0004)
        gauss_amp = kwargs.get('amplitude_oiii', 0.001)

        # associate the central wavelength with a line index
        line_idx = np.argmin(np.abs(wl - gauss_centre))

        print gauss_centre, oiii5007_wl, gauss_sigma, gauss_amp

        for i, xpix in enumerate(np.arange(0, xpixs, 1)):

            for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                spaxel_spec = data[:, i, j]
                spaxel_noise = noise[:, i, j]

                line_counts = np.median(spaxel_spec[line_idx - 3:
                                                    line_idx + 3])

                line_noise = np.median(spaxel_noise[line_idx - 3:
                                                    line_idx + 3])

                line_sn = line_counts / line_noise

                # check for nan, inf, poor s/n
                if np.isnan(line_sn):
                    OIII_vel_array[i, j] = np.nan
                    OIII_sigma_array[i, j] = np.nan

                elif np.isinf(line_sn):
                    OIII_vel_array[i, j] = np.nan
                    OIII_sigma_array[i, j] = np.nan

                elif line_sn < 1.0:
                    OIII_vel_array[i, j] = np.nan
                    OIII_sigma_array[i, j] = np.nan

                # now the condition where we have good s/n
                # can fit a gaussian to the data in each spaxel

                else:

                    # print 'Passed with S/N of: %s' % line_sn
                    # isolate the flux and wavelength data
                    # to be used in the gaussian fit
                    # print 'Gaussian fitting spaxel [%s,%s]' % (i, j)

                    fit_wl = wl[line_idx - 10: line_idx + 10]
                    fit_flux = spaxel_spec[line_idx - 10: line_idx + 10]
                    fit_noise = spaxel_noise[line_idx - 10: line_idx + 10]

                    # construct gaussian model using lmfit
                    gmod = GaussianModel()
                    # set the initial parameter values
                    pars = gmod.make_params()
                    pars['center'].set(value=gauss_centre,
                                       min=gauss_centre - 0.0015,
                                       max=gauss_centre + 0.0015)

                    pars['sigma'].set(value=gauss_sigma,
                                      min=gauss_sigma - (0.5 * gauss_sigma),
                                      max=gauss_sigma + (0.5 * gauss_sigma))

                    pars['amplitude'].set(value=gauss_amp)

                    # perform the fit
                    out = gmod.fit(fit_flux, pars, x=fit_wl)

                    # print the fit report
                    # print out.fit_report()

                    # plot to make sure things are working
#                    fig, ax = plt.subplots(figsize=(14,6))
#                    ax.plot(fit_wl, fit_flux, color='blue')
#                    ax.plot(fit_wl, out.best_fit, color='red')
#                    ax.plot(fit_wl, fit_noise, color='green')
#                    plt.show()

                    # assuming that the redshift measured in qfits is the
                    # correct one - subtract the fitted centre and convert
                    # to kms-1
                    c = 2.99792458E5

                    OIII_vel = c * ((out.best_values['center']
                                    - oiii5007_wl) / oiii5007_wl)

                    OIII_sig = c * ((out.best_values['sigma']) / oiii5007_wl)

                    # add this result to the velocity array
                    OIII_vel_array[i, j] = OIII_vel
                    OIII_sigma_array[i, j] = OIII_sig

        # create a plot of the velocity field

        vel_fig, vel_ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
        # vel_fig.subplots_adjust(right=0.83)
        # cbar_ax = vel_fig.add_axes([0.85, 0.15, 0.02, 0.7])
        vel_ax[0].minorticks_on()
        vel_ax[1].minorticks_on()

        # sometimes this throws a TypeError if hardly any data points
        try:

            vel_min, vel_max = np.nanpercentile(OIII_vel_array, [10.0, 90.0])
            sig_min, sig_max = np.nanpercentile(OIII_sigma_array, [10.0, 90.0])

        except TypeError:

            # origin of the error is lack of good S/N data
            # can set the max and min at whatever
            vel_min, vel_max = [-100, 100]
            sig_min, sig_max = [0, 100]

        im_vel = vel_ax[0].imshow(OIII_vel_array, aspect='auto',
                                  vmin=vel_min,
                                  vmax=vel_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[0].set_title('[OIII] velocity')

        # add colourbar to each plot
        divider_vel = make_axes_locatable(vel_ax[0])
        cax_vel = divider_vel.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_vel, cax=cax_vel)

        im_sig = vel_ax[1].imshow(OIII_sigma_array, aspect='auto',
                                  vmin=sig_min,
                                  vmax=sig_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[1].set_title('[OIII] Dispersion')

        # add colourbar to each plot
        divider_sig = make_axes_locatable(vel_ax[1])
        cax_sig = divider_sig.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_sig, cax=cax_sig)

        # vel_fig.colorbar(im)

        # plt.tight_layout()
        # plt.show()
        if savefig:
            if binning:
                vel_fig.savefig('%s_velocity_OIII_binned.pdf'
                                % self.fileName[:-5])
            else:
                vel_fig.savefig('%s_velocity_OIII.pdf' % self.fileName[:-5])
        plt.close('all')

        # also write out the velocity array to a fits image file
        # will use a very simple format now with no header and
        # only a single primary extension

        hdu = fits.PrimaryHDU(OIII_vel_array)
        hdu.writeto('%s_velocity_map.fits' % self.fileName[:-5], clobber=True)

        # return the velocity array
        return OIII_vel_array, OIII_sigma_array

    def OII_vel_map(self,
                    redshift,
                    savefig=False,
                    binning=False,
                    **kwargs):
        """
        Def:
        given the redshift of the galaxy, compute the associated
        velocity field, taking into account which pixels
        have appropriate signal to noise for the measurements.

        Guess the initial parameters using a gaussian fit to the
        integrated spectrum.

        Input: redshift - redshift of the galaxy
               savefig - option to save plot or not
               binning - choosing to combine the 0.1'' pixels or not
               **kwargs
                      xbin - must be specified if binning
                        this is the number of xpixels to combine together
                      ybin - must be specified if binning
                        this is the number of ypixels to combine together
                      interp - type of interpolation. Either sum, median or
                        mean this is set to median by default
                      params - initial parameters for the gaussian fit. Note
                        it is intended that these paramters should be found
                        by using the galExtact method in pipelineClass. This is
                        a dictionary containing the entries centre, sigma and
                        amplitude

        Output: arrays containing both the OII velocity measured in each
        (possibly binned) spaxel and the OII velocity dispersion
        """
        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        # if binning is true, take the median of adjacent spaxels
        # this uses the spaxel_binning method which can bin in any
        # different combination of shapes
        if binning:

            # check to see if the bins have been defined
            try:
                kwargs['xbin']
            except KeyError:
                raise KeyError('xbin argument not supplied to function')

            try:
                kwargs['ybin']
            except KeyError:
                raise KeyError('ybin argument not supplied to function')

            # check that both bins are integers less than 10

            if (np.equal(np.mod(kwargs['xbin'], 1), 0)
                    and kwargs['xbin'] < 10.0
                    and np.equal(np.mod(kwargs['ybin'], 1), 0)
                    and kwargs['ybin'] < 10.0):

                xbin = kwargs['xbin']
                ybin = kwargs['ybin']

            else:
                raise ValueError("Non-integer or binsize too large")

            # check for an interpolation keyword
            try:
                kwargs['interp']

                if kwargs['interp'] == 'sum':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='sum')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='sum')

                elif kwargs['interp'] == 'mean':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='mean')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='mean')

                # default median value chosen
                else:

                    # important that the data and noise have the same binning
                    data = self.spaxel_binning(data, xbin, ybin)
                    noise = self.spaxel_binning(noise, xbin, ybin)

            # case where no interpolation keyword is supplied
            except KeyError:

                print 'No interpolation keyword - using median'
                data = self.spaxel_binning(data, xbin, ybin)
                noise = self.spaxel_binning(noise, xbin, ybin)

        # the shape of the data is (spectrum, xpixel, ypixel)
        # loop through each x and y pixel and get the OIII5007 S/N
        xpixs = data.shape[1]
        ypixs = data.shape[2]

        # set the central wavelength of the OIII line
        oii_wl = 0.3729875 * (1. + redshift)

        # initialise the empty velocity array
        OII_vel_array = np.empty(shape=(xpixs, ypixs))
        OII_sigma_array = np.empty(shape=(xpixs, ypixs))

        # look for the gaussian parameters
        gauss_centre = kwargs.get('centre_oii', oii_wl)
        gauss_sigma = kwargs.get('sigma_oii', 0.0008)
        gauss_amp = kwargs.get('amplitude_oii', 0.001)

        # associate the central wavelength with a line index
        line_idx = np.argmin(np.abs(wl - gauss_centre))

        print gauss_centre, oii_wl, gauss_sigma, gauss_amp

        for i, xpix in enumerate(np.arange(0, xpixs, 1)):

            for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                spaxel_spec = data[:, i, j]
                spaxel_noise = noise[:, i, j]

                line_counts = np.median(spaxel_spec[line_idx - 3:
                                                    line_idx + 3])

                line_noise = np.median(spaxel_noise[line_idx - 3:
                                                    line_idx + 3])

                line_sn = line_counts / line_noise

                # check for nan, inf, poor s/n
                if np.isnan(line_sn):
                    OII_vel_array[i, j] = np.nan
                    OII_sigma_array[i, j] = np.nan

                elif np.isinf(line_sn):
                    OII_vel_array[i, j] = np.nan
                    OII_sigma_array[i, j] = np.nan

                elif line_sn < 1.8:
                    OII_vel_array[i, j] = np.nan
                    OII_sigma_array[i, j] = np.nan

                # now the condition where we have good s/n
                # can fit a gaussian to the data in each spaxel

                else:

                    # print 'Passed with S/N of: %s' % line_sn
                    # isolate the flux and wavelength data
                    # to be used in the gaussian fit
                    # print 'Gaussian fitting spaxel [%s,%s]' % (i, j)

                    fit_wl = wl[line_idx - 10: line_idx + 10]
                    fit_flux = spaxel_spec[line_idx - 10: line_idx + 10]
                    fit_noise = spaxel_noise[line_idx - 10: line_idx + 10]

                    # construct gaussian model using lmfit
                    gmod = GaussianModel()
                    # set the initial parameter values
                    pars = gmod.make_params()
                    pars['center'].set(value=gauss_centre,
                                       min=gauss_centre - 0.0015,
                                       max=gauss_centre + 0.0015)

                    pars['sigma'].set(value=gauss_sigma,
                                      min=gauss_sigma - (0.5 * gauss_sigma),
                                      max=gauss_sigma + (0.5 * gauss_sigma))

                    pars['amplitude'].set(value=gauss_amp)

                    # perform the fit
                    out = gmod.fit(fit_flux, pars, x=fit_wl)

                    # assuming that the redshift measured in qfits is the
                    # correct one - subtract the fitted centre and convert
                    # to kms-1
                    c = 2.99792458E5
                    OII_vel = c * ((out.best_values['center']
                                   - oii_wl) / oii_wl)

                    OII_sig = c * ((out.best_values['sigma'])
                                   / oii_wl)

                    # add this result to the velocity array
                    OII_vel_array[i, j] = OII_vel
                    OII_sigma_array[i, j] = OII_sig

        # create a plot of the velocity field

        vel_fig, vel_ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
        # vel_fig.subplots_adjust(right=0.83)
        # cbar_ax = vel_fig.add_axes([0.85, 0.15, 0.02, 0.7])
        vel_ax[0].minorticks_on()
        vel_ax[1].minorticks_on()

        # sometimes this throws a TypeError if hardly any data points
        try:

            vel_min, vel_max = np.nanpercentile(OII_vel_array, [2.5, 97.5])
            sig_min, sig_max = np.nanpercentile(OII_sigma_array, [2.5, 97.5])

        except TypeError:

            # origin of the error is lack of good S/N data
            # can set the max and min at whatever
            vel_min, vel_max = [-100, 100]
            sig_min, sig_max = [0, 100]

        im_vel = vel_ax[0].imshow(OII_vel_array, aspect='auto',
                                  vmin=vel_min,
                                  vmax=vel_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[0].set_title('[OII] velocity')

        # add colourbar to each plot
        divider_vel = make_axes_locatable(vel_ax[0])
        cax_vel = divider_vel.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_vel, cax=cax_vel)

        im_sig = vel_ax[1].imshow(OII_sigma_array,
                                  aspect='auto',
                                  vmin=sig_min,
                                  vmax=sig_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[1].set_title('[OII] Dispersion')

        # add colourbar to each plot
        divider_sig = make_axes_locatable(vel_ax[1])
        cax_sig = divider_sig.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_sig, cax=cax_sig)

        # vel_fig.colorbar(im)

        # plt.tight_layout()
        # plt.show()
        if savefig:
            if binning:
                vel_fig.savefig('%s_velocity_OII_binned.pdf'
                                % self.fileName[:-5])
            else:
                vel_fig.savefig('%s_velocity_OII.pdf' % self.fileName[:-5])
        plt.close('all')

        # also write out the velocity array to a fits image file
        # will use a very simple format now with no header and
        # only a single primary extension

        hdu = fits.PrimaryHDU(OII_vel_array)
        hdu.writeto('%s_velocity_map.fits' % self.fileName[:-5], clobber=True)

        # return the velocity array
        return OII_vel_array, OII_sigma_array

    def Hb_vel_map(self,
                   redshift,
                   savefig=False,
                   binning=False,
                   **kwargs):
        """
        Def:
        given the redshift of the galaxy, compute the associated
        velocity field, taking into account which pixels
        have appropriate signal to noise for the measurements.

        Guess the initial parameters using a gaussian fit to the
        integrated spectrum.

        Input: redshift - redshift of the galaxy
               savefig - option to save plot or not
               binning - choosing to combine the 0.1'' pixels or not
               **kwargs
                      xbin - must be specified if binning
                        this is the number of xpixels to combine together
                      ybin - must be specified if binning
                        this is the number of ypixels to combine together
                      interp - type of interpolation. Either sum, median or
                        mean this is set to median by default
                      params - initial parameters for the gaussian fit. Note
                        it is intended that these paramters should be found
                        by using the galExtact method in pipelineClass. This is
                        a dictionary containing the entries centre, sigma and
                        amplitude

        Output: arrays containing both the Hb velocity measured in each
        (possibly binned) spaxel and the Hb velocity dispersion
        """
        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelegnth index of the oiii5007 line:
        wl_0 = self.Table[1].header['CRVAL3']
        dwl = self.Table[1].header['CDELT3']
        wl_n = wl_0 + (data.shape[0] * dwl)

        wl = np.linspace(wl_0, wl_n, data.shape[0])

        # if binning is true, take the median of adjacent spaxels
        # this uses the spaxel_binning method which can bin in any
        # different combination of shapes
        if binning:

            # check to see if the bins have been defined
            try:
                kwargs['xbin']
            except KeyError:
                raise KeyError('xbin argument not supplied to function')

            try:
                kwargs['ybin']
            except KeyError:
                raise KeyError('ybin argument not supplied to function')

            # check that both bins are integers less than 10

            if (np.equal(np.mod(kwargs['xbin'], 1), 0)
                    and kwargs['xbin'] < 10.0
                    and np.equal(np.mod(kwargs['ybin'], 1), 0)
                    and kwargs['ybin'] < 10.0):

                xbin = kwargs['xbin']
                ybin = kwargs['ybin']

            else:
                raise ValueError("Non-integer or binsize too large")

            # check for an interpolation keyword
            try:
                kwargs['interp']

                if kwargs['interp'] == 'sum':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='sum')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='sum')

                elif kwargs['interp'] == 'mean':

                    data = self.spaxel_binning(data,
                                               xbin,
                                               ybin,
                                               interp='mean')

                    noise = self.spaxel_binning(noise,
                                                xbin,
                                                ybin,
                                                interp='mean')

                # default median value chosen
                else:

                    # important that the data and noise have the same binning
                    data = self.spaxel_binning(data, xbin, ybin)
                    noise = self.spaxel_binning(noise, xbin, ybin)

            # case where no interpolation keyword is supplied
            except KeyError:
                print 'No interpolation keyword - using median'
                data = self.spaxel_binning(data, xbin, ybin)
                noise = self.spaxel_binning(noise, xbin, ybin)

        # the shape of the data is (spectrum, xpixel, ypixel)
        # loop through each x and y pixel and get the OIII5007 S/N
        xpixs = data.shape[1]
        ypixs = data.shape[2]

        # set the central wavelength of the OIII line
        hb_wl = 0.486268 * (1. + redshift)
        line_idx = np.argmin(np.abs(wl - hb_wl))

        # initialise the empty velocity array
        Hb_vel_array = np.empty(shape=(xpixs, ypixs))
        Hb_sigma_array = np.empty(shape=(xpixs, ypixs))

        # look for the gaussian parameters
        gauss_centre = kwargs.get('centre_hb', hb_wl)
        gauss_sigma = kwargs.get('sigma_hb', 0.0008)
        gauss_amp = kwargs.get('amplitude_hb', 0.001)

        # associate the central wavelength with a line index
        line_idx = np.argmin(np.abs(wl - gauss_centre))

        print gauss_centre, hb_wl, gauss_sigma, gauss_amp

        for i, xpix in enumerate(np.arange(0, xpixs, 1)):

            for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                spaxel_spec = data[:, i, j]
                spaxel_noise = noise[:, i, j]

                line_counts = np.median(spaxel_spec[line_idx - 3:
                                                    line_idx + 3])

                line_noise = np.median(spaxel_noise[line_idx - 3:
                                                    line_idx + 3])

                line_sn = line_counts / line_noise

                # check for nan, inf, poor s/n
                if np.isnan(line_sn):
                    Hb_vel_array[i, j] = np.nan
                    Hb_sigma_array[i, j] = np.nan

                elif np.isinf(line_sn):
                    Hb_vel_array[i, j] = np.nan
                    Hb_sigma_array[i, j] = np.nan

                elif line_sn < 1.8:
                    Hb_vel_array[i, j] = np.nan
                    Hb_sigma_array[i, j] = np.nan

                # now the condition where we have good s/n
                # can fit a gaussian to the data in each spaxel

                else:

                    # print 'Passed with S/N of: %s' % line_sn
                    # isolate the flux and wavelength data
                    # to be used in the gaussian fit
                    # print 'Gaussian fitting spaxel [%s,%s]' % (i, j)

                    fit_wl = wl[line_idx - 10: line_idx + 10]
                    fit_flux = spaxel_spec[line_idx - 10: line_idx + 10]
                    fit_noise = spaxel_noise[line_idx - 10: line_idx + 10]

                    # construct gaussian model using lmfit
                    gmod = GaussianModel()
                    # set the initial parameter values
                    pars = gmod.make_params()
                    pars['center'].set(value=gauss_centre,
                                       min=gauss_centre - 0.0015,
                                       max=gauss_centre + 0.0015)

                    pars['sigma'].set(value=gauss_sigma,
                                      min=gauss_sigma - (0.5 * gauss_sigma),
                                      max=gauss_sigma + (0.5 * gauss_sigma))

                    pars['amplitude'].set(value=gauss_amp)

                    # perform the fit
                    out = gmod.fit(fit_flux, pars, x=fit_wl)

                    # assuming that the redshift measured in qfits is the
                    # correct one - subtract the fitted centre and convert
                    # to kms-1
                    c = 2.99792458E5

                    Hb_vel = c * ((out.best_values['center']
                                  - hb_wl) / hb_wl)

                    Hb_sig = c * ((out.best_values['sigma'])
                                  / hb_wl)

                    # add this result to the velocity array
                    Hb_vel_array[i, j] = Hb_vel
                    Hb_sigma_array[i, j] = Hb_sig

        # create a plot of the velocity field

        vel_fig, vel_ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
        # vel_fig.subplots_adjust(right=0.83)
        # cbar_ax = vel_fig.add_axes([0.85, 0.15, 0.02, 0.7])
        vel_ax[0].minorticks_on()
        vel_ax[1].minorticks_on()

        # sometimes this throws a TypeError if hardly any data points
        try:

            vel_min, vel_max = np.nanpercentile(Hb_vel_array, [2.5, 97.5])
            sig_min, sig_max = np.nanpercentile(Hb_sigma_array, [2.5, 97.5])

        except TypeError:

            # origin of the error is lack of good S/N data
            # can set the max and min at whatever
            vel_min = -100
            vel_max = 100
            sig_min = 0
            sig_max = 100

        im_vel = vel_ax[0].imshow(Hb_vel_array, aspect='auto',
                                  vmin=vel_min,
                                  vmax=vel_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[0].set_title('[Hb] velocity')

        # add colourbar to each plot
        divider_vel = make_axes_locatable(vel_ax[0])
        cax_vel = divider_vel.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_vel, cax=cax_vel)

        im_sig = vel_ax[1].imshow(Hb_sigma_array, aspect='auto',
                                  vmin=sig_min,
                                  vmax=sig_max,
                                  interpolation='nearest',
                                  cmap=plt.get_cmap('jet'))

        vel_ax[1].set_title('[Hb] Dispersion')

        # add colourbar to each plot
        divider_sig = make_axes_locatable(vel_ax[1])
        cax_sig = divider_sig.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im_sig, cax=cax_sig)

        # vel_fig.colorbar(im)

        # plt.tight_layout()
        # plt.show()
        if savefig:
            if binning:
                vel_fig.savefig('%s_velocity_Hb_binned.pdf'
                                % self.fileName[:-5])
            else:
                vel_fig.savefig('%s_velocity_Hb.pdf' % self.fileName[:-5])
        plt.close('all')

        # also write out the velocity array to a fits image file
        # will use a very simple format now with no header and
        # only a single primary extension

        hdu = fits.PrimaryHDU(Hb_vel_array)
        hdu.writeto('%s_velocity_map.fits' % self.fileName[:-5], clobber=True)

        # return the velocity array
        return Hb_vel_array, Hb_sigma_array

    def make_sn_map(self,
                    line,
                    redshift):

        """
        Def:
        Make a 2D grid of the s/n in the datacube, return this as output.
        This is done for the chosen emission line specified by line.
        Will throw an error if the line is outside the wavelength range.
        Also returns the signal array and the noise array, which really are
        the important outputs for doing the voronoi binning.

        Input:
                line - emission line to fit, must be either oiii, oii, hb
                redshift - the redshift value of the incube
        """

        if not(line != 'oiii' or line != 'oii' or line != 'hb'):

            raise ValueError('Please ensure that you have'
                             + ' chosen an appropriate emission line')

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelength array
        wave_array = self.wave_array

        if line == 'oiii':

            central_wl = 0.500824 * (1. + redshift)

        elif line == 'hb':

            central_wl = 0.486268 * (1. + redshift)

        elif line == 'oii':

            central_wl = 0.3729875 * (1. + redshift)

        # find the index of the chosen emission line
        line_idx = np.argmin(np.abs(wave_array - central_wl))

        # the shape of the data is (spectrum, xpixel, ypixel)
        # loop through each x and y pixel and get the OIII5007 S/N
        xpixs = data.shape[1]
        ypixs = data.shape[2]

        sn_array = np.empty(shape=(xpixs, ypixs))
        signal_array = np.empty(shape=(xpixs, ypixs))
        noise_array = np.empty(shape=(xpixs, ypixs))

        for i, xpix in enumerate(np.arange(0, xpixs, 1)):

            for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                spaxel_spec = data[:, i, j]
                spaxel_noise = noise[:, i, j]

                # account for different spectral resolutions

                if self.filter == 'K':

                    # first search for the linepeak, which may be different
                    # to that specified by the systemic redshift

                    t_index = np.argmax(spaxel_spec[line_idx - 4:
                                                    line_idx + 4])

                    # need this to be an absolute index
                    t_index = t_index + line_idx - 4

                    # then sum the flux inside the region over which the line
                    # will be. Width of line is roughly 0.003, which is 10
                    # spectral elements in K and 6 in HK

                    line_counts = np.nansum(spaxel_spec[t_index - 4:
                                                        t_index + 4])

                elif self.filter == 'HK':

                    t_index = np.argmax(spaxel_spec[line_idx - 2:
                                                    line_idx + 2])

                    t_index = t_index + line_idx - 2

                    line_counts = np.nansum(spaxel_spec[t_index - 3:
                                                        t_index + 3])
                else:

                    t_index = np.argmax(spaxel_spec[line_idx - 4:
                                                    line_idx + 4])

                    t_index = t_index + line_idx - 4

                    line_counts = np.nansum(spaxel_spec[t_index - 4:
                                                        t_index + 4])

                if np.isnan(line_counts):

                    signal_array[i, j] = 0

                else:

                    signal_array[i, j] = line_counts

                # mask out the skylines to compute the noise
                if self.filter == 'K':
                    wavelength_masked, \
                        noise_masked = self.mask_k_sky(wave_array,
                                                       spaxel_spec)
                elif self.filter == 'HK':
                    wavelength_masked, \
                        noise_masked = self.mask_hk_sky(wave_array,
                                                        spaxel_spec)

                # look only at the unmasked flux values
                noise_data = noise_masked.compressed() / 1E-18

                # construct histogram of the noise values and fit
                mod = GaussianModel()
                hist, edges = np.histogram(noise_data, bins=10)
                edges = edges[0: -1]
                params = mod.guess(hist, x=edges)

                # try the fitting with astropy
#                g1 = models.Gaussian1D(amplitude=params['amplitude'].value,
#                                       mean=params['center'].value,
#                                       stddev=params['sigma'].value)
#                fit_g = fitting.LevMarLSQFitter()
#                g_out = fit_g(g1, x=edges, y=hist)
#                print g_out.stddev.value

                out = mod.fit(hist, params, x=edges)
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                ax.plot(edges, hist)
                ax.plot(edges, out.best_fit)
                # plt.show()
                plt.close('all')
#                print out.fit_report()
                if list(np.where(np.isnan(noise_data))[0]):
                    print 'found nan'
                    line_noise = np.nan
                else:
                    line_noise = out.best_values['sigma'] * 1E-18

                # line_noise = g_out.stddev.value
                # print line_noise, i, j

                noise_array[i, j] = line_noise

                line_sn = line_counts / line_noise

                if np.isnan(line_sn):

                    sn_array[i, j] = -99.

                else:

                    sn_array[i, j] = line_sn

        # loop around noise array to clean up nan entries
        for i in range(0, len(noise_array)):
            for j in range(0, len(noise_array[0])):
                if np.isnan(noise_array[i][j]):
                    print 'Fixing nan value'
                    noise_array[i][j] = np.nanmedian(noise_array)

        # print noise_array
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(signal_array / noise_array,
                       cmap=plt.get_cmap('jet'),
                       vmin=0,
                       vmax=10)
        plt.colorbar(im)
        plt.show()

        return signal_array, noise_array

    def mask_k_sky(self, wavelength, flux):

        """
        Def:
        Purpose is to mask the sky emission lines so that a proper estimation
        of the noise of a spaxel can be found. Uses a pre-determined set of
        values at which the skylines are known to exist, masks the wavelength
        array and then applies the mask to the flux array.
        Returns the masked versions of both wavelength and flux arrays.

        Input:
                wavelength - k-band wavelength array
                flux - corresponding flux values

        Output:
                wavelength_masked - np.masked_array version with sky masked
                flux_masked - as above, corresponding
        """
        
        # first hardwire in the known wavelength ranges of the skylines
        initial_value = 1.995
        final_value = 2.253

        # pairs of limits to use in the forloop
        sky_dict = {1: [1.9994, 2.00447],
                    2: [2.00618, 2.00783],
                    3: [2.01817, 2.02044],
                    4: [2.02631, 2.02893],
                    5: [2.03304, 2.0348],
                    6: [2.0398, 2.04237],
                    7: [2.04903, 2.0508],
                    8: [2.05491, 2.05755],
                    9: [2.0718, 2.07444],
                    10: [2.08533, 2.08707],
                    11: [2.08951, 2.09204],
                    12: [2.10123, 2.13916],
                    13: [2.14931, 2.15943],
                    14: [2.1628, 2.16447],
                    15: [2.17038, 2.17207],
                    16: [2.17541, 2.17713],
                    17: [2.17845, 2.18133],
                    18: [2.18639, 2.18809],
                    19: [2.19471, 2.20045],
                    20: [2.20407, 2.2066],
                    21: [2.21167, 2.21366],
                    22: [2.22336, 2.22608],
                    23: [2.23019, 2.23521],
                    24: [2.24529, 2.24702],
                    25: [2.25039, 2.25296]
                    }

        # now do the masking of the wavelength array
        wavelength_masked = ma.masked_where(wavelength < initial_value,
                                            wavelength,
                                            copy=True)

        # now loop through and mask off all of the offending regions

        for entry in sky_dict:

            wavelength_masked = ma.masked_where(
                np.logical_and(wavelength_masked > sky_dict[entry][0],
                               wavelength_masked < sky_dict[entry][1]),
                wavelength_masked, copy=True)

        wavelength_masked = ma.masked_where(wavelength_masked > final_value,
                                            wavelength_masked,
                                            copy=True)

        # apply the final mask to the flux array

        flux_masked = ma.MaskedArray(flux,
                                     mask=wavelength_masked.mask)

        return wavelength_masked, flux_masked

    def mask_hk_sky(self, wavelength, flux):

        """
        Def:
        Purpose is to mask the sky emission lines so that a proper estimation
        of the noise of a spaxel can be found. Uses a pre-determined set of
        values at which the skylines are known to exist, masks the wavelength
        array and then applies the mask to the flux array.
        Returns the masked versions of both wavelength and flux arrays.

        Input:
                wavelength - hk-band wavelength array
                flux - corresponding flux values

        Output:
                wavelength_masked - np.masked_array version with sky masked
                flux_masked - as above, corresponding
        """
        
        # first hardwire in the known wavelength ranges of the skylines
        initial_value = 1.50059
        final_value = 2.253

        # pairs of limits to use in the forloop
        sky_dict = {1: [1.50402, 1.51215],
                    2: [1.51733, 1.51992],
                    3: [1.52272, 1.52546],
                    4: [1.52763, 1.53001],
                    5: [1.53197, 1.53484],
                    6: [1.53834, 1.54059],
                    7: [1.54213, 1.54444],
                    8: [1.5492, 1.56678],
                    9: [1.56923, 1.57127],
                    10: [1.56923, 1.57127],
                    11: [1.59598, 1.59822],
                    12: [1.60192, 1.60427],
                    13: [1.60685, 1.60887],
                    14: [1.61167, 1.61402],
                    15: [1.61834, 1.62053],
                    16: [1.62237, 1.62467],
                    17: [1.62977, 1.65157],
                    18: [1.65437, 1.65616],
                    19: [1.65759, 1.66225],
                    20: [1.60779, 1.67744],
                    21: [1.68312, 1.68496],
                    22: [1.68913, 1.6915],
                    23: [1.69441, 1.69665],
                    24: [1.69982, 1.70207],
                    25: [1.70663, 1.70887],
                    26: [1.71132, 1.71349],
                    27: [1.72017, 1.72215],
                    28: [1.72367, 1.73965],
                    29: [1.74176, 1.74605],
                    30: [1.74942, 1.75365],
                    31: [1.76384, 1.77082],
                    32: [1.78014, 1.78195],
                    33: [1.78703, 1.78936],
                    34: [1.7982, 1.80043],
                    35: [1.8059, 1.80785],
                    36: [1.81075, 1.81294],
                    37: [1.82054, 1.82168],
                    38: [1.82458, 1.82629],
                    39: [1.84234, 1.84704],
                    40: [1.8515, 1.85373],
                    41: [1.8553, 1.85984],
                    42: [1.87313, 1.88056],
                    43: [1.88782, 1.89245],
                    44: [1.90409, 1.90747],
                    45: [1.9182, 1.92629],
                    46: [1.93306, 1.93719],
                    47: [1.95034, 1.97854],
                    48: [1.98, 1.981],
                    49: [1.98274, 1.98514],
                    50: [1.9883, 1.99348],
                    51: [1.9994, 2.00447],
                    52: [2.00618, 2.00783],
                    53: [2.01817, 2.02044],
                    54: [2.02631, 2.02893],
                    55: [2.03304, 2.0348],
                    56: [2.0398, 2.04237],
                    57: [2.04903, 2.0508],
                    58: [2.05491, 2.05755],
                    59: [2.0718, 2.07444],
                    60: [2.08533, 2.08707],
                    61: [2.08951, 2.09204],
                    62: [2.10123, 2.13916],
                    63: [2.14931, 2.15943],
                    64: [2.1628, 2.16447],
                    65: [2.17038, 2.17207],
                    66: [2.17541, 2.17713],
                    67: [2.17845, 2.18133],
                    68: [2.18639, 2.18809],
                    69: [2.19471, 2.20045],
                    70: [2.20407, 2.2066],
                    71: [2.21167, 2.21366],
                    72: [2.22336, 2.22608],
                    73: [2.23019, 2.23521],
                    74: [2.24529, 2.24702],
                    75: [2.25039, 2.25296]
                    }

        # now do the masking of the wavelength array
        wavelength_masked = ma.masked_where(wavelength < initial_value,
                                            wavelength,
                                            copy=True)

        # now loop through and mask off all of the offending regions

        for entry in sky_dict:

            wavelength_masked = ma.masked_where(
                np.logical_and(wavelength_masked > sky_dict[entry][0],
                               wavelength_masked < sky_dict[entry][1]),
                wavelength_masked, copy=True)

        wavelength_masked = ma.masked_where(wavelength_masked > final_value,
                                            wavelength_masked,
                                            copy=True)

        # apply the final mask to the flux array

        flux_masked = ma.MaskedArray(flux,
                                     mask=wavelength_masked.mask)

        return wavelength_masked, flux_masked

    def stott_velocity_field(self,
                             line,
                             redshift,
                             threshold,
                             centre_x,
                             centre_y,
                             tol=40,
                             method='median'):

        """
        Def:
        Make a 2D grid of the s/n in the datacube, and use this to decide
        whether or not to fit a gaussian. Only fit a gaussian when the
        s/n is above a threshold. If below this, expand the area to 3x3 spaxels
        (but for each spaxel independently) and check to see whether the s/n is
        increased or decreased. If decreased - reject that spaxel. If increased
        but still below the threshold try a 5x5 area. If the 5x5 is below the
        threshold, reject that spaxel and move on. Have to take boundary effects
        into account. Once the threshold is exceeded, fit a gaussian to the
        spectral region surrounding the emission line, with central wavelength
        value contrained by the peak flux in the emission line region.

        Input:
                line - emission line to fit, must be either oiii, oii, hb
                redshift - the redshift value of the incube
                threshold - signal to noise threshold for the fit
                redshift - redshift of emission line
                threshold - s/n threshold for inclusion in velocity field
                centre_x - centre of galaxy in x direction
                centre_y - centre of galaxy in y direction
                tol - error tolerance for gaussian fit (default of 40)
                method - stacking method when binning pixels
        Output: 
                signal array, noise array - for the given datacube
        """

        if not(line != 'oiii' or line != 'oii' or line != 'hb'):

            raise ValueError('Please ensure that you have'
                             + ' chosen an appropriate emission line')

        # open the data
        data = self.data
        noise = self.Table[2].data

        # get the wavelength array
        wave_array = self.wave_array

        if line == 'oiii':

            central_wl = 0.500824 * (1. + redshift)

        elif line == 'hb':

            central_wl = 0.486268 * (1. + redshift)

        elif line == 'oii':

            central_wl = 0.3729875 * (1. + redshift)

        # find the index of the chosen emission line
        line_idx = np.argmin(np.abs(wave_array - central_wl))

        # the shape of the data is (spectrum, xpixel, ypixel)
        # loop through each x and y pixel and get the OIII5007 S/N

        xpixs = data.shape[1]

        ypixs = data.shape[2]

        sn_array = np.empty(shape=(xpixs, ypixs))

        signal_array = np.empty(shape=(xpixs, ypixs))

        noise_array = np.empty(shape=(xpixs, ypixs))

        vel_array = np.empty(shape=(xpixs, ypixs))

        disp_array = np.empty(shape=(xpixs, ypixs))

        flux_array = np.empty(shape=(xpixs, ypixs))


        for i, xpix in enumerate(np.arange(0, xpixs, 1)):

            for j, ypix in enumerate(np.arange(0, ypixs, 1)):

                spaxel_spec = data[:, i, j]
                spaxel_noise = noise[:, i, j]

                # account for different spectral resolutions

                if self.filter == 'K':

                    # first search for the linepeak, which may be different
                    # to that specified by the systemic redshift
                    # set the upper and lower ranges for the t_index search

                    lower_t = 8
                    upper_t = 9

                    t_index = np.argmax(spaxel_spec[line_idx - lower_t:
                                                    line_idx + upper_t])

                    # need this to be an absolute index
                    t_index = t_index + line_idx - 8

                    # then sum the flux inside the region over which the line
                    # will be. Width of line is roughly 0.003, which is 10
                    # spectral elements in K and 6 in HK

                    # set the limits for the signal estimate

                    lower_limit = t_index - 10
                    upper_limit = t_index + 11

                    line_counts = np.nansum(spaxel_spec[lower_limit:
                                                        upper_limit])

                elif self.filter == 'HK':

                    lower_t = 5
                    upper_t = 6

                    t_index = np.argmax(spaxel_spec[line_idx - lower_t:
                                                    line_idx + upper_t])

                    t_index = t_index + line_idx - 5

                    # set the limits for the signal estimate

                    lower_limit = t_index - 5
                    upper_limit = t_index + 6

                    line_counts = np.nansum(spaxel_spec[lower_limit:
                                                        upper_limit])

                # any other band follows the same rules as K right now

                else:

                    lower_t = 8
                    upper_t = 9

                    t_index = np.argmax(spaxel_spec[line_idx - lower_t:
                                                    line_idx + upper_t])

                    t_index = t_index + line_idx - 8

                    # set the limits for the signal estimate

                    lower_limit = t_index - 10
                    upper_limit = t_index + 11

                    line_counts = np.nansum(spaxel_spec[lower_limit:
                                                        upper_limit])

                if np.isnan(line_counts):

                    signal_array[i, j] = 0

                else:

                    signal_array[i, j] = line_counts

                # mask out the skylines to compute the noise

                if self.filter == 'K':
                    wavelength_masked, \
                        noise_masked = self.mask_k_sky(wave_array,
                                                       spaxel_spec)

                elif self.filter == 'HK':
                    wavelength_masked, \
                        noise_masked = self.mask_hk_sky(wave_array,
                                                        spaxel_spec)

                # look only at the unmasked flux values

                noise_data = noise_masked.compressed() / 1E-18

                # construct histogram of the noise values and fit

                mod = GaussianModel()
                hist, edges = np.histogram(noise_data, bins=10)
                edges = edges[0: -1]
                params = mod.guess(hist, x=edges)

                # try the fitting with astropy
#                g1 = models.Gaussian1D(amplitude=params['amplitude'].value,
#                                       mean=params['center'].value,
#                                       stddev=params['sigma'].value)
#                fit_g = fitting.LevMarLSQFitter()
#                g_out = fit_g(g1, x=edges, y=hist)
#                print g_out.stddev.value

                out = mod.fit(hist, params, x=edges)

                fig, ax = plt.subplots(1, 1, figsize=(12, 12))

                ax.plot(edges, hist)

                ax.plot(edges, out.best_fit)

                # plt.show()

                plt.close('all')

#                print out.fit_report()

                if list(np.where(np.isnan(noise_data))[0]):

                    print 'found nan'

                    line_noise = np.nan

                else:

                    line_noise = out.best_values['sigma'] * 1E-18

                # line_noise = g_out.stddev.value
                # print line_noise, i, j

                noise_array[i, j] = line_noise

                line_sn = line_counts / line_noise

                print '%s' % line_sn

                # searching the computed signal to noise in this section

                if np.isnan(line_sn):

                    print 'getting rid of nan'

                    # we've got a nan entry - get rid of it

                    sn_array[i, j] = np.nan
                    vel_array[i, j] = np.nan
                    disp_array[i, j] = np.nan
                    flux_array[i, j] = np.nan

                elif line_sn > threshold:

                    print 'Threshold exceeded %s %s %s' % (i, j, line_sn)

                    print '%s %s %s' % (t_index, lower_limit, upper_limit)

                    # do stuff - calculate the velocity

                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                    ax.plot(wave_array[lower_limit: upper_limit],
                            spaxel_spec[lower_limit: upper_limit])

                    plt.close('all')

                    gauss_values, covar = self.gauss_fit(wave_array[lower_limit: upper_limit],
                                                         spaxel_spec[lower_limit: upper_limit])

                    # if the gaussian does not fit correctly this can throw 
                    # a nonetype error, since covar is empty
                    try:

                        if (100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']) > tol \
                           or (100 * np.sqrt(covar[1][1]) / gauss_values['center']) > tol \
                           or (100 * np.sqrt(covar[0][0]) / gauss_values['sigma']) > tol:

                            print 'Gaussian errors too large - reject fit'

                            sn_array[i, j] = line_sn
                            vel_array[i, j] = np.nan
                            disp_array[i, j] = np.nan
                            flux_array[i, j] = np.nan

                        else:

                            c = 2.99792458E5

                            vel = c * ((gauss_values['center']
                                        - central_wl) / central_wl)

                            sig = c * ((gauss_values['sigma']) / central_wl)

                            sn_array[i, j] = line_sn
                            vel_array[i, j] = vel
                            disp_array[i, j] = sig
                            flux_array[i, j] = gauss_values['amplitude']

                    except TypeError:

                        sn_array[i, j] = line_sn
                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan

                # don't bother expanding area if line_sn starts negative

                elif line_sn < 0:

                    print 'Found negative signal %s %s' % (i, j)

                    sn_array[i, j] = np.nan
                    vel_array[i, j] = np.nan
                    disp_array[i, j] = np.nan
                    flux_array[i, j] = np.nan

                # If between 0 and the threshold, search surrounding area
                # for more signal - do this in the direction of the galaxy
                # centre (don't know if this introduces a bias to the
                # measurement or not)

                elif (line_sn > 0 and line_sn < threshold):

                    print 'Attempting to improve signal: %s %s %s' % (line_sn, i, j)

                    # compute the stacked 3x3 spectrum using helper method

                    spec = self.binning_three(data,
                                              i,
                                              j,
                                              lower_limit,
                                              upper_limit,
                                              method)

                    # now that spec has been computed, look at whether
                    # the signal to noise of the stack has improved

                    new_line_counts = np.nansum(spec)

                    new_sn = new_line_counts / line_noise

                    print 'did things improve: %s %s' % (new_sn, line_sn)

                    # if the new signal to noise is greater than the
                    # threshold, save this in the cube and proceed

                    if new_sn > threshold:

                        # add to the signal to noise array

                        print 'The binning raised above threshold!'

                        sn_array[i, j] = line_sn

                        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                        ax.plot(wave_array[lower_limit: upper_limit],
                                spec)

                        plt.close('all')

                        gauss_values, covar = self.gauss_fit(wave_array[lower_limit: upper_limit],
                                                             spec)

                        try:

                            if (100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']) > tol \
                               or (100 * np.sqrt(covar[1][1]) / gauss_values['center']) > tol \
                               or (100 * np.sqrt(covar[0][0]) / gauss_values['sigma']) > tol:

                                print 'Gaussian errors too large - reject fit'

                                sn_array[i, j] = line_sn
                                vel_array[i, j] = np.nan
                                disp_array[i, j] = np.nan
                                flux_array[i, j] = np.nan

                            else:

                                c = 2.99792458E5

                                vel = c * ((gauss_values['center']
                                            - central_wl) / central_wl)

                                sig = c * ((gauss_values['sigma']) / central_wl)

                                sn_array[i, j] = line_sn
                                vel_array[i, j] = vel
                                disp_array[i, j] = sig
                                flux_array[i, j] = gauss_values['amplitude']

                        except TypeError:

                            sn_array[i, j] = line_sn
                            vel_array[i, j] = np.nan
                            disp_array[i, j] = np.nan
                            flux_array[i, j] = np.nan

                    elif new_sn <= line_sn:

                        # got worse - entry becomes a nan

                        print 'no improvement, stop trying to fix'

                        sn_array[i, j] = np.nan
                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan

                    elif (new_sn > line_sn and new_sn < threshold):

                        # try the 5x5 approach towards the cube centre

                        spec = self.binning_five(data,
                                                 i,
                                                 j,
                                                 lower_limit,
                                                 upper_limit,
                                                 method)

                    # now that spec has been computed, look at whether
                    # the signal to noise of the stack has improved

                        final_line_counts = np.nansum(spec)

                        final_sn = final_line_counts / line_noise

                        print 'did things improve: %s %s' % (final_sn, new_sn)

                        # if the new signal to noise is greater than the
                        # threshold, save this in the cube and proceed

                        if final_sn > threshold:

                            # add to the signal to noise array

                            print 'The biggest binning raised above threshold!'
                            

                            sn_array[i, j] = final_sn

                            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                            ax.plot(wave_array[lower_limit: upper_limit],
                                    spec)
                            
                            plt.close('all')

                            gauss_values, covar = self.gauss_fit(wave_array[lower_limit: upper_limit],
                                                                 spec)

                            try:

                                if (100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']) > tol \
                                   or (100 * np.sqrt(covar[1][1]) / gauss_values['center']) > tol \
                                   or (100 * np.sqrt(covar[0][0]) / gauss_values['sigma']) > tol:

                                    print 'Gaussian errors too large - reject fit'

                                    sn_array[i, j] = line_sn
                                    vel_array[i, j] = np.nan
                                    disp_array[i, j] = np.nan
                                    flux_array[i, j] = np.nan

                                else:

                                    c = 2.99792458E5

                                    vel = c * ((gauss_values['center']
                                                - central_wl) / central_wl)

                                    sig = c * ((gauss_values['sigma']) / central_wl)

                                    sn_array[i, j] = line_sn
                                    vel_array[i, j] = vel
                                    disp_array[i, j] = sig
                                    flux_array[i, j] = gauss_values['amplitude']

                            except TypeError:

                                sn_array[i, j] = line_sn
                                vel_array[i, j] = np.nan
                                disp_array[i, j] = np.nan
                                flux_array[i, j] = np.nan

                        else:

                            # didn't reach target - store as nan

                            print 'no improvement, stop trying to fix'

                            sn_array[i, j] = np.nan
                            vel_array[i, j] = np.nan
                            disp_array[i, j] = np.nan
                            flux_array[i, j] = np.nan

        # loop around noise array to clean up nan entries
        for i in range(0, len(noise_array)):
            for j in range(0, len(noise_array[0])):
                if np.isnan(noise_array[i][j]):
                    print 'Fixing nan value'
                    noise_array[i][j] = np.nanmedian(noise_array)

        # print sn_array
        # plot all of the arrays

        try:

            vel_min, vel_max = np.nanpercentile(vel_array, [5.0, 95.0])
            sig_min, sig_max = np.nanpercentile(disp_array, [5.0, 95.0])
            flux_min, flux_max = np.nanpercentile(flux_array, [5.0, 95.0])

        except TypeError:

            # origin of the error is lack of good S/N data
            # can set the max and min at whatever
            vel_min, vel_max = [-100, 100]
            sig_min, sig_max = [0, 150]
            flux_min, flux_max = [0, 5E-3]

        plt.close('all')

        # create 1x3 postage stamps of the different properties

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        im = ax[0].imshow(flux_array,
                          cmap=plt.get_cmap('jet'),
                          vmin=flux_min,
                          vmax=flux_max,
                          interpolation='nearest')

        ax[0].scatter(centre_y, centre_x, marker='x', s=3E2, color='black')
        ax[0].contour(flux_array, colors='k')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[0].set_title('[OIII] Flux')

        im = ax[1].imshow(vel_array,
                          vmin=vel_min,
                          vmax=vel_max,
                          cmap=plt.get_cmap('jet'),
                          interpolation='nearest')

        ax[1].scatter(centre_y, centre_x, marker='x', s=3E2, color='black')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[1].set_title('[OIII] Velocity')

        im = ax[2].imshow(disp_array,
                          vmin=sig_min,
                          vmax=sig_max,
                          cmap=plt.get_cmap('jet'),
                          interpolation='nearest')

        ax[2].scatter(centre_y, centre_x, marker='x', s=3E2, color='black')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[2].set_title('[OIII] Dispersion')

        plt.show()

        fig.savefig('%s_stamps_gauss%s_t%s.pdf' % (self.fileName[:-5],
                                                   str(tol),
                                                   str(threshold)))

        plt.close('all')

        return signal_array, noise_array

    def gauss_fit(self, fit_wl, fit_flux):

        """
        Def:
        Performs simple gaussian fit, guessing initial parameters from the data
        and given input wavelength and input flux values

        Input:
                fit_wl - wavelength of spectrum to fit
                fit_flux - flux of spectrum to fitsWavelength

        Output:
                fit_params - dictionary containing the best fit parameters
                            for each of the spectra
        """

        # construct gaussian model using lmfit

        gmod = GaussianModel()

        # set the initial parameter values

        pars = gmod.guess(fit_flux, x=fit_wl)

        # perform the fit
        out = gmod.fit(fit_flux, pars, x=fit_wl)

        # print the fit report
        print out.fit_report()

        # plot to make sure things are working
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(fit_wl, fit_flux, color='blue')
        ax.plot(fit_wl, out.best_fit, color='red')
        plt.show()
        plt.close('all')

        return out.best_values, out.covar

    def binning_three(self, data, i, j, lower_lim, upper_lim, method):

        """
        Def: Helper method to do the 3x3 spatial binning for the stott
        velocity field function.

        Input:
                data - datacube from the object
                i - spaxel under consideration in the stott loop
                j - same as above
                lower_lim - lower limit to truncate the spectrum
                upper_lim - upper, for the signal of the line computation
                method - method to use to stack the spectra together after

        Output:
                spec - stacked spectrum for spaxel i, j of length lower_lim
                        + upper_lim (i.e. truncated between these limits)
        """

        # first construct loop over the i - 1 - i +1 and same for jet
        # need except statement incase the cube_boundary is reached

        stack_array = []

        try:

            for a in range(i - 1, i + 2):

                for b in range(j - 1, j + 2):

                    stack_array.append(data[:, a, b])

            if method == 'median':

                spec = np.nanmedian(stack_array, axis=0)[lower_lim:upper_lim]

            elif method == 'sum':

                spec = np.nansum(stack_array, axis=0)[lower_lim:upper_lim]

            elif method == 'mean':

                spec = np.nanmean(stack_array, axis=0)[lower_lim:upper_lim]

            else:

                raise ValueError('Please choose a valid stacking method')

        except IndexError:

            print 'encountered the cube boundary'

            spec = data[:, i, j][lower_lim: upper_lim]

        return spec

    def binning_five(self, data, i, j, lower_lim, upper_lim, method):

        """
        Def: Helper method to do the 5x5 spatial binning for the stott
        velocity field function.

        Input:
                data - datacube from the object
                i - spaxel under consideration in the stott loop
                j - same as above
                lower_lim - lower limit to truncate the spectrum
                upper_lim - upper, for the signal of the line computation
                method - method to use to stack the spectra together after

        Output:
                spec - stacked spectrum for spaxel i, j of length lower_lim
                        + upper_lim (i.e. truncated between these limits)
        """

        # first construct loop over the i - 1 - i +1 and same for jet
        # need except statement incase the cube_boundary is reached

        stack_array = []

        try:

            for a in range(i - 2, i + 3):

                for b in range(j - 2, j + 3):

                    stack_array.append(data[:, a, b])

            if method == 'median':

                spec = np.nanmedian(stack_array, axis=0)[lower_lim:upper_lim]

            elif method == 'sum':

                spec = np.nansum(stack_array, axis=0)[lower_lim:upper_lim]

            elif method == 'mean':

                spec = np.nanmean(stack_array, axis=0)[lower_lim:upper_lim]

            else:

                raise ValueError('Please choose a valid stacking method')

        except IndexError:

            print 'encountered the cube boundary'

            spec = data[:, i, j][lower_lim: upper_lim]

        return spec

    def line_flux_extract(self,
                          lower_limit,
                          upper_limit,
                          i,
                          j):

        """
        Def: 
        Extract the flux around an emission line given the lower limit
        and upper limit on the indices from the vel_field_sigma method.
        This will be for spaxel i, j and will consist of summing the
        flux values between the lower limit and upper limit.

        Input:
                lower_limit, upper_limit - index limits for flux sum
                i, j - defining which spaxel to choose
        """

        # define the data as the ith,jth spaxel between computed limits

        data = self.data[:, i, j][lower_limit:upper_limit]

        # signal is the sum of the pixels

        signal = np.nansum(data)

        return signal

