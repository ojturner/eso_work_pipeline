# class for carrying out analysis of 2D velocity fields created via the
# cube class. i.e. the input data file should be a 2D field of velocity
# measurements, which are made via the pipeline class initially

# import the relevant modules
import os, sys, numpy as np, random, math
import pyraf
import numpy.polynomial.polynomial as poly
import lmfit
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import pickle
from scipy import stats
from scipy import interpolate
from lmfit.models import GaussianModel, PolynomialModel
from scipy import optimize
from lmfit import Model
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from astropy.io import fits
from pylab import *
from matplotlib.colors import LogNorm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from scipy.spatial import distance
from copy import copy
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry
from sys import stdout
import scipy.optimize as op
import emcee
import corner

# add the functions folder to the PYTHONPATH
sys.path.append('/scratch2/oturner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

import psf_blurring as psf
import rotate_pa as rt_pa

####################################################################

class vel_field(object):

    """
    Def: 
    Class for housing analysis methods relevant to 2D velocity 
    field data computed elsewhere
    Input: 
    2D array of velocity field data (IMAGE)
    """

    # Initialiser creates an instance of the cube object
    # Input must be a combined data cube with two extensions - data and noise 

    def __init__(self, fileName, c_rot_x, c_rot_y):

        """
        Def:
        Initialiser method 
        Input: filename - name of file containing 1D spectrum 
                z - The redshift of the galaxy being manipulated
        """

        self.self = self

        # Initialise the fileName object 

        self.fileName = fileName

        # set an actual object name

        try:

            if self.fileName.find("/") == -1:

                self.gal_name = copy(self.fileName)

            # Otherwise the directory structure is included and have to
            # search for the backslash and omit up to the last one

            else:

                self.gal_name = self.fileName[len(self.fileName) -
                                              self.fileName[::-1].find("/"):]

        except:

            print 'Trouble setting galaxy name'

        # print self.gal_name

        # initialise the centre of rotation in the x and y

        self.c_rot_x = c_rot_x

        self.c_rot_y = c_rot_y

        # Variable containing all the fits extensions 

        self.Table = fits.open(fileName)

        # create an object to house the data 

        self.vel_data = self.Table[0].data

        # also load the associated velocity field errors

        try:

            self.er_table = fits.open('%serror_field.fits' % self.fileName[:-14])

            self.error_data = self.er_table[0].data

        except IOError:

            print 'No associated error field'

        try:

            self.sig_table = fits.open('%ssig_field.fits' % self.fileName[:-14])

            self.sig_data = self.sig_table[0].data

        except IOError:

            print 'No associated error field'

        try:

            self.sig_er_table = fits.open('%ssig_error_field.fits' % self.fileName[:-14])

            self.sig_error_data = self.sig_er_table[0].data

        except IOError:

            print 'No associated error field'

        # variables housing the pickled chain and lnp

        self.chain_name = self.fileName[:-5] + '_chain.obj'

        self.ln_p_name = self.fileName[:-5] + '_lnp.obj'

        self.param_file = self.fileName[:-5] + '_params.txt'

        self.param_file_fixed = self.fileName[:-5] + '_params_fixed.txt'

        self.param_file_fixed_inc_fixed = self.fileName[:-5] + '_params_fixed_inc_fixed.txt'

        self.param_file_fixed_inc_vary = self.fileName[:-5] + '_params_fixed_inc_vary.txt'

        self.param_file_chi_squared = self.fileName[:-5] + '_chi_squared_params.txt'

        #initialise x and y dimensions

        self.xpix = self.vel_data.shape[0]

        self.ypix = self.vel_data.shape[1]

        # set the lengths of the bins as x & y

        xbin = np.arange(0, self.vel_data.shape[0], 1)

        ybin = np.arange(0, self.vel_data.shape[1], 1)

        self.xbin, self.ybin = np.meshgrid(xbin, ybin)

        self.xbin = np.ravel(self.xbin)

        self.ybin = np.ravel(self.ybin)

        # initialise the flattened velocity array

        self.vel_flat = []

        # now that we have the xbins and ybins, these are the coordinates
        # at which we want to evaluate the velocity in the
        # fit_kinematic_pa method. We can loop around the
        # the coordinates and create a flattened array
        # so that the velocity data values correspond to the bins

        for x, y in zip(self.xbin, self.ybin):

            # evaluate the velocity point

            self.vel_flat.append(self.vel_data[x][y])

        # make sure that vel_flat is a numpy array 

        self.vel_flat = np.array(self.vel_flat)

        # now can subtract the central positions from 
        # xbins and ybins 
        self.xbin = self.xbin - self.c_rot_x
        self.ybin = self.ybin - self.c_rot_y


        # print (self.vel_flat)

        # print (len(self.vel_flat))
        # print (len(self.xbin))
        # print (len(self.ybin))
        # print (len(self.xbin) * len(self.ybin))
        # sauron colour dictionary
        self._cdict = {'red':((0.000,   0.01,   0.01),
                             (0.170,   0.0,    0.0),
                             (0.336,   0.4,    0.4),
                             (0.414,   0.5,    0.5),
                             (0.463,   0.3,    0.3),
                             (0.502,   0.0,    0.0),
                             (0.541,   0.7,    0.7),
                             (0.590,   1.0,    1.0),
                             (0.668,   1.0,    1.0),
                             (0.834,   1.0,    1.0),
                             (1.000,   0.9,    0.9)),
                    'green':((0.000,   0.01,   0.01), 
                             (0.170,   0.0,    0.0),
                             (0.336,   0.85,   0.85),
                             (0.414,   1.0,    1.0),
                             (0.463,   1.0,    1.0),
                             (0.502,   0.9,    0.9),
                             (0.541,   1.0,    1.0),
                             (0.590,   1.0,    1.0),
                             (0.668,   0.85,   0.85),
                             (0.834,   0.0,    0.0),
                             (1.000,   0.9,    0.9)),
                     'blue':((0.000,   0.01,   0.01),
                             (0.170,   1.0,    1.0),
                             (0.336,   1.0,    1.0),
                             (0.414,   1.0,    1.0),
                             (0.463,   0.7,    0.7),
                             (0.502,   0.0,    0.0),
                             (0.541,   0.0,    0.0),
                             (0.590,   0.0,    0.0),
                             (0.668,   0.0,    0.0),
                             (0.834,   0.0,    0.0),
                             (1.000,   0.9,    0.9))
                      }

        self.sauron = colors.LinearSegmentedColormap('sauron', self._cdict)


    def _rotate_points(self, x, y, ang):
        """
        Rotates points counter-clockwise by an angle ANG in degrees.
        Michele cappellari, Paranal, 10 November 2013
        
        """
        theta = np.radians(ang - 90.)
        xNew = x*np.cos(theta) - y*np.sin(theta)
        yNew = x*np.sin(theta) + y*np.cos(theta)
        return xNew, yNew


    def symmetrize_velfield(self, xbin, ybin, velBin, sym=2, pa=90.):
        """
        This routine generates a bi-symmetric ('axisymmetric') 
        version of a given set of kinematical measurements.
        PA: is the angle in degrees, measured counter-clockwise,
          from the vertical axis (Y axis) to the galaxy major axis.
        SYM: by-simmetry: is 1 for (V,h3,h5) and 2 for (sigma,h4,h6)

        """

        xbin, ybin, velBin = map(np.asarray, [xbin, ybin, velBin])
        x, y = self._rotate_points(xbin, ybin, -pa)  # Negative PA for counter-clockwise
        
        xyIn = np.column_stack([x, y])
        xout = np.hstack([x,-x, x,-x])
        yout = np.hstack([y, y,-y,-y])
        xyOut = np.column_stack([xout, yout])
        velOut = interpolate.griddata(xyIn, velBin, xyOut)
        velOut = velOut.reshape(4, xbin.size)
        if sym == 1:
            velOut[[1, 3], :] *= -1.
        velSym = np.nanmean(velOut, axis=0)
        
        # print ('This is the symmetrised vel_field: %s' % np.nanmean(velSym))
        return velSym

    def plot_velfield(self, x, y, vel, vmin=None, vmax=None, ncolors=64, nodots=False,
                      colorbar=False, label=None, flux=None, fixpdf=False,
                      nticks=7, **kwargs):

        if vmin is None:
            vmin = -1000

        if vmax is None:
            vmax = 1000

        x, y, vel = map(np.ravel, [x, y, vel])
        levels = np.linspace(vmin, vmax, ncolors)

        ax = plt.gca()
        cs = ax.tricontourf(x, y, vel.clip(vmin, vmax), levels=levels,
                           cmap=kwargs.get("cmap", self.sauron))

        ax.axis('image')  # Equal axes and no rescaling
        ax.minorticks_on()
        ax.tick_params(length=10, which='major')
        ax.tick_params(length=5, which='minor')

        if flux is not None:
            ax.tricontour(x, y, -2.5*np.log10(flux/np.max(flux).ravel()),
                          levels=np.arange(20), colors='k') # 1 mag contours

        if fixpdf:  # remove white contour lines in PDF at expense of larger file size
            ax.tricontour(x, y, vel.clip(vmin, vmax), levels=levels, zorder=0,
                          cmap=kwargs.get("cmap", self.sauron))

        if not nodots:
            ax.plot(x, y, '.k', markersize=kwargs.get("markersize", 3))

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
            cbar = plt.colorbar(cs, cax=cax, ticks=ticks)
            if label:
                cbar.set_label(label)


        return cs

    def display_pixels(self, x, y, val, pixelsize=None, angle=None, **kwargs):
        """
        Display vectors of square pixels at coordinates (x,y) coloured with "val".
        An optional rotation around the origin can be applied to the whole image.

        This routine is designed to be fast even with large images and to produce
        minimal file sizes when the output is saved in a vector format like PDF.

        """
        if pixelsize is None:
            pixelsize = np.min(distance.pdist(np.column_stack([x, y])))

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        nx = round((xmax - xmin)/pixelsize) + 1
        ny = round((ymax - ymin)/pixelsize) + 1
        j = np.round((x - xmin)/pixelsize).astype(int)
        k = np.round((y - ymin)/pixelsize).astype(int)
        mask = np.ones((nx, ny), dtype=bool)
        img = np.empty((nx, ny))
        mask[j, k] = 0
        img[j, k] = val
        img = np.ma.masked_array(img, mask)

        ax = plt.gca()

        if (angle is None) or (angle == 0):

            f = ax.imshow(np.rot90(img), interpolation='none',
                          cmap=kwargs.get("cmap", self.sauron),
                          extent=[xmin-pixelsize/2, xmax+pixelsize/2,
                                  ymin-pixelsize/2, ymax+pixelsize/2])

        else:

            x, y = np.ogrid[xmin-pixelsize/2 : xmax+pixelsize/2 : (nx+1)*1j,
                            ymin-pixelsize/2 : ymax+pixelsize/2 : (ny+1)*1j]
            ang = np.radians(angle)
            x, y = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)

            mask1 = np.ones_like(x, dtype=bool)
            mask1[:-1, :-1] *= mask  # Flag the four corners of the mesh
            mask1[:-1, 1:] *= mask
            mask1[1:, :-1] *= mask
            mask1[1:, 1:] *= mask
            x = np.ma.masked_array(x, mask1)  # Mask is used for proper plot range
            y = np.ma.masked_array(y, mask1)

            f = ax.pcolormesh(x, y, img, cmap=kwargs.get("cmap", self.sauron))
            ax.axis('image')

        ax.minorticks_on()
        ax.tick_params(length=10, width=1, which='major')
        ax.tick_params(length=5, width=1, which='minor')

        return f

    def display_bins(self, x, y, binNum, velBin):
        """
        NAME:
            display_bins()
            
        AUTHOR:
            Michele Cappellari, University of Oxford
            cappellari_at_astro.ox.ac.uk

        PURPOSE:
            This simple routine illustrates how to display a Voronoi binned map.
            
        INPUTS:
            (x, y): (length npix) Coordinates of the original spaxels before binning;
            binNum: (length npix) Bin number corresponding to each (x, y) pair,
                    as provided in output by the voronoi_2d_binning() routine;
            velBin: (length nbins) Quantity associated to each bin, resulting
                    e.g. from the kinematic extraction from the binned spectra.
                  
        MODIFICATION HISTORY:
            V1.0.0: Michele Cappellari, Oxford, 15 January 2015          
        
        """
        npix = len(binNum)
        if (npix != len(x)) or (npix != len(y)):
            raise ValueError('The vectors (x, y, binNum) must have the same size')
            
        f = self.display_pixels(x, y, velBin[binNum])
        
        return f

    def fit_kinematic_pa(self, debug=False, nsteps=361, 
                         quiet=False, plot=True, dvel=None):
        """
             NAME:
           FIT_KINEMATIC_PA

         PURPOSE:
           Determine the global kinematic position angle of a
           galaxy with the method described in Appendix C of
           Krajnovic, Cappellari, de Zeeuw, & Copin 2006, MNRAS, 366, 787


         INPUT PARAMETERS:
           XBIN, YBIN: vectors with the coordinates of the bins (or pixels)
               measured from the centre of rotation (typically the galaxy centre).
             - IMPORTANT: The routine will not give meaningful output unless 
               (X,Y)=(0,0) is an estimate of the centre of rotation.        
           VEL: measured velocity at the position (XBIN,YBIN). 
             - IMPORTANT: An estimate of the systemic velocity has to be already 
               subtracted from this velocity [e.g. VEL = VEL - median(VEL)]. 
               The routine will then provide in the output VELSYST a correction 
               to be added to the adopted systemic velocity.

         INPUT KEYWORDS:
           NSTEPS: number of steps along which the angle is sampled.
               Default is 361 steps which gives a 0.5 degr accuracy.
               Decrease this number to limit computation time during testing.

         OUTPUT PARAMETER:
           ANGLEBEST: kinematical PA. Note that this is the angle along which
               |Vel| is maximum (note modulus!). If one reverses the sense of
               rotation in a galaxy ANGLEBEST does not change. The sense of
               rotation can be trivially determined by looking at the map of Vel.
           ANGLEERROR: corresponding error to assign to ANGLEBEST.
           VELSYST: Best-fitting correction to the adopted systemic velocity 
               for the galaxy.
             - If the median was subtracted to the input velocity VEL before 
               the PA fit, then the corrected systemnic velocity will be 
               median(VEL)+VELSYST.

         REQUIRED ROUTINES:
           The following five additional routines are needed:
           - 1. CAP_SYMMETRIZE_VELFIELD and 2. CAP_RANGE: by Michele Cappellari
             (included in this FIT_KINEMATIC_PA distribution)
           - 3. SAURON_COLORMAP and 4. PLOT_VELFIELD: can be obtained from:
             http://purl.org/cappellari/idl#binning
           - 5. SIGRANGE: from IDL astro library http://idlastro.gsfc.nasa.gov/

        """
        vel = copy(self.vel_flat)
        if dvel is None:
            dvel = vel*0 + 10.0 # Adopt here constant 10 km/s errors!
        
        nbins = self.xbin.size
        n = nsteps
        angles = np.linspace(0, 180, n) # 0.5 degrees steps by default
        chi2 = np.empty_like(angles)
        for j, ang in enumerate(angles):
            velSym = self.symmetrize_velfield(self.xbin, self.ybin, vel, sym=1, pa=ang)
            chi2[j] = np.nansum(((vel-velSym)/dvel)**2)
            if debug:
                print('Ang, chi2/DOF:', ang, chi2[j]/nbins)
                self.plot_velfield(self.xbin, self.ybin, velSym)
                plt.pause(0.01)
        k = np.argmin(chi2)
        angBest = angles[k]
        
        # Compute fit at the best position
        #
        velSym = self.symmetrize_velfield(self.xbin, self.ybin, vel, sym=1, pa=angBest)
        if angBest < 0:
            angBest += 180
        
        # 3sigma confidence limit, including error on chi^2
        #
        f = chi2 - chi2[k] <= 9 + 3*np.sqrt(2*nbins)
        if f.sum():
            angErr = (np.max(angles[f]) - np.min(angles[f]))/2.0
            if angErr >= 45:
                good = np.degrees(np.arctan(np.tan(np.radians(angles[f]))))
                angErr = (np.max(good) - np.min(good))/2.0
        else:
            angErr = max(0.5, (angles[1]-angles[0])/2.0)
        
        # angErr = angErr.clip(max(0.5, (angles[1]-angles[0])/2.0)) # Force errors to be larger than 0.5 deg
        vSyst = np.nanmedian(vel - velSym)
        
        if not quiet:
            print('  Kin PA:', angBest, ' +/- ', angErr, ' (3*sigma error)')
            print('Velocity Offset:', vSyst)
        
        # Plot results
        #
        if plot:    
        
            mn, mx = stats.scoreatpercentile(velSym, [2.5, 97.5])
            mx = np.nanmin([mx, -mn])

            plt.subplot(121)
            self.plot_velfield(self.xbin, self.ybin, velSym, vmin=-mx, vmax=mx) 
            plt.title('Symmetrized')
            
            # debugging 
            print (velSym)
            print (np.nanmin(vel - vSyst), np.nanmax(vel - vSyst))

            plt.subplot(122)
            self.plot_velfield(self.xbin, self.ybin, velSym, vmin=-mx, vmax=mx) 
            plt.title('Data and best PA')
            rad = np.sqrt(np.max(self.xbin**2 + self.ybin**2))
            ang = [0,np.pi] + np.radians(angBest)
            plt.plot(rad*np.cos(ang), rad*np.sin(ang), '--', linewidth=3) # Zero-velocity line
            plt.plot(-rad*np.sin(ang), rad*np.cos(ang), linewidth=3) # Major axis PA

        plt.show()
        return angBest, angErr, vSyst


    def disk_function(self,
                      theta,
                      xpos,
                      ypos):
        """
        Def: Function to calculate disk velocity given input values.
        Note that all angles must be given in radians
        """
        # unpack the parameters

        xcen, ycen, inc, pa, rt, vasym = theta

        # look at the difference between central pixel and pixel
        # being modelled

        diff_x = (xcen - xpos) * 1.0

        diff_y = (ycen - ypos) * 1.0

        # print diff_x, diff_y

        # calculate the pixel angle

        if diff_y == 0 and diff_x != 0:

            pixel_angle = np.arctan(np.sign(diff_x)*np.inf)

            # print 'This is the pixel angle %s' % pixel_angle

        elif diff_y == 0 and diff_x == 0:

            # print 'In the middle'

            pixel_angle = 0.0

        else:

            # print 'computing pixel angle'

            pixel_angle = np.arctan(diff_x / diff_y)

            # print 'pixel angle %s' % (pixel_angle * 180 / np.pi)

        # work out phi which is the overall angle between
        # the spaxel being modelled and the central spaxel/position angle
        # this involves summing with a rotation angle which depends on
        # the spaxel quadrant

        if diff_x >= 0 and diff_y >= 0 and not(diff_x == 0 and diff_y == 0):

            # print 'top left'
            # we're in the upper left quadrant, want rot to be 270

            rot = 3 * np.pi / 2

        elif diff_x >= 0 and diff_y < 0:

            # print 'top right'

            # we're in the upper right quandrant, want rot to be 90

            rot = np.pi / 2

        elif diff_x < 0 and diff_y < 0:

            # print 'lower right'

            # we're in the lower right quadrant

            rot = np.pi / 2

        elif diff_x < 0 and diff_y >= 0:

            # print 'lower left'

            # we're in the lower left quadrant

            rot = 3 * np.pi / 2

        elif diff_x == 0 and diff_y == 0:

            # print 'middle'

            # we're in the middle

            rot = pa

        phi = pixel_angle - pa + rot

    #    print 'differences: %s %s' % (diff_x, diff_y)
    #    print 'pixel angle %s' % (pixel_angle * 180 / np.pi)
    #    print 'position angle %s' % (pa * 180 / np.pi)
    #    print 'rotation angle %s' % (rot * 180 / np.pi)
    #    print 'overall angle %s' % (phi * 180 / np.pi)
    #    print 'cosine of angle %s' % (np.cos(phi))

        r = np.sqrt(diff_x*diff_x + diff_y*diff_y)

        vel = np.cos(phi) * np.sin(inc) * (2 / np.pi) * vasym * np.arctan(r / rt)

        # print vel, xpix, ypix

        return vel

    def grid(self):

        """
        Def: return an empty grid with the specified dimensions
        """

        # create a 1D arrays of length dim_x * dim_y containing the 
        # spaxel coordinates

        xbin = np.arange(0, self.xpix, 1)

        ybin = np.arange(0, self.ypix, 1)

        ybin, xbin = np.meshgrid(ybin, xbin)

        xbin = np.ravel(xbin)

        ybin = np.ravel(ybin)

        return np.array(xbin) * 1.0, np.array(ybin) * 1.0

    def grid_factor(self,
                    res_factor):

        """
        Def: return an empty grid with 10 times spatial resolution of
        the velocity data
        """

        # create a 1D arrays of length dim_x * dim_y containing the
        # spaxel coordinates

        xbin = np.arange(0, self.xpix * res_factor, 1)

        ybin = np.arange(0, self.ypix * res_factor, 1)

        ybin, xbin = np.meshgrid(ybin, xbin)

        xbin = np.ravel(xbin)

        ybin = np.ravel(ybin)

        return np.array(xbin) * 1.0, np.array(ybin) * 1.0

    def compute_model_grid(self,
                           theta,
                           redshift,
                           wave_array,
                           xcen,
                           ycen,
                           seeing,
                           sersic_n,
                           sigma,
                           pix_scale,
                           psf_factor,
                           sersic_factor,
                           m_factor,
                           smear=False):

        """
        Def:
        Use the grid function to construct a basis for the model.
        Then apply the disk function to each spaxel in the basis
        reshape back to 2d array and plot the model velocity
        """

        xbin, ybin = self.grid_factor(m_factor)

        # setup list to house the velocity measurements

        vel_array = []

        # compute the model at each spaxel location

        for xpos, ypos in zip(xbin, ybin):

            # run the disk function

            vel_array.append(self.disk_function(theta,
                                                xpos * m_factor,
                                                ypos * m_factor))

        # create numpy array from the vel_array list

        vel_array = np.array(vel_array)

        # reshape back to the chosen grid dimensions

        vel_2d = vel_array.reshape((self.xpix * m_factor,
                                    self.ypix * m_factor))

        if float(m_factor) != 1.0:

            vel_2d = psf.bin_by_factor(vel_2d,
                                       m_factor) 

        # computationally expensive

        pa = theta[3]

        if smear:

            vel_2d, sig_2d = psf.cube_blur(vel_2d,
                                           redshift,
                                           wave_array,
                                           xcen,
                                           ycen,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           sersic_factor,
                                           pa,
                                           sigma,
                                           sersic_n)

        return vel_2d

    def lnlike(self, 
               theta,
               seeing,
               pix_scale,
               psf_factor,
               smear=False):
        """
        Def: Return the log likelihood for the velocity field function.
        All that has to be done is to compute the model in a grid the same size
        as the data and then plug into the standard likelihood formula.

        Input:
                vel_data - the actual velocity field unicodedata
                vel_errors - the velocity field error grid
                theta - list of parameter values to be fed into the model

        Output:
                some single numerical value for the log likelihood
        """
        # sometimes nice to see what parameters are being tried in the
        # MCMC step

        # print theta

        # compute the model grid

        if smear:

            model = self.compute_model_grid(theta,
                                            seeing,
                                            pix_scale,
                                            psf_factor,
                                            smear=True)
        else:

            model = self.compute_model_grid(theta,
                                            seeing,
                                            pix_scale,
                                            psf_factor)

        # find the grid of inverse sigma values

        inv_sigma2 = 1.0 / (self.error_data * self.error_data)

        ans = -0.5 * (np.nansum((self.vel_data - model)*(self.vel_data - model) *
                                inv_sigma2 - np.log(inv_sigma2)))

        # print ans

        return ans

    def lnprior(self,
                theta):

        """
        Def:
        Set an uninformative prior distribution for the parameters in the model
        """

        xcen, ycen, inc, pa, rt, vasym = theta

        if 5 < xcen < 30.0 and \
           5 < ycen < 30.0 and \
           0.0 < inc < np.pi / 2.0 and \
           0 < pa < 2 * np.pi and \
           1.0 < rt < 5.0 and \
           0 < vasym < 350:

            return 0.0

        return -np.inf

    def lnprob(self,
               theta,
               seeing,
               pix_scale,
               psf_factor,
               smear=False):

        lp = self.lnprior(theta)

        if not np.isfinite(lp):

            return -np.inf

        return lp + self.lnlike(theta,
                                seeing,
                                pix_scale,
                                psf_factor,
                                smear)

    def run_emcee(self,
                  theta,
                  nsteps,
                  nwalkers,
                  burn_no,
                  seeing,
                  pix_scale,
                  psf_factor,
                  smear=False):

        ndim = len(theta)

        pos = [theta + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.lnprob,
                                        args=[seeing,
                                              pix_scale,
                                              psf_factor,
                                              smear])

        for i, (pos, lnp, state) in enumerate(sampler.sample(pos,
                                                             iterations=nsteps)):


            stdout.write("\rObject %s %.1f%% complete" % (self.gal_name[:-15],
                                                        100 * float(i + 1) / nsteps))
            stdout.flush()

        stdout.write('\n')

        samples = sampler.chain[:, burn_no:, :].reshape((-1, ndim))

        fig = corner.corner(samples,
                            labels=["$xcen$",
                                    "$ycen$",
                                    "$inc$",
                                    "$pa$",
                                    "rt",
                                    "vasym"],
                            truths=theta)

        fig.savefig('%s_corner_plot.png' % self.fileName[:-5])

        plt.show()

        # print samples
        # going to save pickled versions of the chain and the lnprobability
        # so that these can be accessed again later if necessary

        if os.path.isfile(self.chain_name):

            os.system('rm %s' % self.chain_name)

        if os.path.isfile(self.ln_p_name):

            os.system('rm %s' % self.ln_p_name)

        chain_file = open(self.chain_name, 'w')

        pickle.dump(sampler.chain, chain_file)

        chain_file.close()

        ln_p_file = open(self.ln_p_name, 'w')

        pickle.dump(sampler.lnprobability, ln_p_file)

        ln_p_file.close()

        # now use the helper function below to open up
        # the pickled files and write to params file

        self.write_params(burn_no)

        # set a variable to the log probability value

    def write_params(self,
                     burn_no):

        """
        Def:
        Helper function to open up the pickled chain and lnp files
        and write the maximum likelihood, 50th percentile, 16th per and 84th
        per parameters to file for ease of application later on

        Input:
                burn_no - number of entries (steps) to burn from the chain
        """

        chain = pickle.load(open(self.chain_name, 'r'))

        lnp = pickle.load(open(self.ln_p_name, 'r'))

        samples = chain[:, burn_no:, :].reshape((-1, chain.shape[2]))

        # initialise the parameter names

        param_names = ['type',
                       'centre_x',
                       'centre_y',
                       'inclination',
                       'position_angle',
                       'Flattening_Radius',
                       'Flattening_Velocity']

        # find the max likelihood parameters

        max_p = np.unravel_index(lnp.argmax(), lnp.shape)

        max_params = chain[max_p[0], max_p[1], :]

        x_mcmc, y_mcmc, i_mcmc, \
            pa_mcmc, rt_mcmc, va_mcmc \
            = zip(*np.percentile(samples, [16, 50, 84],
                  axis=0))

        param_file = self.fileName[:-5] + '_params.txt'

        if os.path.isfile(param_file):

            os.system('rm %s' % param_file)

        # write all of these values to file

        with open(param_file, 'a') as f:

            for item in param_names:

                f.write('%s\t' % item)

            f.write('\nMAX_lnp:\t')

            for item in max_params:

                f.write('%s\t' % item)

            f.write('\n50th_lnp:\t%s\t%s\t%s\t%s\t%s\t%s\t' % (x_mcmc[1],
                                                               y_mcmc[1],
                                                               i_mcmc[1],
                                                               pa_mcmc[1],
                                                               rt_mcmc[1],
                                                               va_mcmc[1]))

            f.write('\n16th_lnp:\t%s\t%s\t%s\t%s\t%s\t%s\t' % (x_mcmc[0],
                                                               y_mcmc[0],
                                                               i_mcmc[0],
                                                               pa_mcmc[0],
                                                               rt_mcmc[0],
                                                               va_mcmc[0]))

            f.write('\n84th_lnp:\t%s\t%s\t%s\t%s\t%s\t%s\t' % (x_mcmc[2],
                                                               y_mcmc[2],
                                                               i_mcmc[2],
                                                               pa_mcmc[2],
                                                               rt_mcmc[2],
                                                               va_mcmc[2]))


    def plot_comparison(self,
                        seeing,
                        pix_scale,
                        psf_factor,
                        smear=False):

        """
        Def:
        Plot the best fitting model alongside the original velocity field
        with position angle and morphological angle also plotted

        Input:
                theta - the now best fit set of parameters
                vel_data - the velocity field unicodedata
                vel_errors - the velocity field errors

        """

        # load in the file

        param_file = np.genfromtxt(self.param_file)

        theta_max = param_file[1][1:]

        theta_50 = param_file[2][1:]

        theta_16 = param_file[3][1:]

        theta_84 = param_file[4][1:]

        # compute the model grid with the specified parameters


        model_max = self.compute_model_grid(theta_max,
                                            seeing,
                                            pix_scale,
                                            psf_factor,
                                            smear)

        model_50 = self.compute_model_grid(theta_50,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        model_16 = self.compute_model_grid(theta_16,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        model_84 = self.compute_model_grid(theta_84,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        # only want to see the evaluated model at the grid points
        # where the data is not nan. Loop round the data and create
        # a mask which can multiply the model

        mask_array = np.empty(shape=(self.xpix, self.ypix))

        for i in range(0, self.xpix):

            for j in range(0, self.ypix):

                if np.isnan(self.vel_data[i][j]):

                    mask_array[i][j] = np.nan

                else:

                    mask_array[i][j] = 1.0

        # take product of model and mask_array to return new data

        trunc_model_max = mask_array * model_max

        trunc_model_50 = mask_array * model_50

        trunc_model_16 = mask_array * model_16

        trunc_model_84 = mask_array * model_84

        # plot the results

        vel_min, vel_max = np.nanpercentile(self.vel_data,
                                            [5.0, 95.0])

        mod_min, mod_max = np.nanpercentile(trunc_model_max,
                                            [5.0, 95.0])

        plt.close('all')

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        im = ax[0].imshow(self.vel_data,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[0].set_title('[OIII] Velocity Data')

        im = ax[1].imshow(trunc_model_50,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[1].set_title('[OIII] Velocity Model')

        # plt.show()

        fig.savefig('%s_model_comparison.png' % self.fileName[:-5])

        plt.close('all')

    def extract_in_apertures(self,
                             r_aper,
                             d_aper,
                             seeing,
                             pix_scale,
                             psf_factor,
                             smear=False):

        """
        Def: Extract the velocity field along the kinematic axis returned by the
        model fitting in both the data and the model for comparison. The model
        will show a perfect arctangent function.

        Input:
                theta - array of best fitting model parameter values
                model_data - best fit model computed from the compute_model_grid
                vel_data - array containing the actual velocity data
                r_aper - aperture size in pixels to use for each aperture
                d_aper - distance spacing between apertures
        Output:
                1D arrays containing the extracted model and real velocity fields
                along the kinematic major axis
        """

        # assign the best fit parameters to variables from the theta array

        # load in the file

        param_file = np.genfromtxt(self.param_file)

        theta_max = param_file[1][1:]

        pa_max = theta_max[3]

        xcen_max = theta_max[0]

        ycen_max = theta_max[1]

        theta_50 = param_file[2][1:]

        pa_50 = theta_50[3]

        xcen_50 = theta_50[0]

        ycen_50 = theta_50[1]

        inc_50 = theta_50[2]

        va_50 = theta_50[5]

        theta_16 = param_file[3][1:]

        pa_16 = theta_16[3]

        xcen_16 = theta_16[0]

        ycen_16 = theta_16[1]

        inc_16 = theta_16[2]

        theta_84 = param_file[4][1:]

        pa_84 = theta_84[3]

        xcen_84 = theta_84[0]

        ycen_84 = theta_84[1]

        inc_84 = theta_84[2]

        # compute the model grid with the specified parameters

        model_max = self.compute_model_grid(theta_max,
                                            seeing,
                                            pix_scale,
                                            psf_factor,
                                            smear)

        model_50 = self.compute_model_grid(theta_50,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        model_16 = self.compute_model_grid(theta_16,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        model_84 = self.compute_model_grid(theta_84,
                                           seeing,
                                           pix_scale,
                                           psf_factor,
                                           smear)

        # initialise the list of aperture positions with the xcen and ycen

        positions_max = []

        positions_50 = []

        positions_16 = []

        positions_84 = []

        # first job is to compute the central locations of the apertures
        # do this by fixing the distance along the KA between aperture centres

        xdim = self.xpix - 2

        ydim = self.ypix - 2

        # find the steps along the KA with which to increment

        x_inc_max = d_aper * abs(np.sin((np.pi / 2.0) - pa_max))

        y_inc_max = d_aper * abs(np.cos((np.pi / 2.0) - pa_max))

        x_inc_50 = d_aper * abs(np.sin((np.pi / 2.0) - pa_50))

        y_inc_50 = d_aper * abs(np.cos((np.pi / 2.0) - pa_50))

        x_inc_16 = d_aper * abs(np.sin((np.pi / 2.0) - pa_16))

        y_inc_16 = d_aper * abs(np.cos((np.pi / 2.0) - pa_16))

        x_inc_84 = d_aper * abs(np.sin((np.pi / 2.0) - pa_84))

        y_inc_84 = d_aper * abs(np.cos((np.pi / 2.0) - pa_84))

        # now find the sequence of aperture centres up until the boundaries
        # this is tricky - depending on the PA need to increase and decrease
        # both x and y together, or increase one and decrease the other

        # statements for the maximum likelihood position angle

        if 0 < pa_max < np.pi / 2.0 or np.pi < pa_max < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_max = xcen_max + x_inc_max

            new_y_max = ycen_max - y_inc_max

            # while loop until xdim is breached or 0 is breached for y

            while new_x_max < xdim and new_y_max > 2:

                # append aperture centre to the positions array

                positions_max.append((new_y_max, new_x_max))

                new_x_max += x_inc_max

                new_y_max -= y_inc_max

                # print new_x_max, new_y_max

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_max = positions_max[::-1]

            positions_max.append((ycen_max, xcen_max))

            # now go in the other direction

            new_x_max = xcen_max - x_inc_max

            new_y_max = ycen_max + y_inc_max

            # while loop until xdim is breached or 0 is breached for y

            while new_x_max > 2 and new_y_max < ydim:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max -= x_inc_max

                new_y_max += y_inc_max

                # print new_x, new_y_max

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_max = xcen_max - x_inc_max

            new_y_max = ycen_max - y_inc_max

            # while loop until xdim is 2 or ydim is 2

            while new_x_max > 2 and new_y_max > 2:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max -= x_inc_max

                new_y_max -= y_inc_max

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_max = positions_max[::-1]

            positions_max.append((ycen_max, xcen_max))

            # now go in the other direction

            new_x_max = xcen_max + x_inc_max

            new_y_max = ycen_max + y_inc_max

            # while loop until xdim is breached or ydim is breached

            while new_x_max < xdim and new_y_max < ydim:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max += x_inc_max

                new_y_max += y_inc_max

        # statements for the 50th percentile position angle

        if 0 < pa_50 < np.pi / 2.0 or np.pi < pa_50 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_50 = xcen_50 + x_inc_50

            new_y_50 = ycen_50 - y_inc_50

            # while loop until xdim is breached or 0 is breached for y

            while new_x_50 < xdim and new_y_50 > 2:

                # append aperture centre to the positions array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 += x_inc_50

                new_y_50 -= y_inc_50

                # print new_x_50, new_y_50

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_50 = positions_50[::-1]

            positions_50.append((ycen_50, xcen_50))

            # now go in the other direction

            new_x_50 = xcen_50 - x_inc_50

            new_y_50 = ycen_50 + y_inc_50

            # while loop until xdim is breached or 0 is breached for y

            while new_x_50 > 2 and new_y_50 < ydim:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 -= x_inc_50

                new_y_50 += y_inc_50

                # print new_x, new_y_50

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_50 = xcen_50 - x_inc_50

            new_y_50 = ycen_50 - y_inc_50

            # while loop until xdim is 2 or ydim is 2

            while new_x_50 > 2 and new_y_50 > 2:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 -= x_inc_50

                new_y_50 -= y_inc_50

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_50 = positions_50[::-1]

            positions_50.append((ycen_50, xcen_50))

            # now go in the other direction

            new_x_50 = xcen_50 + x_inc_50

            new_y_50 = ycen_50 + y_inc_50

            # while loop until xdim is breached or ydim is breached

            while new_x_50 < xdim and new_y_50 < ydim:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 += x_inc_50

                new_y_50 += y_inc_50

        # statements for the 16th percentile position angle

        if 0 < pa_16 < np.pi / 2.0 or np.pi < pa_16 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_16 = xcen_16 + x_inc_16

            new_y_16 = ycen_16 - y_inc_16

            # while loop until xdim is breached or 0 is breached for y

            while new_x_16 < xdim and new_y_16 > 2:

                # append aperture centre to the positions array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 += x_inc_16

                new_y_16 -= y_inc_16

                # print new_x_16, new_y_16

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_16 = positions_16[::-1]

            positions_16.append((ycen_16, xcen_16))

            # now go in the other direction

            new_x_16 = xcen_16 - x_inc_16

            new_y_16 = ycen_16 + y_inc_16

            # while loop until xdim is breached or 0 is breached for y

            while new_x_16 > 2 and new_y_16 < ydim:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 -= x_inc_16

                new_y_16 += y_inc_16

                # print new_x, new_y_16

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_16 = xcen_16 - x_inc_16

            new_y_16 = ycen_16 - y_inc_16

            # while loop until xdim is 2 or ydim is 2

            while new_x_16 > 2 and new_y_16 > 2:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 -= x_inc_16

                new_y_16 -= y_inc_16

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_16 = positions_16[::-1]

            positions_16.append((ycen_16, xcen_16))

            # now go in the other direction

            new_x_16 = xcen_16 + x_inc_16

            new_y_16 = ycen_16 + y_inc_16

            # while loop until xdim is breached or ydim is breached

            while new_x_16 < xdim and new_y_16 < ydim:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 += x_inc_16

                new_y_16 += y_inc_16

        # statements for the 84th percenntile position angle

        if 0 < pa_84 < np.pi / 2.0 or np.pi < pa_84 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_84 = xcen_84 + x_inc_84

            new_y_84 = ycen_84 - y_inc_84

            # while loop until xdim is breached or 0 is breached for y

            while new_x_84 < xdim and new_y_84 > 2:

                # append aperture centre to the positions array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 += x_inc_84

                new_y_84 -= y_inc_84

                # print new_x_84, new_y_84

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_84 = positions_84[::-1]

            positions_84.append((ycen_84, xcen_84))

            # now go in the other direction

            new_x_84 = xcen_84 - x_inc_84

            new_y_84 = ycen_84 + y_inc_84

            # while loop until xdim is breached or 0 is breached for y

            while new_x_84 > 2 and new_y_84 < ydim:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 -= x_inc_84

                new_y_84 += y_inc_84

                # print new_x, new_y_84

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_84 = xcen_84 - x_inc_84

            new_y_84 = ycen_84 - y_inc_84

            # while loop until xdim is 2 or ydim is 2

            while new_x_84 > 2 and new_y_84 > 2:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 -= x_inc_84

                new_y_84 -= y_inc_84

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_84 = positions_84[::-1]

            positions_84.append((ycen_84, xcen_84))

            # now go in the other direction

            new_x_84 = xcen_84 + x_inc_84

            new_y_84 = ycen_84 + y_inc_84

            # while loop until xdim is breached or ydim is breached

            while new_x_84 < xdim and new_y_84 < ydim:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 += x_inc_84

                new_y_84 += y_inc_84

        # construct the x_axis for the aperture extraction plot

        x_max_array = []

        for entry in positions_max:

            x_max_array.append(entry[1])

        x_max_array = np.array(x_max_array) - xcen_max

        # print x_max_array

        x_max_index = np.where(x_max_array == 0.0)[0]

        x_max = np.linspace(-1. * d_aper * x_max_index,
                            d_aper * (len(x_max_array) - x_max_index - 1),
                            num=len(x_max_array))

        x_max = x_max * pix_scale

        # print 'This is x_max: %s' % x_max

        x_50_array = []

        for entry in positions_50:

            x_50_array.append(entry[1])

        x_50_array = np.array(x_50_array) - xcen_50

        x_50_index = np.where(x_50_array == 0.0)[0]

        x_50 = np.linspace(-1. * d_aper * x_50_index,
                            d_aper * (len(x_50_array) - x_50_index - 1),
                            num=len(x_50_array))

        x_50 = x_50 * pix_scale

        x_16_array = []

        for entry in positions_16:

            x_16_array.append(entry[1])

        x_16_array = np.array(x_16_array) - xcen_16

        x_16_index = np.where(x_16_array == 0.0)[0]

        x_16 = np.linspace(-1. * d_aper * x_16_index,
                            d_aper * (len(x_16_array) - x_16_index - 1),
                            num=len(x_16_array))

        x_16 = x_16 * pix_scale

        x_84_array = []

        for entry in positions_84:

            x_84_array.append(entry[1])

        x_84_array = np.array(x_84_array) - xcen_84

        x_84_index = np.where(x_84_array == 0.0)[0]

        x_84 = np.linspace(-1. * d_aper * x_84_index,
                            d_aper * (len(x_84_array) - x_84_index - 1),
                            num=len(x_84_array))

        x_84 = x_84 * pix_scale

        # positions array should now be populated with all of the apertures

        # print positions

        # now perform aperture photometry on the model data to check that this
        # actually works. Remember that the velocity computed for each
        # aperture will be the sum returned divided by the area

        pixel_area = np.pi * r_aper * r_aper

        # the max likelihood extraction parameters

        apertures_max = CircularAperture(positions_max, r=r_aper)

        mod_phot_table_max = aperture_photometry(model_max, apertures_max)

        real_phot_table_max = aperture_photometry(self.vel_data, apertures_max)

        real_error_table_max = aperture_photometry(self.error_data, apertures_max)

        sig_table_max = aperture_photometry(self.sig_data, apertures_max)

        sig_error_table_max = aperture_photometry(self.sig_error_data, apertures_max)

        mod_velocity_values_max = mod_phot_table_max['aperture_sum'] / pixel_area

        real_velocity_values_max = real_phot_table_max['aperture_sum'] / pixel_area

        real_error_values_max = real_error_table_max['aperture_sum'] / pixel_area

        sig_values_max = sig_table_max['aperture_sum'] / pixel_area

        sig_error_values_max = sig_error_table_max['aperture_sum'] / pixel_area

        # the 50th percentile extraction parameters

        apertures_50 = CircularAperture(positions_50, r=r_aper)

        mod_phot_table_50 = aperture_photometry(model_50, apertures_50)

        real_phot_table_50 = aperture_photometry(self.vel_data, apertures_50)

        real_error_table_50 = aperture_photometry(self.error_data, apertures_50)

        sig_table_50 = aperture_photometry(self.sig_data, apertures_50)

        sig_error_table_50 = aperture_photometry(self.sig_error_data, apertures_50)

        mod_velocity_values_50 = mod_phot_table_50['aperture_sum'] / pixel_area

        real_velocity_values_50 = real_phot_table_50['aperture_sum'] / pixel_area

        real_error_values_50 = real_error_table_50['aperture_sum'] / pixel_area

        sig_values_50 = sig_table_50['aperture_sum'] / pixel_area

        sig_error_values_50 = sig_error_table_50['aperture_sum'] / pixel_area

        # the 16th percentile extraction parameters

        apertures_16 = CircularAperture(positions_16, r=r_aper)

        mod_phot_table_16 = aperture_photometry(model_16, apertures_16)

        real_phot_table_16 = aperture_photometry(self.vel_data, apertures_16)

        real_error_table_16 = aperture_photometry(self.error_data, apertures_16)

        sig_table_16 = aperture_photometry(self.sig_data, apertures_16)

        sig_error_table_16 = aperture_photometry(self.sig_error_data, apertures_16)

        mod_velocity_values_16 = mod_phot_table_16['aperture_sum'] / pixel_area

        real_velocity_values_16 = real_phot_table_16['aperture_sum'] / pixel_area

        real_error_values_16 = real_error_table_16['aperture_sum'] / pixel_area

        sig_values_16 = sig_table_16['aperture_sum'] / pixel_area

        sig_error_values_16 = sig_error_table_16['aperture_sum'] / pixel_area

        # the 84th percentile extraction parameters

        apertures_84 = CircularAperture(positions_84, r=r_aper)

        mod_phot_table_84 = aperture_photometry(model_84, apertures_84)

        real_phot_table_84 = aperture_photometry(self.vel_data, apertures_84)

        real_error_table_84 = aperture_photometry(self.error_data, apertures_84)

        sig_table_84 = aperture_photometry(self.sig_data, apertures_84)

        sig_error_table_84 = aperture_photometry(self.sig_error_data, apertures_84)

        mod_velocity_values_84 = mod_phot_table_84['aperture_sum'] / pixel_area

        real_velocity_values_84 = real_phot_table_84['aperture_sum'] / pixel_area

        real_error_values_84 = real_error_table_84['aperture_sum'] / pixel_area

        sig_values_84 = sig_table_84['aperture_sum'] / pixel_area

        sig_error_values_84 = sig_error_table_84['aperture_sum'] / pixel_area

        # plotting the model and extracted quantities

        min_ind = 0
        max_ind = 0

        try:

            while np.isnan(real_velocity_values_50[min_ind]):

                min_ind += 1

        except IndexError:

            min_ind = 0

        try:

            while np.isnan(real_velocity_values_50[::-1][max_ind]):

                max_ind += 1

            max_ind = max_ind + 1

        except IndexError:

            max_ind = 0

        # construct dictionary of these velocity values and
        # the final distance at which the data is extracted from centre

        extract_d = {'50': [mod_velocity_values_50[min_ind] / np.sin(inc_50),
                            mod_velocity_values_50[-max_ind] / np.sin(inc_50)],
                     '16': [mod_velocity_values_16[min_ind] / np.sin(inc_16),
                            mod_velocity_values_16[-max_ind] / np.sin(inc_16)],
                     '84': [mod_velocity_values_84[min_ind] / np.sin(inc_84),
                            mod_velocity_values_84[-max_ind] / np.sin(inc_84)],
                     'real': [real_velocity_values_50[min_ind] / np.sin(inc_50),
                              real_velocity_values_50[-max_ind] / np.sin(inc_50)],
                     'distance': [x_50[min_ind],
                                  x_50[-max_ind]],
                     'vel_error': [real_error_values_50[min_ind],
                                   real_error_values_50[-max_ind]],
                     'vel_max': [np.nanmax(abs(real_velocity_values_50 / np.sin(inc_50)))],
                     'inclination' : inc_50,
                     'mod_50_velocity' : mod_velocity_values_50 / np.sin(inc_50),
                     'mod_50_positions' : x_50,
                     'mod_16_velocity' : mod_velocity_values_16 / np.sin(inc_16),
                     'mod_16_positions' : x_16,
                     'mod_84_velocity' : mod_velocity_values_84 / np.sin(inc_84),
                     'mod_84_positions' : x_84}

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(x_max,
                mod_velocity_values_max,
                color='red',
                label='max_model')

        ax.errorbar(x_max,
                    real_velocity_values_max,
                    yerr=real_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.plot(x_50,
                mod_velocity_values_50,
                color='blue',
                label='50_model')

        ax.errorbar(x_50,
                    real_velocity_values_50,
                    yerr=real_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.plot(x_16,
                mod_velocity_values_16,
                color='orange',
                linestyle='--',
                label='16_model')

        ax.plot(x_84,
                mod_velocity_values_84,
                color='purple',
                linestyle='--',
                label='84_model')

        # ax.legend(prop={'size':10})
        ax.set_xlim(-1.5, 1.5)

        # ax.legend(prop={'size':5}, loc=1)

        ax.axhline(0, color='silver', ls='-.')
        ax.axvline(0, color='silver', ls='-.')
        ax.axhline(va_50, color='silver', ls='--')
        ax.axhline(-1.*va_50, color='silver', ls='--')
        ax.set_title('Model and Real Velocity')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        # plt.show()

        fig.savefig('%s_1d_velocity_plot.png' % self.fileName[:-5])

        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.errorbar(x_max,
                    sig_values_max,
                    yerr=sig_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.errorbar(x_50,
                    sig_values_50,
                    yerr=sig_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.set_title('Velocity Dispersion')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        ax.legend(prop={'size':10})

        plt.close('all')

        # plt.show()

        fig.savefig('%s_1d_dispersion_plot.png' % self.fileName[:-5])

        # return the data used in plotting for use elsewhere

        return {'max': [x_max,
                        mod_velocity_values_max,
                        real_velocity_values_max,
                        real_error_values_max,
                        sig_values_max,
                        sig_error_values_max],
                '50': [x_50,
                        mod_velocity_values_50,
                        real_velocity_values_50,
                        real_error_values_50,
                        sig_values_50,
                        sig_error_values_50],
                '16': [x_16,
                        mod_velocity_values_16,
                        real_velocity_values_16,
                        real_error_values_16,
                        sig_values_16,
                        sig_error_values_16],
                '84': [x_84,
                        mod_velocity_values_84,
                        real_velocity_values_84,
                        real_error_values_84,
                        sig_values_84,
                        sig_error_values_84]}, extract_d

    def disk_function_fixed(self,
                            theta,
                            xcen,
                            ycen,
                            xpos,
                            ypos):
        """
        Def: Function to calculate disk velocity given input values.
        Note that all angles must be given in radians
        """
        # unpack the parameters

        inc, pa, rt, vasym = theta

        # look at the difference between central pixel and pixel
        # being modelled

        diff_x = (xcen - xpos) * 1.0

        diff_y = (ycen - ypos) * 1.0

        # print diff_x, diff_y

        # calculate the pixel angle

        if diff_y == 0 and diff_x != 0:

            pixel_angle = np.arctan(np.sign(diff_x)*np.inf)

            # print 'This is the pixel angle %s' % pixel_angle

        elif diff_y == 0 and diff_x == 0:

            # print 'In the middle'

            pixel_angle = 0.0

        else:

            # print 'computing pixel angle'

            pixel_angle = np.arctan(diff_x / diff_y)

            # print 'pixel angle %s' % (pixel_angle * 180 / np.pi)

        # work out phi which is the overall angle between
        # the spaxel being modelled and the central spaxel/position angle
        # this involves summing with a rotation angle which depends on
        # the spaxel quadrant

        if diff_x >= 0 and diff_y >= 0 and not(diff_x == 0 and diff_y == 0):

            # print 'top left'
            # we're in the upper left quadrant, want rot to be 270

            rot = 3 * np.pi / 2

        elif diff_x >= 0 and diff_y < 0:

            # print 'top right'

            # we're in the upper right quandrant, want rot to be 90

            rot = np.pi / 2

        elif diff_x < 0 and diff_y < 0:

            # print 'lower right'

            # we're in the lower right quadrant

            rot = np.pi / 2

        elif diff_x < 0 and diff_y >= 0:

            # print 'lower left'

            # we're in the lower left quadrant

            rot = 3 * np.pi / 2

        elif diff_x == 0 and diff_y == 0:

            # print 'middle'

            # we're in the middle

            rot = pa

        phi = pixel_angle - pa + rot

    #    print 'differences: %s %s' % (diff_x, diff_y)
    #    print 'pixel angle %s' % (pixel_angle * 180 / np.pi)
    #    print 'position angle %s' % (pa * 180 / np.pi)
    #    print 'rotation angle %s' % (rot * 180 / np.pi)
    #    print 'overall angle %s' % (phi * 180 / np.pi)
    #    print 'cosine of angle %s' % (np.cos(phi))

        r = np.sqrt(diff_x*diff_x + diff_y*diff_y)

        vel = np.cos(phi) * np.sin(inc) * (2 / np.pi) * vasym * np.arctan(r / rt)

        # print vel, xpix, ypix

        return vel

    def shrink(self,
               data,
               rows,
               cols):
        return np.nanmedian(np.nanmedian(data.reshape(rows,
                                                      data.shape[0] / float(rows),
                                                      cols,
                                                      data.shape[1] / float(cols)),
                                         axis=1),
                            axis=2)

    def compute_model_grid_fixed(self,
                                 theta,
                                 xcen,
                                 ycen,
                                 seeing,
                                 pix_scale,
                                 psf_factor,
                                 smear=False):

        """
        Def:
        Use the grid function to construct a basis for the model.
        Then apply the disk function to each spaxel in the basis
        reshape back to 2d array and plot the model velocity.

        Toying around with constructing the model at much higher spatial
        resolution to properly capture the arctangent function, and then
        re-binning back to the original dimensions
        """

        xbin, ybin = self.grid()

        # setup list to house the velocity measurements

        vel_array = []

        # compute the model at each spaxel location

        for xpos, ypos in zip(xbin, ybin):

            # run the disk function

            vel_array.append(self.disk_function_fixed(theta,
                                                      xcen,
                                                      ycen,
                                                      xpos,
                                                      ypos))

        # create numpy array from the vel_array list

        vel_array = np.array(vel_array)

        # reshape back to the chosen grid dimensions

        vel_2d = vel_array.reshape((self.xpix, self.ypix))

        if smear:

            vel_2d = psf.blur_by_psf(vel_2d,
                                     seeing,
                                     pix_scale,
                                     psf_factor)

        return vel_2d

    def compute_model_grid_fixed_100(self,
                                     theta,
                                     xcen,
                                     ycen):

        """
        Def:
        Use the grid function to construct a basis for the model.
        Then apply the disk function to each spaxel in the basis
        reshape back to 2d array and plot the model velocity.

        Toying around with constructing the model at much higher spatial
        resolution to properly capture the arctangent function, and then
        re-binning back to the original dimensions
        """

        xbin, ybin = self.grid_100()

        # setup list to house the velocity measurements

        vel_array = []
         
        # compute the model at each spaxel location

        for xpos, ypos in zip(xbin, ybin):

            # run the disk function

            vel_array.append(self.disk_function_fixed(theta,
                                                      xcen * 100,
                                                      ycen * 100,
                                                      xpos,
                                                      ypos))

        # create numpy array from the vel_array list

        vel_array = np.array(vel_array)

        # reshape back to the chosen grid dimensions

        vel_2d = vel_array.reshape((self.xpix * 100, self.ypix * 100))

        vel_2d = self.shrink(vel_2d, self.xpix, self.ypix)

        # plot as a 2d array

#        print 'Showing high resolution velocity'
#        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#        im = ax.imshow(vel_2d,
#                       cmap=plt.get_cmap('jet'),
#                       interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(ax)
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # set the title
#        ax.set_title('model velocity')
#        plt.show()
#        plt.close('all')

        # vel_2d_blurred = psf.blur_by_psf(vel_2d, 0.5, 0.1)

        return vel_2d

    def lnlike_fixed(self, 
                     theta,
                     xcen,
                     ycen,
                     seeing,
                     pix_scale,
                     psf_factor,
                     smear=False):
        """
        Def: Return the log likelihood for the velocity field function.
        All that has to be done is to compute the model in a grid the same size
        as the data and then plug into the standard likelihood formula.

        Input:
                vel_data - the actual velocity field unicodedata
                vel_errors - the velocity field error grid
                theta - list of parameter values to be fed into the model

        Output:
                some single numerical value for the log likelihood
        """
        # sometimes nice to see what parameters are being tried in the
        # MCMC step

        # print theta

        # compute the model grid

        model = self.compute_model_grid_fixed(theta,
                                              xcen,
                                              ycen,
                                              seeing,
                                              pix_scale,
                                              psf_factor,
                                              smear)

        # find the grid of inverse sigma values

        inv_sigma2 = 1.0 / (self.error_data * self.error_data)

        ans = -0.5 * (np.nansum((self.vel_data - model)*(self.vel_data - model) *
                                inv_sigma2 - np.log(inv_sigma2)))

        # print ans

        return ans

    def lnprior_fixed(self,
                      theta):

        """
        Def:
        Set an uninformative prior distribution for the parameters in the model
        """

        inc, pa, rt, vasym = theta

        if 0.0 < inc < np.pi / 2.0 and \
           0 < pa < 2 * np.pi and \
           1.0 < rt < 5.0 and \
           0 < vasym < 350:

            return 0.0

        return -np.inf

    def lnprob_fixed(self,
                     theta,
                     xcen,
                     ycen,
                     seeing,
                     pix_scale,
                     psf_factor,
                     smear=False):

        lp = self.lnprior_fixed(theta)

        if not np.isfinite(lp):

            return -np.inf

        return lp + self.lnlike_fixed(theta,
                                      xcen,
                                      ycen,
                                      seeing,
                                      pix_scale,
                                      psf_factor,
                                      smear)

    def run_emcee_fixed(self,
                        theta,
                        xcen,
                        ycen,
                        nsteps,
                        nwalkers,
                        burn_no,
                        seeing,
                        pix_scale,
                        psf_factor,
                        smear=False):

        ndim = len(theta)

        pos = [theta + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.lnprob_fixed,
                                        args=[xcen,
                                              ycen,
                                              seeing,
                                              pix_scale,
                                              psf_factor,
                                              smear])

        for i, (pos, lnp, state) in enumerate(sampler.sample(pos,
                                                             iterations=nsteps)):

            stdout.write("\rObject %s %.1f%% complete" % (self.gal_name[:-15],
                                                        100 * float(i + 1) / nsteps))
            stdout.flush()

        stdout.write('\n')

        samples = sampler.chain[:, burn_no:, :].reshape((-1, ndim))

        fig = corner.corner(samples,
                            labels=["$inc$",
                                    "$pa$",
                                    "rt",
                                    "vasym"],
                            truths=theta)

        fig.savefig('%s_corner_plot_fixed.png' % self.fileName[:-5])

        # plt.show()

        # print samples
        # going to save pickled versions of the chain and the lnprobability
        # so that these can be accessed again later if necessary

        if os.path.isfile(self.chain_name):

            os.system('rm %s' % self.chain_name)

        if os.path.isfile(self.ln_p_name):

            os.system('rm %s' % self.ln_p_name)

        chain_file = open(self.chain_name, 'w')

        pickle.dump(sampler.chain, chain_file)

        chain_file.close()

        ln_p_file = open(self.ln_p_name, 'w')

        pickle.dump(sampler.lnprobability, ln_p_file)

        ln_p_file.close()

        # now use the helper function below to open up
        # the pickled files and write to params file

        self.write_params_fixed(burn_no)

        # set a variable to the log probability value

    def lnprior_fixed_inc_vary(self,
                               theta,
                               inc_middle):

        """
        Def:
        Set an uninformative prior distribution for the parameters in the model
        """

        inc, pa, rt, vasym = theta

        if inc_middle - 0.26 < inc < inc_middle + 0.26 and \
           0 < pa < 2 * np.pi and \
           1.0 < rt < 5.0 and \
           0 < vasym < 350:

            return 0.0

        return -np.inf

    def lnprob_fixed_inc_vary(self,
                              theta,
                              xcen,
                              ycen,
                              inc_middle,
                              seeing,
                              pix_scale,
                              psf_factor,
                              smear=False):

        lp = self.lnprior_fixed_inc_vary(theta,
                                         inc_middle)

        if not np.isfinite(lp):

            return -np.inf

        return lp + self.lnlike_fixed(theta,
                                      xcen,
                                      ycen,
                                      seeing,
                                      pix_scale,
                                      psf_factor,
                                      smear)

    def run_emcee_fixed_inc_vary(self,
                                 theta,
                                 xcen,
                                 ycen,
                                 inc_middle,
                                 nsteps,
                                 nwalkers,
                                 burn_no,
                                 seeing,
                                 pix_scale,
                                 psf_factor,
                                 smear=False):

        ndim = len(theta)

        pos = [theta + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.lnprob_fixed_inc_vary,
                                        args=[xcen,
                                              ycen,
                                              inc_middle,
                                              seeing,
                                              pix_scale,
                                              psf_factor,
                                              smear])

        for i, (pos, lnp, state) in enumerate(sampler.sample(pos,
                                                             iterations=nsteps)):

            stdout.write("\rObject %s %.1f%% complete" % (self.gal_name[:-15],
                                                        100 * float(i + 1) / nsteps))
            stdout.flush()

        stdout.write('\n')

        samples = sampler.chain[:, burn_no:, :].reshape((-1, ndim))

        fig = corner.corner(samples,
                            labels=["$inc$",
                                    "$pa$",
                                    "rt",
                                    "vasym"],
                            truths=theta)

        fig.savefig('%s_corner_plot_fixed_inc_vary.png' % self.fileName[:-5])

        # plt.show()

        # print samples
        # going to save pickled versions of the chain and the lnprobability
        # so that these can be accessed again later if necessary

        if os.path.isfile(self.chain_name):

            os.system('rm %s' % self.chain_name)

        if os.path.isfile(self.ln_p_name):

            os.system('rm %s' % self.ln_p_name)

        chain_file = open(self.chain_name, 'w')

        pickle.dump(sampler.chain, chain_file)

        chain_file.close()

        ln_p_file = open(self.ln_p_name, 'w')

        pickle.dump(sampler.lnprobability, ln_p_file)

        ln_p_file.close()

        # now use the helper function below to open up
        # the pickled files and write to params file

        self.write_params_fixed(burn_no, vary=True)

        # set a variable to the log probability value

    def write_params_fixed(self,
                           burn_no,
                           vary=False):

        """
        Def:
        Helper function to open up the pickled chain and lnp files
        and write the maximum likelihood, 50th percentile, 16th per and 84th
        per parameters to file for ease of application later on

        Input:
                burn_no - number of entries (steps) to burn from the chain
        """

        chain = pickle.load(open(self.chain_name, 'r'))

        lnp = pickle.load(open(self.ln_p_name, 'r'))

        samples = chain[:, burn_no:, :].reshape((-1, chain.shape[2]))

        # initialise the parameter names

        param_names = ['type',
                       'inclination',
                       'position_angle',
                       'Flattening_Radius',
                       'Flattening_Velocity']

        # find the max likelihood parameters

        max_p = np.unravel_index(lnp.argmax(), lnp.shape)

        max_params = chain[max_p[0], max_p[1], :]

        i_mcmc, \
            pa_mcmc, rt_mcmc, va_mcmc \
            = zip(*np.percentile(samples, [16, 50, 84],
                  axis=0))

        if vary:

            param_file = self.fileName[:-5] + '_params_fixed_inc_vary.txt'

        else:

            param_file = self.fileName[:-5] + '_params_fixed.txt'

        if os.path.isfile(param_file):

            os.system('rm %s' % param_file)

        # write all of these values to file

        with open(param_file, 'a') as f:

            for item in param_names:

                f.write('%s\t' % item)

            f.write('\nMAX_lnp:\t')

            for item in max_params:

                f.write('%s\t' % item)

            f.write('\n50th_lnp:\t%s\t%s\t%s\t%s\t' % (i_mcmc[1],
                                                       pa_mcmc[1],
                                                       rt_mcmc[1],
                                                       va_mcmc[1]))

            f.write('\n16th_lnp:\t%s\t%s\t%s\t%s\t' % (i_mcmc[0],
                                                       pa_mcmc[0],
                                                       rt_mcmc[0],
                                                       va_mcmc[0]))

            f.write('\n84th_lnp:\t%s\t%s\t%s\t%s\t' % (i_mcmc[2],
                                                       pa_mcmc[2],
                                                       rt_mcmc[2],
                                                       va_mcmc[2]))


    def plot_comparison_fixed(self,
                              xcen,
                              ycen,
                              seeing,
                              pix_scale,
                              psf_factor,
                              smear=False,
                              vary=False):

        """
        Def:
        Plot the best fitting model alongside the original velocity field
        with position angle and morphological angle also plotted

        Input:
                theta - the now best fit set of parameters
                vel_data - the velocity field unicodedata
                vel_errors - the velocity field errors

        """

        # load in the file

        if vary:

            param_file = np.genfromtxt(self.param_file_fixed_inc_vary)

        else:

            param_file = np.genfromtxt(self.param_file_fixed)

        theta_max = param_file[1][1:]

        theta_50 = param_file[2][1:]

        theta_16 = param_file[3][1:]

        theta_84 = param_file[4][1:]

        # compute the model grid with the specified parameters

        model_max = self.compute_model_grid_fixed(theta_max,
                                                  xcen,
                                                  ycen,
                                                  seeing,
                                                  pix_scale,
                                                  psf_factor,
                                                  smear)

        model_50 = self.compute_model_grid_fixed(theta_50,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        model_16 = self.compute_model_grid_fixed(theta_16,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        model_84 = self.compute_model_grid_fixed(theta_84,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        # only want to see the evaluated model at the grid points
        # where the data is not nan. Loop round the data and create
        # a mask which can multiply the model

        mask_array = np.empty(shape=(self.xpix, self.ypix))

        for i in range(0, self.xpix):

            for j in range(0, self.ypix):

                if np.isnan(self.vel_data[i][j]):

                    mask_array[i][j] = np.nan

                else:

                    mask_array[i][j] = 1.0

        # take product of model and mask_array to return new data

        trunc_model_max = mask_array * model_max

        trunc_model_50 = mask_array * model_50

        trunc_model_16 = mask_array * model_16

        trunc_model_84 = mask_array * model_84

        # plot the results

        vel_min, vel_max = np.nanpercentile(self.vel_data,
                                            [5.0, 95.0])

        mod_min, mod_max = np.nanpercentile(trunc_model_max,
                                            [5.0, 95.0])

        plt.close('all')

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        im = ax[0].imshow(self.vel_data,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[0].set_title('[OIII] Velocity Data')

        im = ax[1].imshow(trunc_model_50,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[1].set_title('[OIII] Velocity Model')

        # plt.show()

        if vary:

            fig.savefig('%s_model_comparison_fixed_inc_vary.png' % self.fileName[:-5])

        else:

            fig.savefig('%s_model_comparison_fixed.png' % self.fileName[:-5])

        plt.close('all')

    def extract_in_apertures_fixed(self,
                                   xcen,
                                   ycen,
                                   r_aper,
                                   d_aper,
                                   seeing,
                                   pix_scale,
                                   psf_factor,
                                   smear=False,
                                   vary=False):

        """
        Def: Extract the velocity field along the kinematic axis returned by the
        model fitting in both the data and the model for comparison. The model
        will show a perfect arctangent function.

        Input:
                theta - array of best fitting model parameter values
                model_data - best fit model computed from the compute_model_grid
                vel_data - array containing the actual velocity data
                r_aper - aperture size in pixels to use for each aperture
                d_aper - distance spacing between apertures
        Output:
                1D arrays containing the extracted model and real velocity fields
                along the kinematic major axis
        """

        # assign the best fit parameters to variables from the theta array

        # load in the file

        if vary:

            param_file = np.genfromtxt(self.param_file_fixed_inc_vary)

        else:

            param_file = np.genfromtxt(self.param_file_fixed)

        theta_max = param_file[1][1:]

        pa_max = theta_max[1]

        theta_50 = param_file[2][1:]

        pa_50 = theta_50[1]

        va_50 = theta_50[3]

        inc_50 = theta_50[0]

        theta_16 = param_file[3][1:]

        pa_16 = theta_16[1]

        inc_16 = theta_16[0]

        theta_84 = param_file[4][1:]

        pa_84 = theta_84[1]

        inc_84 = theta_84[0]

        # compute the model grid with the specified parameters

        model_max = self.compute_model_grid_fixed(theta_max,
                                                  xcen,
                                                  ycen,
                                                  seeing,
                                                  pix_scale,
                                                  psf_factor,
                                                  smear)

        model_50 = self.compute_model_grid_fixed(theta_50,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        model_16 = self.compute_model_grid_fixed(theta_16,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        model_84 = self.compute_model_grid_fixed(theta_84,
                                                 xcen,
                                                 ycen,
                                                 seeing,
                                                 pix_scale,
                                                 psf_factor,
                                                 smear)

        # initialise the list of aperture positions with the xcen and ycen

        positions_max = []

        positions_50 = []

        positions_16 = []

        positions_84 = []

        # first job is to compute the central locations of the apertures
        # do this by fixing the distance along the KA between aperture centres

        xdim = self.xpix - 2

        ydim = self.ypix - 2

        # find the steps along the KA with which to increment

        x_inc_max = d_aper * abs(np.sin((np.pi / 2.0) - pa_max))

        y_inc_max = d_aper * abs(np.cos((np.pi / 2.0) - pa_max))

        x_inc_50 = d_aper * abs(np.sin((np.pi / 2.0) - pa_50))

        y_inc_50 = d_aper * abs(np.cos((np.pi / 2.0) - pa_50))

        x_inc_16 = d_aper * abs(np.sin((np.pi / 2.0) - pa_16))

        y_inc_16 = d_aper * abs(np.cos((np.pi / 2.0) - pa_16))

        x_inc_84 = d_aper * abs(np.sin((np.pi / 2.0) - pa_84))

        y_inc_84 = d_aper * abs(np.cos((np.pi / 2.0) - pa_84))

        # now find the sequence of aperture centres up until the boundaries
        # this is tricky - depending on the PA need to increase and decrease
        # both x and y together, or increase one and decrease the other

        # statements for the maximum likelihood position angle

        if 0 < pa_max < np.pi / 2.0 or np.pi < pa_max < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_max = xcen + x_inc_max

            new_y_max = ycen - y_inc_max

            # while loop until xdim is breached or 0 is breached for y

            while new_x_max < xdim and new_y_max > 2:

                # append aperture centre to the positions array

                positions_max.append((new_y_max, new_x_max))

                new_x_max += x_inc_max

                new_y_max -= y_inc_max

                # print new_x_max, new_y_max

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_max = positions_max[::-1]

            positions_max.append((ycen, xcen))

            # now go in the other direction

            new_x_max = xcen - x_inc_max

            new_y_max = ycen + y_inc_max

            # while loop until xdim is breached or 0 is breached for y

            while new_x_max > 2 and new_y_max < ydim:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max -= x_inc_max

                new_y_max += y_inc_max

                # print new_x, new_y_max

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_max = xcen - x_inc_max

            new_y_max = ycen - y_inc_max

            # while loop until xdim is 2 or ydim is 2

            while new_x_max > 2 and new_y_max > 2:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max -= x_inc_max

                new_y_max -= y_inc_max

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_max = positions_max[::-1]

            positions_max.append((ycen, xcen))

            # now go in the other direction

            new_x_max = xcen + x_inc_max

            new_y_max = ycen + y_inc_max

            # while loop until xdim is breached or ydim is breached

            while new_x_max < xdim and new_y_max < ydim:

                # append aperture centre to the positions_max array

                positions_max.append((new_y_max, new_x_max))

                new_x_max += x_inc_max

                new_y_max += y_inc_max

        # statements for the 50th percentile position angle

        if 0 < pa_50 < np.pi / 2.0 or np.pi < pa_50 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_50 = xcen + x_inc_50

            new_y_50 = ycen - y_inc_50

            # while loop until xdim is breached or 0 is breached for y

            while new_x_50 < xdim and new_y_50 > 2:

                # append aperture centre to the positions array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 += x_inc_50

                new_y_50 -= y_inc_50

                # print new_x_50, new_y_50

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_50 = positions_50[::-1]

            positions_50.append((ycen, xcen))

            # now go in the other direction

            new_x_50 = xcen - x_inc_50

            new_y_50 = ycen + y_inc_50

            # while loop until xdim is breached or 0 is breached for y

            while new_x_50 > 2 and new_y_50 < ydim:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 -= x_inc_50

                new_y_50 += y_inc_50

                # print new_x, new_y_50

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_50 = xcen - x_inc_50

            new_y_50 = ycen - y_inc_50

            # while loop until xdim is 2 or ydim is 2

            while new_x_50 > 2 and new_y_50 > 2:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 -= x_inc_50

                new_y_50 -= y_inc_50

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_50 = positions_50[::-1]

            positions_50.append((ycen, xcen))

            # now go in the other direction

            new_x_50 = xcen + x_inc_50

            new_y_50 = ycen + y_inc_50

            # while loop until xdim is breached or ydim is breached

            while new_x_50 < xdim and new_y_50 < ydim:

                # append aperture centre to the positions_50 array

                positions_50.append((new_y_50, new_x_50))

                new_x_50 += x_inc_50

                new_y_50 += y_inc_50

        # statements for the 16th percentile position angle

        if 0 < pa_16 < np.pi / 2.0 or np.pi < pa_16 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_16 = xcen + x_inc_16

            new_y_16 = ycen - y_inc_16

            # while loop until xdim is breached or 0 is breached for y

            while new_x_16 < xdim and new_y_16 > 2:

                # append aperture centre to the positions array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 += x_inc_16

                new_y_16 -= y_inc_16

                # print new_x_16, new_y_16

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_16 = positions_16[::-1]

            positions_16.append((ycen, xcen))

            # now go in the other direction

            new_x_16 = xcen - x_inc_16

            new_y_16 = ycen + y_inc_16

            # while loop until xdim is breached or 0 is breached for y

            while new_x_16 > 2 and new_y_16 < ydim:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 -= x_inc_16

                new_y_16 += y_inc_16

                # print new_x, new_y_16

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_16 = xcen - x_inc_16

            new_y_16 = ycen - y_inc_16

            # while loop until xdim is 2 or ydim is 2

            while new_x_16 > 2 and new_y_16 > 2:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 -= x_inc_16

                new_y_16 -= y_inc_16

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_16 = positions_16[::-1]

            positions_16.append((ycen, xcen))

            # now go in the other direction

            new_x_16 = xcen + x_inc_16

            new_y_16 = ycen + y_inc_16

            # while loop until xdim is breached or ydim is breached

            while new_x_16 < xdim and new_y_16 < ydim:

                # append aperture centre to the positions_16 array

                positions_16.append((new_y_16, new_x_16))

                new_x_16 += x_inc_16

                new_y_16 += y_inc_16

        # statements for the 84th percenntile position angle

        if 0 < pa_84 < np.pi / 2.0 or np.pi < pa_84 < 3 * np.pi / 2.0:

            # print 'Top Right and Bottom Left'

            # need to increase x and decrease y and vice versa

            new_x_84 = xcen + x_inc_84

            new_y_84 = ycen - y_inc_84

            # while loop until xdim is breached or 0 is breached for y

            while new_x_84 < xdim and new_y_84 > 2:

                # append aperture centre to the positions array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 += x_inc_84

                new_y_84 -= y_inc_84

                # print new_x_84, new_y_84

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_84 = positions_84[::-1]

            positions_84.append((ycen, xcen))

            # now go in the other direction

            new_x_84 = xcen - x_inc_84

            new_y_84 = ycen + y_inc_84

            # while loop until xdim is breached or 0 is breached for y

            while new_x_84 > 2 and new_y_84 < ydim:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 -= x_inc_84

                new_y_84 += y_inc_84

                # print new_x, new_y_84

        # deal with the other cases of position angle

        else:

            # print 'Top Left and Bottom Right'

            # need to increase x and increase y and vice versa

            new_x_84 = xcen - x_inc_84

            new_y_84 = ycen - y_inc_84

            # while loop until xdim is 2 or ydim is 2

            while new_x_84 > 2 and new_y_84 > 2:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 -= x_inc_84

                new_y_84 -= y_inc_84

            # starting from the left so need to reverse list direction
            # and append the central point

            positions_84 = positions_84[::-1]

            positions_84.append((ycen, xcen))

            # now go in the other direction

            new_x_84 = xcen + x_inc_84

            new_y_84 = ycen + y_inc_84

            # while loop until xdim is breached or ydim is breached

            while new_x_84 < xdim and new_y_84 < ydim:

                # append aperture centre to the positions_84 array

                positions_84.append((new_y_84, new_x_84))

                new_x_84 += x_inc_84

                new_y_84 += y_inc_84

        # construct the x_axis for the aperture extraction plot

        x_max_array = []

        for entry in positions_max:

            x_max_array.append(entry[1])

        x_max_array = np.array(x_max_array) - xcen

        # print x_max_array

        x_max_index = np.where(x_max_array == 0.0)[0]

        x_max = np.linspace(-1. * d_aper * x_max_index,
                            d_aper * (len(x_max_array) - x_max_index - 1),
                            num=len(x_max_array))

        x_max = x_max * pix_scale

        # print 'This is x_max: %s' % x_max

        x_50_array = []

        for entry in positions_50:

            x_50_array.append(entry[1])

        x_50_array = np.array(x_50_array) - xcen

        x_50_index = np.where(x_50_array == 0.0)[0]

        x_50 = np.linspace(-1. * d_aper * x_50_index,
                            d_aper * (len(x_50_array) - x_50_index - 1),
                            num=len(x_50_array))

        x_50 = x_50 * pix_scale

        x_16_array = []

        for entry in positions_16:

            x_16_array.append(entry[1])

        x_16_array = np.array(x_16_array) - xcen

        x_16_index = np.where(x_16_array == 0.0)[0]

        x_16 = np.linspace(-1. * d_aper * x_16_index,
                            d_aper * (len(x_16_array) - x_16_index - 1),
                            num=len(x_16_array))

        x_16 = x_16 * pix_scale

        x_84_array = []

        for entry in positions_84:

            x_84_array.append(entry[1])

        x_84_array = np.array(x_84_array) - xcen

        x_84_index = np.where(x_84_array == 0.0)[0]

        x_84 = np.linspace(-1. * d_aper * x_84_index,
                            d_aper * (len(x_84_array) - x_84_index - 1),
                            num=len(x_84_array))

        x_84 = x_84 * pix_scale

        # positions array should now be populated with all of the apertures

        # print positions

        # now perform aperture photometry on the model data to check that this
        # actually works. Remember that the velocity computed for each
        # aperture will be the sum returned divided by the area

        pixel_area = np.pi * r_aper * r_aper

        # the max likelihood extraction parameters

        apertures_max = CircularAperture(positions_max, r=r_aper)

        mod_phot_table_max = aperture_photometry(model_max, apertures_max)

        real_phot_table_max = aperture_photometry(self.vel_data, apertures_max)

        real_error_table_max = aperture_photometry(self.error_data, apertures_max)

        sig_table_max = aperture_photometry(self.sig_data, apertures_max)

        sig_error_table_max = aperture_photometry(self.sig_error_data, apertures_max)

        mod_velocity_values_max = mod_phot_table_max['aperture_sum'] / pixel_area

        real_velocity_values_max = real_phot_table_max['aperture_sum'] / pixel_area

        real_error_values_max = real_error_table_max['aperture_sum'] / pixel_area

        sig_values_max = sig_table_max['aperture_sum'] / pixel_area

        sig_error_values_max = sig_error_table_max['aperture_sum'] / pixel_area

        # the 50th percentile extraction parameters

        apertures_50 = CircularAperture(positions_50, r=r_aper)

        mod_phot_table_50 = aperture_photometry(model_50, apertures_50)

        real_phot_table_50 = aperture_photometry(self.vel_data, apertures_50)

        real_error_table_50 = aperture_photometry(self.error_data, apertures_50)

        sig_table_50 = aperture_photometry(self.sig_data, apertures_50)

        sig_error_table_50 = aperture_photometry(self.sig_error_data, apertures_50)

        mod_velocity_values_50 = mod_phot_table_50['aperture_sum'] / pixel_area

        real_velocity_values_50 = real_phot_table_50['aperture_sum'] / pixel_area

        real_error_values_50 = real_error_table_50['aperture_sum'] / pixel_area

        sig_values_50 = sig_table_50['aperture_sum'] / pixel_area

        sig_error_values_50 = sig_error_table_50['aperture_sum'] / pixel_area

        # the 16th percentile extraction parameters

        apertures_16 = CircularAperture(positions_16, r=r_aper)

        mod_phot_table_16 = aperture_photometry(model_16, apertures_16)

        real_phot_table_16 = aperture_photometry(self.vel_data, apertures_16)

        real_error_table_16 = aperture_photometry(self.error_data, apertures_16)

        sig_table_16 = aperture_photometry(self.sig_data, apertures_16)

        sig_error_table_16 = aperture_photometry(self.sig_error_data, apertures_16)

        mod_velocity_values_16 = mod_phot_table_16['aperture_sum'] / pixel_area

        real_velocity_values_16 = real_phot_table_16['aperture_sum'] / pixel_area

        real_error_values_16 = real_error_table_16['aperture_sum'] / pixel_area

        sig_values_16 = sig_table_16['aperture_sum'] / pixel_area

        sig_error_values_16 = sig_error_table_16['aperture_sum'] / pixel_area

        # the 84th percentile extraction parameters

        apertures_84 = CircularAperture(positions_84, r=r_aper)

        mod_phot_table_84 = aperture_photometry(model_84, apertures_84)

        real_phot_table_84 = aperture_photometry(self.vel_data, apertures_84)

        real_error_table_84 = aperture_photometry(self.error_data, apertures_84)

        sig_table_84 = aperture_photometry(self.sig_data, apertures_84)

        sig_error_table_84 = aperture_photometry(self.sig_error_data, apertures_84)

        mod_velocity_values_84 = mod_phot_table_84['aperture_sum'] / pixel_area

        real_velocity_values_84 = real_phot_table_84['aperture_sum'] / pixel_area

        real_error_values_84 = real_error_table_84['aperture_sum'] / pixel_area

        sig_values_84 = sig_table_84['aperture_sum'] / pixel_area

        sig_error_values_84 = sig_error_table_84['aperture_sum'] / pixel_area

        # find the indices of the first and last non-nan real velocity
        # values for extraction

        min_ind = 0
        max_ind = 0

        try:

            while np.isnan(real_velocity_values_50[min_ind]):

                min_ind += 1

        except IndexError:

            min_ind = 0

        try:

            while np.isnan(real_velocity_values_50[::-1][max_ind]):

                max_ind += 1

            max_ind = max_ind + 1

        except IndexError:

            max_ind = 0

        # construct dictionary of these velocity values and
        # the final distance at which the data is extracted from centre

        extract_d = {'50': [mod_velocity_values_50[min_ind] / np.sin(inc_50),
                            mod_velocity_values_50[-max_ind] / np.sin(inc_50)],
                     '16': [mod_velocity_values_16[min_ind] / np.sin(inc_16),
                            mod_velocity_values_16[-max_ind] / np.sin(inc_16)],
                     '84': [mod_velocity_values_84[min_ind] / np.sin(inc_84),
                            mod_velocity_values_84[-max_ind] / np.sin(inc_84)],
                     'real': [real_velocity_values_50[min_ind] / np.sin(inc_50),
                              real_velocity_values_50[-max_ind] / np.sin(inc_50)],
                     'distance': [x_50[min_ind],
                                  x_50[-max_ind]],
                     'vel_error': [real_error_values_50[min_ind],
                                   real_error_values_50[-max_ind]],
                     'vel_max': [np.nanmax(abs(real_velocity_values_50 / np.sin(inc_50)))],
                     'inclination' : inc_50,
                     'mod_50_velocity' : mod_velocity_values_50 / np.sin(inc_50),
                     'mod_50_positions' : x_50,
                     'mod_16_velocity' : mod_velocity_values_16 / np.sin(inc_16),
                     'mod_16_positions' : x_16,
                     'mod_84_velocity' : mod_velocity_values_84 / np.sin(inc_84),
                     'mod_84_positions' : x_84}

        # plotting the model and extracted quantities

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(x_max,
                mod_velocity_values_max,
                color='red',
                label='max_model')

        ax.errorbar(x_max,
                    real_velocity_values_max,
                    yerr=real_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.plot(x_50,
                mod_velocity_values_50,
                color='blue',
                label='50_model')

        ax.errorbar(x_50,
                    real_velocity_values_50,
                    yerr=real_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.plot(x_16,
                mod_velocity_values_16,
                color='orange',
                linestyle='--',
                label='16_model')

        ax.plot(x_84,
                mod_velocity_values_84,
                color='purple',
                linestyle='--',
                label='84_model')

        # ax.legend(prop={'size':10})
        ax.set_xlim(-1.5, 1.5)

        # ax.legend(prop={'size':5}, loc=1)

        ax.axhline(0, color='silver', ls='-.')
        ax.axvline(0, color='silver', ls='-.')
        ax.axhline(va_50, color='silver', ls='--')
        ax.axhline(-1.*va_50, color='silver', ls='--')
        ax.set_title('Model and Real Velocity')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        # plt.show()

        if vary:

            fig.savefig('%s_1d_velocity_plot_fixed_inc_vary.png' % self.fileName[:-5])

        else:

            fig.savefig('%s_1d_velocity_plot_fixed.png' % self.fileName[:-5])

        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.errorbar(x_max,
                    sig_values_max,
                    yerr=sig_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.errorbar(x_50,
                    sig_values_50,
                    yerr=sig_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.set_title('Velocity Dispersion')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        ax.legend(prop={'size':10})

        plt.close('all')

        # plt.show()

        if vary:

            fig.savefig('%s_1d_dispersion_plot_fixed_inc_vary.png' % self.fileName[:-5])

        else:

            fig.savefig('%s_1d_dispersion_plot_fixed.png' % self.fileName[:-5])

        # return the data used in plotting for use elsewhere

        return {'max': [x_max, 
                        mod_velocity_values_max,
                        real_velocity_values_max,
                        real_error_values_max,
                        sig_values_max,
                        sig_error_values_max],
                '50': [x_50, 
                        mod_velocity_values_50,
                        real_velocity_values_50,
                        real_error_values_50,
                        sig_values_50,
                        sig_error_values_50],
                '16': [x_16, 
                        mod_velocity_values_16,
                        real_velocity_values_16,
                        real_error_values_16,
                        sig_values_16,
                        sig_error_values_16],
                '84': [x_84, 
                        mod_velocity_values_84,
                        real_velocity_values_84,
                        real_error_values_84,
                        sig_values_84,
                        sig_error_values_84]}, extract_d

    def disk_function_fixed_inc_fixed(self,
                                      theta,
                                      xcen,
                                      ycen,
                                      inc,
                                      xpos,
                                      ypos):
        """
        Def: Function to calculate disk velocity given input values.
        Note that all angles must be given in radians
        """
        # unpack the parameters

        pa, rt, vasym = theta

        # look at the difference between central pixel and pixel
        # being modelled

        diff_x = (xcen - xpos) * 1.0

        diff_y = (ycen - ypos) * 1.0

        # print diff_x, diff_y

        # calculate the pixel angle

        if diff_y == 0 and diff_x != 0:

            pixel_angle = np.arctan(np.sign(diff_x)*np.inf)

            # print 'This is the pixel angle %s' % pixel_angle

        elif diff_y == 0 and diff_x == 0:

            # print 'In the middle'

            pixel_angle = 0.0

        else:

            # print 'computing pixel angle'

            pixel_angle = np.arctan(diff_x / diff_y)

            # print 'pixel angle %s' % (pixel_angle * 180 / np.pi)

        # work out phi which is the overall angle between
        # the spaxel being modelled and the central spaxel/position angle
        # this involves summing with a rotation angle which depends on
        # the spaxel quadrant

        if diff_x >= 0 and diff_y >= 0 and not(diff_x == 0 and diff_y == 0):

            # print 'top left'
            # we're in the upper left quadrant, want rot to be 270

            rot = 3 * np.pi / 2

        elif diff_x >= 0 and diff_y < 0:

            # print 'top right'

            # we're in the upper right quandrant, want rot to be 90

            rot = np.pi / 2

        elif diff_x < 0 and diff_y < 0:

            # print 'lower right'

            # we're in the lower right quadrant

            rot = np.pi / 2

        elif diff_x < 0 and diff_y >= 0:

            # print 'lower left'

            # we're in the lower left quadrant

            rot = 3 * np.pi / 2

        elif diff_x == 0 and diff_y == 0:

            # print 'middle'

            # we're in the middle

            rot = pa

        phi = pixel_angle - pa + rot

    #    print 'differences: %s %s' % (diff_x, diff_y)
    #    print 'pixel angle %s' % (pixel_angle * 180 / np.pi)
    #    print 'position angle %s' % (pa * 180 / np.pi)
    #    print 'rotation angle %s' % (rot * 180 / np.pi)
    #    print 'overall angle %s' % (phi * 180 / np.pi)
    #    print 'cosine of angle %s' % (np.cos(phi))

        r = np.sqrt(diff_x*diff_x + diff_y*diff_y)

        vel = np.cos(phi) * np.sin(inc) * (2 / np.pi) * vasym * np.arctan(r / rt)

        # print vel, xpix, ypix

        return vel

    def compute_model_grid_fixed_inc_fixed(self,
                                           theta,
                                           inc,
                                           redshift,
                                           wave_array,
                                           xcen,
                                           ycen,
                                           seeing,
                                           sersic_n,
                                           sigma,
                                           pix_scale,
                                           psf_factor,
                                           sersic_factor,
                                           m_factor,
                                           light_profile,
                                           smear=False):

        """
        Def:
        Use the grid function to construct a basis for the model.
        Then apply the disk function to each spaxel in the basis
        reshape back to 2d array and plot the model velocity
        """

        xbin, ybin = self.grid_factor(m_factor)

        # setup list to house the velocity measurements

        vel_array = []

        # need to increase rt by the model factor, m_factor.

        pa, rt, v = theta

        rt = rt * m_factor

        # and reconstruct theta

        theta = [pa, rt, v]

        # compute the model at each spaxel location

        for xpos, ypos in zip(xbin, ybin):

            # run the disk function

            vel_array.append(self.disk_function_fixed_inc_fixed(theta,
                                                                xcen * m_factor,
                                                                ycen * m_factor,
                                                                inc,
                                                                xpos,
                                                                ypos))

        # create numpy array from the vel_array list

        vel_array = np.array(vel_array)

        # reshape back to the chosen grid dimensions

        vel_2d = vel_array.reshape((self.xpix * m_factor,
                                    self.ypix * m_factor))

        if float(m_factor) != 1.0:

            vel_2d = psf.bin_by_factor(vel_2d,
                                       m_factor)

        pa = theta[0]

        rt = theta[1]

        if smear:

            vel_2d, sigma_2d = psf.cube_blur(vel_2d,
                                             redshift,
                                             wave_array,
                                             xcen,
                                             ycen,
                                             seeing,
                                             pix_scale,
                                             psf_factor,
                                             sersic_factor,
                                             pa,
                                             inc,
                                             rt,
                                             light_profile,
                                             sigma,
                                             sersic_n)


        return vel_2d

    def compute_model_grid_for_chi_squared(self,
                                           theta,
                                           inc,
                                           redshift,
                                           wave_array,
                                           xcen,
                                           ycen,
                                           seeing,
                                           sersic_n,
                                           sigma,
                                           pix_scale,
                                           psf_factor,
                                           sersic_factor,
                                           m_factor,
                                           light_profile,
                                           smear=False):

        """
        Def:
        Use the grid function to construct a basis for the model.
        Then apply the disk function to each spaxel in the basis
        reshape back to 2d array and plot the model velocity
        """

        xbin, ybin = self.grid_factor(m_factor)

        # setup list to house the velocity measurements

        vel_array = []

        # need to increase rt by the model factor, m_factor.

        pa, rt, v = theta

        rt = rt * m_factor

        # and reconstruct theta

        theta = [pa, rt, v]

        # compute the model at each spaxel location

        for xpos, ypos in zip(xbin, ybin):

            # run the disk function

            vel_array.append(self.disk_function_fixed_inc_fixed(theta,
                                                                xcen * m_factor,
                                                                ycen * m_factor,
                                                                inc,
                                                                xpos,
                                                                ypos))

        # create numpy array from the vel_array list

        vel_array = np.array(vel_array)

        # reshape back to the chosen grid dimensions

        vel_2d = vel_array.reshape((self.xpix * m_factor,
                                    self.ypix * m_factor))

        if float(m_factor) != 1.0:

            vel_2d = psf.bin_by_factor(vel_2d,
                                       m_factor)

        pa = theta[0]

        rt = theta[1]

        if smear:

            vel_2d_blurred, sigma_2d = psf.cube_blur(vel_2d,
                                                     redshift,
                                                     wave_array,
                                                     xcen,
                                                     ycen,
                                                     seeing,
                                                     pix_scale,
                                                     psf_factor,
                                                     sersic_factor,
                                                     pa,
                                                     inc,
                                                     rt,
                                                     light_profile,
                                                     sigma,
                                                     sersic_n)


            return [vel_2d, vel_2d_blurred, sigma_2d]

        return vel_2d

    def lnlike_fixed_inc_fixed(self, 
                               theta,
                               inc,
                               redshift,
                               wave_array,
                               xcen,
                               ycen,
                               seeing,
                               sersic_n,
                               sigma,
                               pix_scale,
                               psf_factor,
                               sersic_factor,
                               m_factor,
                               light_profile,
                               smear=False):
        """
        Def: Return the log likelihood for the velocity field function.
        All that has to be done is to compute the model in a grid the same size
        as the data and then plug into the standard likelihood formula.

        Input:
                vel_data - the actual velocity field unicodedata
                vel_errors - the velocity field error grid
                theta - list of parameter values to be fed into the model

        Output:
                some single numerical value for the log likelihood
        """
        # sometimes nice to see what parameters are being tried in the
        # MCMC step

        # print theta

        # compute the model grid

        model = self.compute_model_grid_fixed_inc_fixed(theta,
                                                        inc,
                                                        redshift,
                                                        wave_array,
                                                        xcen,
                                                        ycen,
                                                        seeing,
                                                        sersic_n,
                                                        sigma,
                                                        pix_scale,
                                                        psf_factor,
                                                        sersic_factor,
                                                        m_factor,
                                                        light_profile,
                                                        smear)

        # find the grid of inverse sigma values

        inv_sigma2 = 1.0 / (self.error_data * self.error_data)

        ans = -0.5 * (np.nansum((self.vel_data - model)*(self.vel_data - model) *
                                inv_sigma2 - np.log(inv_sigma2)))

        # print ans

        return ans

    def lnprior_fixed_inc_fixed(self,
                                theta):

        """
        Def:
        Set an uninformative prior distribution for the parameters in the model
        """

        pa, rt, vasym = theta

        if 0 < pa < 2 * np.pi and \
           0.01 < rt < 2.0 and \
           0 < vasym < 250:

            return 0.0

        return -np.inf

    def lnprob_fixed_inc_fixed(self,
                               theta,
                               inc,
                               redshift,
                               wave_array,
                               xcen,
                               ycen,
                               seeing,
                               sersic_n,
                               sigma,
                               pix_scale,
                               psf_factor,
                               sersic_factor,
                               m_factor,
                               light_profile,
                               smear=False):

        lp = self.lnprior_fixed_inc_fixed(theta)

        if not np.isfinite(lp):

            return -np.inf

        return lp + self.lnlike_fixed_inc_fixed(theta,
                                                inc,
                                                redshift,
                                                wave_array,
                                                xcen,
                                                ycen,
                                                seeing,
                                                sersic_n,
                                                sigma,
                                                pix_scale,
                                                psf_factor,
                                                sersic_factor,
                                                m_factor,
                                                light_profile,
                                                smear)

    def grid_fixed_inc_fixed_params(self, 
                                    inc,
                                    pa,
                                    redshift,
                                    wave_array,
                                    xcen,
                                    ycen,
                                    seeing,
                                    sersic_n,
                                    sigma,
                                    pix_scale,
                                    psf_factor,
                                    sersic_factor,
                                    m_factor,
                                    light_profile,
                                    smear=True):

        """
        Def:
        Instead of using MCMC, since we pretty much know what the
        parameters are from earlier MCMC runs, can simply use a grid
        based chi squared approach to find what the best velocity
        and rt values are. i.e. what values of both of these give the
        best match to the data - and are there degeneracies between the
        value of rt and the beam smearing effect?
        """

        # first set up the parameter grids 

        vel_grid = np.arange(20, 150, 1.0)

        rt_grid = np.arange(0.1, 2, 0.1)

        result_array = []

        for i in range(len(vel_grid)):

            stdout.write("\rObject %s %.1f%% complete" % (self.gal_name[:-15],
                                                          100 * float(i) / len(vel_grid)))
            stdout.flush()


            for j in range(len(rt_grid)):

                # set up theta
                theta = [pa, rt_grid[j], vel_grid[i]]

                result_array.append(self.lnlike_fixed_inc_fixed(theta,
                                                                inc,
                                                                redshift,
                                                                wave_array,
                                                                xcen,
                                                                ycen,
                                                                seeing,
                                                                sersic_n,
                                                                sigma,
                                                                pix_scale,
                                                                psf_factor,
                                                                sersic_factor,
                                                                m_factor,
                                                                light_profile,
                                                                smear=smear))

        stdout.write('\n')

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        x = np.arange(0, len(vel_grid) * len(rt_grid), 1)
        ax.plot(x, result_array)
        #plt.show()
        fig.savefig(self.gal_name[:-15] + '_chi_evaluations.png')
        plt.close('all')

        # process the results

        result_array = np.array(result_array)

        result_array = np.reshape(result_array,
                                  newshape=(len(vel_grid), len(rt_grid)))

        vel_idx, rt_idx = np.argwhere(result_array == np.max(result_array))[0]

        print vel_grid[vel_idx]
        print rt_grid[rt_idx]

        return [vel_grid[vel_idx],
                rt_grid[rt_idx]]

    def run_emcee_fixed_inc_fixed(self,
                                  theta,
                                  inc,
                                  redshift,
                                  wave_array,
                                  xcen,
                                  ycen,
                                  nsteps,
                                  nwalkers,
                                  burn_no,
                                  seeing,
                                  sersic_n,
                                  sigma,
                                  pix_scale,
                                  psf_factor,
                                  sersic_factor,
                                  m_factor,
                                  light_profile,
                                  smear=True):

        """
        Def:
        Need to add a description to this
        """        
        ndim = len(theta)

        pos = [theta + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.lnprob_fixed_inc_fixed,
                                        args=[inc,
                                              redshift,
                                              wave_array,
                                              xcen,
                                              ycen,
                                              seeing,
                                              sersic_n,
                                              sigma,
                                              pix_scale,
                                              psf_factor,
                                              sersic_factor,
                                              m_factor,
                                              light_profile,
                                              smear])

        for i, (pos, lnp, state) in enumerate(sampler.sample(pos,
                                                             iterations=nsteps)):

            stdout.write("\rObject %s %.1f%% complete" % (self.gal_name[:-15],
                                                        100 * float(i + 1) / nsteps))
            stdout.flush()

        stdout.write('\n')

        samples = sampler.chain[:, burn_no:, :].reshape((-1, ndim))

        fig = corner.corner(samples,
                            labels=["$pa$",
                                    "rt",
                                    "vasym"],
                            truths=theta)

        fig.savefig('%s_corner_plot_fixed_inc_fixed.png' % self.fileName[:-5])

        # plt.show()

        # print samples
        # going to save pickled versions of the chain and the lnprobability
        # so that these can be accessed again later if necessary

        if os.path.isfile(self.chain_name):

            os.system('rm %s' % self.chain_name)

        if os.path.isfile(self.ln_p_name):

            os.system('rm %s' % self.ln_p_name)

        chain_file = open(self.chain_name, 'w')

        pickle.dump(sampler.chain, chain_file)

        chain_file.close()

        ln_p_file = open(self.ln_p_name, 'w')

        pickle.dump(sampler.lnprobability, ln_p_file)

        ln_p_file.close()

        # now use the helper function below to open up
        # the pickled files and write to params file

        self.write_params_fixed_inc_fixed(burn_no)

        # set a variable to the log probability value

    def write_params_fixed_inc_fixed(self,
                                     burn_no):

        """
        Def:
        Helper function to open up the pickled chain and lnp files
        and write the maximum likelihood, 50th percentile, 16th per and 84th
        per parameters to file for ease of application later on

        Input:
                burn_no - number of entries (steps) to burn from the chain
        """

        chain = pickle.load(open(self.chain_name, 'r'))

        lnp = pickle.load(open(self.ln_p_name, 'r'))

        samples = chain[:, burn_no:, :].reshape((-1, chain.shape[2]))

        # initialise the parameter names

        param_names = ['type',
                       'position_angle',
                       'Flattening_Radius',
                       'Flattening_Velocity']

        # find the max likelihood parameters

        max_p = np.unravel_index(lnp.argmax(), lnp.shape)

        max_params = chain[max_p[0], max_p[1], :]

        pa_mcmc, rt_mcmc, va_mcmc \
            = zip(*np.percentile(samples, [16, 50, 84],
                  axis=0))

        param_file = self.fileName[:-5] + '_params_fixed_inc_fixed.txt'

        if os.path.isfile(param_file):

            os.system('rm %s' % param_file)

        # write all of these values to file

        with open(param_file, 'a') as f:

            for item in param_names:

                f.write('%s\t' % item)

            f.write('\nMAX_lnp:\t')

            for item in max_params:

                f.write('%s\t' % item)

            f.write('\n50th_lnp:\t%s\t%s\t%s\t' % (pa_mcmc[1],
                                                   rt_mcmc[1],
                                                   va_mcmc[1]))

            f.write('\n16th_lnp:\t%s\t%s\t%s\t' % (pa_mcmc[0],
                                                   rt_mcmc[0],
                                                   va_mcmc[0]))

            f.write('\n84th_lnp:\t%s\t%s\t%s\t' % (pa_mcmc[2],
                                                   rt_mcmc[2],
                                                   va_mcmc[2]))


    def plot_comparison_fixed_inc_fixed(self,
                                        inc,
                                        redshift,
                                        wave_array,
                                        xcen,
                                        ycen,
                                        seeing,
                                        sersic_n,
                                        sigma,
                                        pix_scale,
                                        psf_factor,
                                        sersic_factor,
                                        m_factor,
                                        light_profile,
                                        smear=False):

        """
        Def:
        Plot the best fitting model alongside the original velocity field
        with position angle and morphological angle also plotted

        Input:
                theta - the now best fit set of parameters
                vel_data - the velocity field unicodedata
                vel_errors - the velocity field errors

        """

        # load in the file

        param_file = np.genfromtxt(self.param_file_fixed_inc_fixed)

        theta_max = param_file[1][1:]

        theta_50 = param_file[2][1:]

        theta_16 = param_file[3][1:]

        theta_84 = param_file[4][1:]

        # compute the model grid with the specified parameters

        model_max = self.compute_model_grid_fixed_inc_fixed(theta_max,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        model_50 = self.compute_model_grid_fixed_inc_fixed(theta_50,
                                                           inc,
                                                           redshift,
                                                           wave_array,
                                                           xcen,
                                                           ycen,
                                                           seeing,
                                                           sersic_n,
                                                           sigma,
                                                           pix_scale,
                                                           psf_factor,
                                                           sersic_factor,
                                                           m_factor,
                                                           light_profile,
                                                           smear)

        model_16 = self.compute_model_grid_fixed_inc_fixed(theta_16,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        model_84 = self.compute_model_grid_fixed_inc_fixed(theta_84,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        # only want to see the evaluated model at the grid points
        # where the data is not nan. Loop round the data and create
        # a mask which can multiply the model

        mask_array = np.empty(shape=(self.xpix, self.ypix))

        for i in range(0, self.xpix):

            for j in range(0, self.ypix):

                if np.isnan(self.vel_data[i][j]):

                    mask_array[i][j] = np.nan

                else:

                    mask_array[i][j] = 1.0

        # take product of model and mask_array to return new data

        trunc_model_max = mask_array * model_max

        trunc_model_50 = mask_array * model_50

        trunc_model_16 = mask_array * model_16

        trunc_model_84 = mask_array * model_84

        # plot the results

        vel_min, vel_max = np.nanpercentile(self.vel_data,
                                            [5.0, 95.0])

        mod_min, mod_max = np.nanpercentile(trunc_model_max,
                                            [5.0, 95.0])

        plt.close('all')

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        im = ax[0].imshow(self.vel_data,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[0])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[0].set_title('[OIII] Velocity Data')

        im = ax[1].imshow(trunc_model_50,
                          cmap=plt.get_cmap('jet'),
                          vmin=mod_min,
                          vmax=mod_max,
                          interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(ax[1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        ax[1].set_title('[OIII] Velocity Model')

        # plt.show()

        fig.savefig('%s_model_comparison_fixed_inc_fixed.png' % self.fileName[:-5])

        plt.close('all')

    def extract_in_apertures_fixed_inc_fixed(self,
                                             inc,
                                             redshift,
                                             wave_array,
                                             xcen,
                                             ycen,
                                             r_aper,
                                             d_aper,
                                             seeing,
                                             sersic_n,
                                             sigma,
                                             pix_scale,
                                             psf_factor,
                                             sersic_factor,
                                             m_factor,
                                             light_profile,
                                             smear=False):

        """
        Def: Extract the velocity field along the kinematic axis returned by the
        model fitting in both the data and the model for comparison. The model
        will show a perfect arctangent function.

        Input:
                theta - array of best fitting model parameter values
                model_data - best fit model computed from the compute_model_grid
                vel_data - array containing the actual velocity data
                r_aper - aperture size in pixels to use for each aperture
                d_aper - distance spacing between apertures
        Output:
                1D arrays containing the extracted model and real velocity fields
                along the kinematic major axis
        """

        # assign the best fit parameters to variables from the theta array

        # load in the file



        param_file = np.genfromtxt(self.param_file_fixed_inc_fixed)

        theta_max = param_file[1][1:]

        pa_max = theta_max[0]

        theta_50 = param_file[2][1:]

        pa_50 = theta_50[0]

        va_50 = theta_50[2]

        theta_16 = param_file[3][1:]

        pa_16 = theta_16[0]

        theta_84 = param_file[4][1:]

        pa_84 = theta_84[0]

        # compute the model grid with the specified parameters

        model_max = self.compute_model_grid_fixed_inc_fixed(theta_max,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        model_50 = self.compute_model_grid_fixed_inc_fixed(theta_50,
                                                           inc,
                                                           redshift,
                                                           wave_array,
                                                           xcen,
                                                           ycen,
                                                           seeing,
                                                           sersic_n,
                                                           sigma,
                                                           pix_scale,
                                                           psf_factor,
                                                           sersic_factor,
                                                           m_factor,
                                                           light_profile,
                                                           smear)

        model_16 = self.compute_model_grid_fixed_inc_fixed(theta_16,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        model_84 = self.compute_model_grid_fixed_inc_fixed(theta_84,
                                                            inc,
                                                            redshift,
                                                            wave_array,
                                                            xcen,
                                                            ycen,
                                                            seeing,
                                                            sersic_n,
                                                            sigma,
                                                            pix_scale,
                                                            psf_factor,
                                                            sersic_factor,
                                                            m_factor,
                                                            light_profile,
                                                            smear)

        # use the external rot_pa class to extract the 
        # velocity values and x values along the different pa's
        # have to do this for the different model values and the
        # different data values

        # modelled velocity values
        mod_velocity_values_max, x_max = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_max,
                                                 model_max,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        mod_velocity_values_50, x_50 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_50,
                                                 model_50,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        mod_velocity_values_16, x_16 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_16,
                                                 model_16,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        mod_velocity_values_84, x_84 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_84,
                                                 model_84,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        # data velocity values
        real_velocity_values_max, x_max = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_max,
                                                 self.vel_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_velocity_values_50, x_50 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_50,
                                                 self.vel_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_velocity_values_16, x_16 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_16,
                                                 self.vel_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_velocity_values_84, x_84 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_84,
                                                 self.vel_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        # data velocity error values
        real_error_values_max, x_max = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_max,
                                                 self.error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_error_values_50, x_50 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_50,
                                                 self.error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_error_values_16, x_16 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_16,
                                                 self.error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        real_error_values_84, x_84 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_84,
                                                 self.error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        # data sigma values
        sig_values_max, x_max = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_max,
                                                 self.sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_values_50, x_50 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_50,
                                                 self.sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_values_16, x_16 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_16,
                                                 self.sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_values_84, x_84 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_84,
                                                 self.sig_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        # data sigma error values
        sig_error_values_max, x_max = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_max,
                                                 self.sig_error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_error_values_50, x_50 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_50,
                                                 self.sig_error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_error_values_16, x_16 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_16,
                                                 self.sig_error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)

        sig_error_values_84, x_84 = rt_pa.extract(d_aper,
                                                 r_aper,
                                                 pa_84,
                                                 self.sig_error_data,
                                                 xcen,
                                                 ycen,
                                                 pix_scale)



        # plotting the model and extracted quantities

        min_ind = 0
        max_ind = 0

        try:

            while np.isnan(real_velocity_values_50[min_ind]):

                min_ind += 1

        except IndexError:

            min_ind = 0

        try:

            while np.isnan(real_velocity_values_50[::-1][max_ind]):

                max_ind += 1

            max_ind = max_ind + 1

        except IndexError:

            max_ind = 0

        # construct dictionary of these velocity values and 
        # the final distance at which the data is extracted from centre

        extract_d = {'50': [mod_velocity_values_50[min_ind] / np.sin(inc),
                            mod_velocity_values_50[-max_ind] / np.sin(inc)],
                     '16': [mod_velocity_values_16[min_ind] / np.sin(inc),
                            mod_velocity_values_16[-max_ind] / np.sin(inc)],
                     '84': [mod_velocity_values_84[min_ind] / np.sin(inc),
                            mod_velocity_values_84[-max_ind] / np.sin(inc)],
                     'real': [real_velocity_values_50[min_ind] / np.sin(inc),
                              real_velocity_values_50[-max_ind] / np.sin(inc)],
                     'distance': [x_50[min_ind],
                                  x_50[-max_ind]],
                     'vel_error': [real_error_values_50[min_ind],
                                   real_error_values_50[-max_ind]],
                     'vel_max': [np.nanmax(abs(real_velocity_values_50 / np.sin(inc)))],
                     'inclination' : inc,
                     'mod_50_velocity' : mod_velocity_values_50 / np.sin(inc),
                     'mod_50_positions' : x_50,
                     'mod_16_velocity' : mod_velocity_values_16 / np.sin(inc),
                     'mod_16_positions' : x_16,
                     'mod_84_velocity' : mod_velocity_values_84 / np.sin(inc),
                     'mod_84_positions' : x_84}

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(x_max,
                mod_velocity_values_max,
                color='red',
                label='max_model')

        ax.errorbar(x_max,
                    real_velocity_values_max,
                    yerr=real_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.plot(x_50,
                mod_velocity_values_50,
                color='blue',
                label='50_model')

        ax.errorbar(x_50,
                    real_velocity_values_50,
                    yerr=real_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.plot(x_16,
                mod_velocity_values_16,
                color='orange',
                linestyle='--',
                label='16_model')

        ax.plot(x_84,
                mod_velocity_values_84,
                color='purple',
                linestyle='--',
                label='84_model')

        # ax.legend(prop={'size':10})
        ax.set_xlim(-1.5, 1.5)

        # ax.legend(prop={'size':5}, loc=1)

        ax.axhline(0, color='silver', ls='-.')
        ax.axvline(0, color='silver', ls='-.')
        ax.axhline(va_50, color='silver', ls='--')
        ax.axhline(-1.*va_50, color='silver', ls='--')
        ax.set_title('Model and Real Velocity')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        # plt.show()

        fig.savefig('%s_1d_velocity_plot_fixed_inc_fixed.png' % self.fileName[:-5])

        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.errorbar(x_max,
                    sig_values_max,
                    yerr=sig_error_values_max,
                    fmt='o',
                    color='red',
                    label='max_data')

        ax.errorbar(x_50,
                    sig_values_50,
                    yerr=sig_error_values_50,
                    fmt='o',
                    color='blue',
                    label='50_data')

        ax.set_title('Velocity Dispersion')

        ax.set_ylabel('velocity (kms$^{-1}$)')

        ax.set_xlabel('arcsec')

        ax.legend(prop={'size':10})

        plt.close('all')

        # plt.show()

        fig.savefig('%s_1d_dispersion_plot_fixed_inc_fixed.png' % self.fileName[:-5])

        # return the data used in plotting for use elsewhere

        return {'max': [x_max, 
                        mod_velocity_values_max,
                        real_velocity_values_max,
                        real_error_values_max,
                        sig_values_max,
                        sig_error_values_max],
                '50': [x_50, 
                        mod_velocity_values_50,
                        real_velocity_values_50,
                        real_error_values_50,
                        sig_values_50,
                        sig_error_values_50],
                '16': [x_16, 
                        mod_velocity_values_16,
                        real_velocity_values_16,
                        real_error_values_16,
                        sig_values_16,
                        sig_error_values_16],
                '84': [x_84, 
                        mod_velocity_values_84,
                        real_velocity_values_84,
                        real_error_values_84,
                        sig_values_84,
                        sig_error_values_84]}, extract_d

    def v_over_sigma(self,
                     inc,
                     redshift,
                     wave_array,
                     xcen,
                     ycen,
                     i_option,
                     sig_option,
                     r_aper,
                     d_aper,
                     seeing,
                     sersic_n,
                     sigma,
                     pix_scale,
                     psf_factor,
                     sersic_factor,
                     m_factor,
                     light_profile,
                     smear=False):

        """
        Def:
        Use the different versions of the extract in apertures method
        to look into the v_over_sigma ratios of given objects.
        This is controlled by the i_option which can be one of either
        free, fixed, fixed_vary or fixed_fixed. The sigma option dictates
        how sigma is computed, whether we use a mean, or a flux weighted mean,
        or central value or outskirts value.

        Input:
                i_option - inclination option, one of the values listed in the
                            definition
                sig_option - how do we compute sigma
                r_aper - apertures size in pixels
                d_aper - distance between adjacent apertures in pixels

        Output:
                v_over_sigma ratios for the model 50th, 16th and 84th pcntile,
                as well as for the real data.
        """

        # dont do anything if an invalid i_option or sig_option is provided

        if not((i_option == 'free' or i_option == 'fixed' or
                i_option == 'fixed_vary' or i_option == 'fixed_fixed') and
               (sig_option == 'mean' or
                sig_option == 'weighted_mean' or sig_option == 'median')):

            raise ValueError('Choose valid inclination and sigma options')

        if sig_option == 'mean':

            indices = ~np.isnan(self.sig_data)

            sigma_o = np.average(self.sig_data[indices],
                                 weights=1.0 / self.sig_error_data[indices])

            sigma_e = np.average(self.sig_error_data[indices],
                                 weights=1.0 / self.sig_error_data[indices])

        elif sig_option == 'median':

            sigma_o = np.nanmedian(self.sig_data)

            sigma_e = np.nanmedian(self.sig_error_data)

        if i_option == 'free':

            param_file = np.genfromtxt(self.param_file)

            theta_50 = param_file[2][1:]
            theta_16 = param_file[3][1:]
            theta_84 = param_file[4][1:]

            r_half_50 = theta_50[4]
            r_half_16 = theta_16[4]
            r_half_84 = theta_84[4]

            other, e_val = self.extract_in_apertures(r_aper,
                                                     d_aper,
                                                     seeing,
                                                     pix_scale,
                                                     psf_factor,
                                                     smear)

        elif i_option == 'fixed':

            param_file = np.genfromtxt(self.param_file_fixed)

            theta_50 = param_file[2][1:]
            theta_16 = param_file[3][1:]
            theta_84 = param_file[4][1:]

            r_half_50 = theta_50[2]
            r_half_16 = theta_16[2]
            r_half_84 = theta_84[2]

            other, e_val = self.extract_in_apertures_fixed(xcen,
                                                           ycen,
                                                           r_aper,
                                                           d_aper,
                                                           seeing,
                                                           pix_scale,
                                                           psf_factor,
                                                           smear,
                                                           vary=False)

        elif i_option == 'fixed_vary':

            param_file = np.genfromtxt(self.param_file_fixed_inc_vary)

            theta_50 = param_file[2][1:]
            theta_16 = param_file[3][1:]
            theta_84 = param_file[4][1:]

            r_half_50 = theta_50[2]
            r_half_16 = theta_16[2]
            r_half_84 = theta_84[2]

            other, e_val = self.extract_in_apertures_fixed(xcen,
                                                           ycen,
                                                           r_aper,
                                                           d_aper,
                                                           seeing,
                                                           pix_scale,
                                                           psf_factor,
                                                           smear,
                                                           vary=True)

        elif i_option == 'fixed_fixed':

            param_file = np.genfromtxt(self.param_file_fixed_inc_fixed)

            theta_50 = param_file[2][1:]
            theta_16 = param_file[3][1:]
            theta_84 = param_file[4][1:]

            r_half_50 = theta_50[1]
            r_half_16 = theta_16[1]
            r_half_84 = theta_84[1]

            other, e_val = self.extract_in_apertures_fixed_inc_fixed(inc,
                                                                     redshift,
                                                                     wave_array,
                                                                     xcen,
                                                                     ycen,
                                                                     r_aper,
                                                                     d_aper,
                                                                     seeing,
                                                                     sersic_n,
                                                                     sigma,
                                                                     pix_scale,
                                                                     psf_factor,
                                                                     sersic_factor,
                                                                     m_factor,
                                                                     light_profile,
                                                                     smear)

        # evaluate the v2.2 parameter for each of the models
        # need to check if 2.2*r_half is greater than the positive
        # and negative range of positions

        # r_22_50

        r_22_50 = pix_scale * 2.2 * r_half_50

        pos_bigger = r_22_50 > e_val['mod_50_positions'][-1]
        neg_bigger = -1.*r_22_50 < e_val['mod_50_positions'][0]

        if pos_bigger and neg_bigger:

            v_22_50 = abs(e_val['mod_50_velocity'][0])

        elif pos_bigger and not(neg_bigger):

            v_22_idx_50 = (np.abs(r_22_50 - e_val['mod_50_positions'])).argmin()
            v_22_50 = abs(e_val['mod_50_velocity'][v_22_idx_50])

        elif (not(pos_bigger) and neg_bigger) or (not(pos_bigger) and not(neg_bigger)):

            v_22_idx_50 = (np.abs(e_val['mod_50_positions'] - r_22_50)).argmin()
            v_22_50 = abs(e_val['mod_50_velocity'][v_22_idx_50])

        # r_22_16

        r_22_16 = pix_scale * 2.2 * r_half_16

        pos_bigger = r_22_16 > e_val['mod_16_positions'][-1]
        neg_bigger = -1.*r_22_16 < e_val['mod_16_positions'][0]

        if pos_bigger and neg_bigger:

            v_22_16 = abs(e_val['mod_16_velocity'][0])

        elif pos_bigger and not(neg_bigger):

            v_22_idx_16 = (np.abs(r_22_16 - e_val['mod_16_positions'])).argmin()
            v_22_16 = abs(e_val['mod_16_velocity'][v_22_idx_16])

        elif (not(pos_bigger) and neg_bigger) or (not(pos_bigger) and not(neg_bigger)):

            v_22_idx_16 = (np.abs(e_val['mod_16_positions'] - r_22_16)).argmin()
            v_22_16 = abs(e_val['mod_16_velocity'][v_22_idx_16])

        # r_22_84

        r_22_84 = pix_scale * 2.2 * r_half_84

        pos_bigger = r_22_84 > e_val['mod_84_positions'][-1]
        neg_bigger = -1.*r_22_84 < e_val['mod_84_positions'][0]

        if pos_bigger and neg_bigger:

            v_22_84 = abs(e_val['mod_84_velocity'][0])

        elif pos_bigger and not(neg_bigger):

            v_22_idx_84 = (np.abs(r_22_84 - e_val['mod_84_positions'])).argmin()
            v_22_84 = abs(e_val['mod_84_velocity'][v_22_idx_84])

        elif (not(pos_bigger) and neg_bigger) or (not(pos_bigger) and not(neg_bigger)):

            v_22_idx_84 = (np.abs(e_val['mod_84_positions'] - r_22_84)).argmin()
            v_22_84 = abs(e_val['mod_84_velocity'][v_22_idx_84])

        mod_50_min = e_val['50'][0]
        mod_50_max = e_val['50'][1]
        mod_50_v = (abs(mod_50_min) + abs(mod_50_max)) / 2.0

        mod_16_min = e_val['16'][0]
        mod_16_max = e_val['16'][1]
        mod_16_v = (abs(mod_16_min) + abs(mod_16_max)) / 2.0

        mod_84_min = e_val['84'][0]
        mod_84_max = e_val['84'][1]
        mod_84_v = (abs(mod_84_min) + abs(mod_84_max)) / 2.0

        real_min = e_val['real'][0]
        real_max = e_val['real'][1]
        real_v = (abs(real_min) + abs(real_max)) / 2.0

        error_v_min = e_val['vel_error'][0]
        error_v_max = e_val['vel_error'][1]

        max_v_value = e_val['vel_max'][0]

        min_d_value = e_val['distance'][0]
        max_d_value = e_val['distance'][1]

        return [real_v / sigma_o,
                mod_50_v / sigma_o,
                mod_84_v / sigma_o,
                mod_16_v / sigma_o,
                sigma_o,
                sigma_e,
                error_v_min,
                error_v_max,
                max_v_value,
                min_d_value,
                max_d_value,
                e_val['inclination'],
                r_half_16,
                r_half_50,
                r_half_84,
                v_22_16,
                v_22_50,
                v_22_84]


# genetic algorithm attempt
    
    def individual(self,
                   xcen,
                   ycen,
                   inc,
                   pa,
                   rt,
                   va):

        """
        Def: Construct an individual member of the population.
        For starting will used fixed standard deviation values and
        will be careful not to stray into the unphysical.

        Input:
                infile - standard all_names_new.txt file

        Output:
                parameter list - i.e. an individual of the population
        """

        # construct each parameter individually by drawing
        # from gaussians of fixed standard deviation and
        # input given by the central values above

        xparam = np.random.normal(xcen, 0.1)

        # monitor whether things get unphysical
        # if outside the limits - return the central value
        # if it does go outside

        if xparam < 0 or xparam > 32:

            xparam = xcen

        yparam = np.random.normal(ycen, 0.1)

        if yparam < 0 or yparam > 32:

            yparam = ycen

        inc_param = inc

        pa_param = np.random.normal(pa, 0.01)

        # account for the fact that if less than zero it should be 2pi
        # plus that which is the used parameter

        if pa_param < 0:

            pa_param = (2 * np.pi) + pa_param

        rt_param = np.random.normal(rt, 0.5)

        if rt_param < 0:

            rt_param = rt

        va_param = np.random.normal(va, 1)

        if va_param < 0:

            va_param = va

        return [xparam, yparam, inc_param, pa_param, rt_param, va_param]

    def population(self,
                   count,
                   xcen,
                   ycen,
                   inc,
                   pa,
                   rt,
                   va):

        """
        Def: Return *count* individuals as a list of lists

        Input:
                count - number of individuals to return

        output: List of lists. The population.
        """

        return [self.individual(xcen,
                                ycen,
                                inc,
                                pa,
                                rt,
                                va) for x in xrange(count)]

    def fitness(self,
                individual,
                psf_factor):

        """
        Def:
        Determine the fitness of an individual from chi-squared. So values
        which are low are desirable.

        Input:
                Individual - set of parameters to feed to the model

        Output:
                Chi-squared value - the fitness result
        """

        model = self.compute_model_grid(individual,
                                        seeing,
                                        pix_scale,
                                        psf_factor)

        # find the grid of inverse sigma values

        inv_sigma2 = 1.0 / (self.error_data * self.error_data)

        ans = np.nansum((self.vel_data - model) * (self.vel_data - model) *
                            inv_sigma2)

        # print ans

        return ans

    def grade(self,
              population):

        """
        Def:
        Return the average fitness of a population.
        """
        return np.nanmean([self.fitness(entry) for entry in population])

    def evolve(self,
               population,
               retain=0.2,
               random_select=0.05,
               mutate=0.01):

        """
        Def:
        Evolve a population from one generation to the next so that the
        overall population becomes increasingly fit

        Input:
                retain - fraction to survive without crossover
                random_select - fraction of bad guys selected to promote
                                genetic diversity
                mutate - fraction to randomly mutate to promote genetic
                            diversity

        Output:
                population - the evolved population after fitness evaluation,
                                crossover, random selection and mutation
        """

        # find the mean properties of the population
        xcen = np.mean([entry[0] for entry in population])
        ycen = np.mean([entry[1] for entry in population])
        inc = np.mean([entry[2] for entry in population])
        pa = np.mean([entry[3] for entry in population])
        rt = np.mean([entry[4] for entry in population])
        va = np.mean([entry[5] for entry in population])

        # first create the graded fitness list

        graded = [(self.fitness(x), x) for x in population]

        graded = [x[1] for x in sorted(graded)]

        # retain the best guys in the population

        retain_length = int(len(graded)*retain)

        parents = graded[:retain_length]

        # randomly add others

        for individual in graded[retain_length:]:

            if random_select > np.random.random():

                parents.append(individual)

        # mutate some individuals - how do we do this?

        for individual in parents:

            if mutate > np.random.random():

                pos_to_mutate = np.random.randint(0, len(individual) - 1)

                if pos_to_mutate == 0:

                    xparam = np.random.normal(xcen, 0.2)

                    # monitor whether things get unphysical
                    # if outside the limits - return the central value
                    # if it does go outside

                    if xparam < 0 or xparam > 32:

                        xparam = xcen

                    individual[pos_to_mutate] = xparam

                elif pos_to_mutate == 1:

                    yparam = np.random.normal(ycen, 0.2)

                    if yparam < 0 or yparam > 32:

                        yparam = ycen

                    individual[pos_to_mutate] = yparam

                elif pos_to_mutate == 2:

                    inc_param = inc

                    individual[pos_to_mutate] = inc_param

                elif pos_to_mutate == 3:

                    pa_param = np.random.normal(pa, 0.1)

                # account for the fact that if less than zero it should be 2pi
                # plus that which is the used parameter

                    if pa_param < 0:

                        pa_param = (2 * np.pi) + pa_param

                    individual[pos_to_mutate] = pa_param

                elif pos_to_mutate == 4:

                    rt_param = np.random.normal(rt, 1)

                    if rt_param < 0:

                        rt_param = rt

                    individual[pos_to_mutate] = rt_param

                elif pos_to_mutate == 5:

                    va_param = np.random.normal(va, 5)

                    if va_param < 0:

                        va_param = va

                    individual[pos_to_mutate] = va_param

        # crossover parents to create children

        parents_length = len(parents)

        desired_length = len(population) - parents_length

        children = []

        while len(children) < desired_length:

            male = np.random.randint(0, parents_length-1)

            female = np.random.randint(0, parents_length-1)

            if male != female:

                male = parents[male]

                female = parents[female]

                half = len(male) / 2

                child = male[:half] + female[half:]

                children.append(child)

        parents.extend(children)

        grade = self.grade(parents)

        print grade

        return parents

