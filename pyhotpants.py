import glob
import os
import sys
import warnings

import numpy as np
import ccdproc
import photutils
from astropy import units as u
from astropy.convolution import convolve_fft as convolve
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.nddata import CCDData, NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from photutils.psf import extract_stars


def genrate_file_list(input_path, output_path, filetype='fits'):
    file_list = glob.glob(os.path.join(input_path, "*." + filetype))
    np.savetxt(os.path.join(output_path, 'file_list.txt'), file_list, fmt='%s')
    return file_list


def correlate2d(im1, im2, boundary='wrap', nthreads=1, **kwargs):
    """
    taken from
    https://github.com/keflavich/image_registration
    https://image-registration.readthedocs.io/en/latest/image_registration.html

    Cross-correlation of two images of arbitrary size.  Returns an image
    cropped to the largest of each dimension of the input images
    Parameters
    ----------
    return_fft - if true, return fft(im1)*fft(im2[::-1,::-1]), which is the power
        spectral density
    fftshift - if true, return the shifted psd so that the DC component is in
        the center of the image
    pad - Default on.  Zero-pad image to the nearest 2^n
    crop - Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the largest
        image in both directions.
    boundary: str, optional
        A flag indicating how to handle boundaries:
            * 'fill' : set values outside the array boundary to fill_value
                       (default)
            * 'wrap' : periodic boundary
    nthreads : bool
        Number of threads to use for fft (only matters if you have fftw
        installed)
    WARNING: Normalization may be arbitrary if you use the PSD
    """

    fftn, ifftn = np.fft.fftn, np.fft.ifftn

    return convolve(np.conjugate(im1),
                    im2[::-1, ::-1],
                    normalize_kernel=False,
                    fftn=fftn,
                    ifftn=ifftn,
                    boundary=boundary,
                    nan_treatment='fill',
                    **kwargs)


def second_derivative(image):
    """
    taken from
    https://github.com/keflavich/image_registration
    https://image-registration.readthedocs.io/en/latest/image_registration.html

    Compute the second derivative of an image
    The derivatives are set to zero at the edges
    Parameters
    ----------
    image: np.ndarray
    Returns
    -------
    d/dx^2, d/dy^2, d/dxdy
    All three are np.ndarrays with the same shape as image.
    """
    shift_right = np.roll(image, 1, 1)
    shift_right[:, 0] = 0
    shift_left = np.roll(image, -1, 1)
    shift_left[:, -1] = 0
    shift_down = np.roll(image, 1, 0)
    shift_down[0, :] = 0
    shift_up = np.roll(image, -1, 0)
    shift_up[-1, :] = 0

    shift_up_right = np.roll(shift_up, 1, 1)
    shift_up_right[:, 0] = 0
    shift_down_left = np.roll(shift_down, -1, 1)
    shift_down_left[:, -1] = 0
    shift_down_right = np.roll(shift_right, 1, 0)
    shift_down_right[0, :] = 0
    shift_up_left = np.roll(shift_left, -1, 0)
    shift_up_left[-1, :] = 0

    dxx = shift_right + shift_left - 2 * image
    dyy = shift_up + shift_down - 2 * image
    dxy = 0.25 * (shift_up_right + shift_down_left - shift_up_left -
                  shift_down_right)

    return dxx, dyy, dxy


def cross_correlation_shifts(image1,
                             image2,
                             errim1=None,
                             errim2=None,
                             maxoff=None,
                             verbose=False,
                             gaussfit=False,
                             return_error=False,
                             zeromean=True,
                             **kwargs):
    """
    taken from
    https://github.com/keflavich/image_registration
    https://image-registration.readthedocs.io/en/latest/image_registration.html

    Use cross-correlation and a 2nd order taylor expansion to measure the
    offset between two images
    Given two images, calculate the amount image2 is offset from image1 to
    sub-pixel accuracy using 2nd order taylor expansion.
    Parameters
    ----------
    image1: np.ndarray
        The reference image
    image2: np.ndarray
        The offset image.  Must have the same shape as image1
    errim1: np.ndarray [optional]
        The pixel-by-pixel error on the reference image
    errim2: np.ndarray [optional]
        The pixel-by-pixel error on the offset image.
    maxoff: int
        Maximum allowed offset (in pixels).  Useful for low s/n images that you
        know are reasonably well-aligned, but might find incorrect offsets due to
        edge noise
    zeromean : bool
        Subtract the mean from each image before performing cross-correlation?
    verbose: bool
        Print out extra messages?
    gaussfit : bool
        Use a Gaussian fitter to fit the peak of the cross-correlation?
    return_error: bool
        Return an estimate of the error on the shifts.  WARNING: I still don't
        understand how to make these agree with simulations.
        The analytic estimate comes from
        http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        At high signal-to-noise, the analytic version overestimates the error
        by a factor of about 1.8, while the gaussian version overestimates
        error by about 1.15.  At low s/n, they both UNDERestimate the error.
        The transition zone occurs at a *total* S/N ~ 1000 (i.e., the total
        signal in the map divided by the standard deviation of the map -
        it depends on how many pixels have signal)
    **kwargs are passed to correlate2d, which in turn passes them to convolve.
    The available options include image padding for speed and ignoring NaNs.
    References
    ----------
    From http://solarmuri.ssl.berkeley.edu/~welsch/public/software/cross_cor_taylor.pro
    Examples
    --------
    >>> import numpy as np
    >>> im1 = np.zeros([10,10])
    >>> im2 = np.zeros([10,10])
    >>> im1[4,3] = 1
    >>> im2[5,5] = 1
    >>> import image_registration
    >>> yoff,xoff = image_registration.cross_correlation_shifts(im1,im2)
    >>> im1_aligned_to_im2 = np.roll(np.roll(im1,int(yoff),1),int(xoff),0)
    >>> assert (im1_aligned_to_im2-im2).sum() == 0
    """

    if not image1.shape == image2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        image1 = image1 - (image1[image1 == image1].mean())
        image2 = image2 - (image2[image2 == image2].mean())

    image1 = np.nan_to_num(image1)
    image2 = np.nan_to_num(image2)

    quiet = kwargs.pop('quiet') if 'quiet' in kwargs else not verbose
    ccorr = (correlate2d(image1, image2, **kwargs) / image1.size)
    # allow for NaNs set by convolve (i.e., ignored pixels)
    ccorr[ccorr != ccorr] = 0
    if ccorr.shape != image1.shape:
        raise ValueError(
            "Cross-correlation image must have same shape as input images.  This can only be violated if you pass a strange kwarg to correlate2d."
        )

    ylen, xlen = image1.shape
    xcen = xlen / 2 - (1 - xlen % 2)
    ycen = ylen / 2 - (1 - ylen % 2)

    if ccorr.max() == 0:
        warnings.warn("WARNING: No signal found!  Offset is defaulting to 0,0")
        return 0, 0

    if maxoff is not None:
        if verbose: print("Limiting maximum offset to %i" % maxoff)
        subccorr = ccorr[ycen - maxoff:ycen + maxoff + 1,
                         xcen - maxoff:xcen + maxoff + 1]
        ymax, xmax = np.unravel_index(subccorr.argmax(), subccorr.shape)
        xmax = xmax + xcen - maxoff
        ymax = ymax + ycen - maxoff
    else:
        ymax, xmax = np.unravel_index(ccorr.argmax(), ccorr.shape)
        subccorr = ccorr

    if return_error:
        if errim1 is None:
            errim1 = np.ones(ccorr.shape) * image1[image1 == image1].std()
        if errim2 is None:
            errim2 = np.ones(ccorr.shape) * image2[image2 == image2].std()
        eccorr = (
            (correlate2d(errim1**2, image2**2, quiet=quiet, **kwargs) +
             correlate2d(errim2**2, image1**2, quiet=quiet, **kwargs))**0.5 /
            image1.size)
        if maxoff is not None:
            subeccorr = eccorr[ycen - maxoff:ycen + maxoff + 1,
                               xcen - maxoff:xcen + maxoff + 1]
        else:
            subeccorr = eccorr

    if gaussfit:
        try:
            from agpy import gaussfitter
        except ImportError:
            raise ImportError(
                "Couldn't import agpy.gaussfitter; try using cross_correlation_shifts with gaussfit=False"
            )
        if return_error:
            pars, epars = gaussfitter.gaussfit(subccorr,
                                               err=subeccorr,
                                               return_all=True)
            exshift = epars[2]
            eyshift = epars[3]
        else:
            pars, epars = gaussfitter.gaussfit(subccorr, return_all=True)
        xshift = maxoff - pars[2] if maxoff is not None else xcen - pars[2]
        yshift = maxoff - pars[3] if maxoff is not None else ycen - pars[3]
        if verbose:
            print("Gaussian fit pars: ", xshift, yshift, epars[2], epars[3],
                  pars[4], pars[5], epars[4], epars[5])

    else:

        xshift_int = xmax - xcen
        yshift_int = ymax - ycen

        local_values = ccorr[ymax - 1:ymax + 2, xmax - 1:xmax + 2]

        d1y, d1x = np.gradient(local_values)
        d2y, d2x, dxy = second_derivative(local_values)

        fx, fy, fxx, fyy, fxy = d1x[1, 1], d1y[1, 1], d2x[1, 1], d2y[1,
                                                                     1], dxy[1,
                                                                             1]
        shiftsubx = (fyy * fx - fy * fxy) / (fxy**2 - fxx * fyy)
        shiftsuby = (fxx * fy - fx * fxy) / (fxy**2 - fxx * fyy)

        xshift = -(xshift_int + shiftsubx)
        yshift = -(yshift_int + shiftsuby)

        # http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        # Zucker error

        if return_error:
            #acorr1 = (correlate2d(image1,image1,quiet=quiet,**kwargs) / image1.size)
            #acorr2 = (correlate2d(image2,image2,quiet=quiet,**kwargs) / image2.size)
            #ccorrn = ccorr / eccorr**2 / ccorr.size #/ (errim1.mean()*errim2.mean()) #/ eccorr**2
            normalization = 1. / ((image1**2).sum() / image1.size) / (
                (image2**2).sum() / image2.size)
            ccorrn = ccorr * normalization
            exshift = (np.abs(
                -1 * ccorrn.size * fxx * normalization / ccorrn[ymax, xmax] *
                (ccorrn[ymax, xmax]**2 / (1 - ccorrn[ymax, xmax]**2)))**-0.5)
            eyshift = (np.abs(
                -1 * ccorrn.size * fyy * normalization / ccorrn[ymax, xmax] *
                (ccorrn[ymax, xmax]**2 / (1 - ccorrn[ymax, xmax]**2)))**-0.5)
            if np.isnan(exshift):
                raise ValueError("Error: NAN error!")

    if return_error:
        return xshift, yshift, exshift, eyshift
    else:
        return xshift, yshift


def align_images(ref,
                 file_list,
                 output_path,
                 overwrite,
                 xl=0,
                 xr=0,
                 yb=0,
                 yt=0,
                 stack=False,
                 stack_sigma_clip_low=2,
                 stack_sigma_clip_high=5,
                 stack_sigma_clip_func=np.ma.median,
                 stack_clip_extrema_low=1,
                 stack_clip_extrema_high=1):
    '''
    Parameters
    ----------
    ref : string
        File path to the reference image
    file_list : string
        List of file paths of the images to be aligned
    xl : int
        Trim left
    xr : int
        Trim right
    yb : int
        Trim bottom
    yt : int
        Trim top
    '''

    # open the reference image
    f_ref = fits.open(ref)[-1]
    header_ref = f_ref.header
    image_ref = f_ref.data
    aligned_file_list = []

    if stack:
        combiner_list = []

    for i, f_to_align in enumerate(file_list):
        print('Aligning image ' + str(i + 1) + ' of ' + str(len(file_list)) +
              '.')
        f = fits.open(f_to_align)[-1]
        header = f.header
        image = f.data
        header['NAXIS1'] -= (xl + xr)
        header['NAXIS2'] -= (yb + yt)

        # 2d cross correlate two frames
        yoff, xoff = cross_correlation_shifts(
            image[yb:yb + header['NAXIS2'], xl:xl + header['NAXIS1']],
            image_ref[yb:yb + header['NAXIS2'], xl:xl + header['NAXIS1']])

        # shift the image to the nearest pixel
        image_aligned_to_ref = np.roll(np.roll(image, int(yoff), 1), int(xoff),
                                       0)[yb:yb + header['NAXIS2'], xl:xl +
                                          header['NAXIS1']].astype('float32')

        # append to the combiner if stacking
        if stack:
            combiner_list.append(CCDData(image_aligned_to_ref, unit=u.ct))

        # Set output file name and save as fit
        outfile_name = f_to_align.split('.')[0] + '_aligned.fits'
        outfile_path = os.path.join(output_path, outfile_name.split('/')[-1])
        aligned_file_list.append(outfile_path)
        fits.writeto(outfile_path,
                     image_aligned_to_ref,
                     header=header,
                     overwrite=overwrite)

    np.savetxt(os.path.join(output_path, 'aligned_file_list.txt'),
               aligned_file_list, fmt='%s')

    # return the file list for the aligned images, and the combiner if stacking
    if stack:
        combiner = ccdproc.Combiner(combiner_list)
        combiner.sigma_clipping(low_thresh=stack_sigma_clip_low,
                                high_thresh=stack_sigma_clip_high,
                                func=stack_sigma_clip_func)
        combiner.clip_extrema(nlow=stack_clip_extrema_low,
                              nhigh=stack_clip_extrema_high)

        return aligned_file_list, combiner

    else:

        return aligned_file_list


def get_background(data, box_size=(50, 50), filter_size=(3, 3), sigma=3.):
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(data,
                                 box_size=box_size,
                                 filter_size=filter_size,
                                 sigma_clip=sigma_clip,
                                 bkg_estimator=bkg_estimator)
    return bkg

def build_psf(data,
              threshold=100.,
              size=25,
              epsf_oversampling=4,
              epsf_maxiters=3,
              epsf_progress_bar=True,
              show_stamps=False,
              stamps_nrows=5,
              stamps_ncols=5,
              figsize=(20, 20)):
    '''
    data is best background subtracted
    '''
    # detect peaks
    peaks_tbl = photutils.find_peaks(data, threshold=threshold)

    # pick isolated point sources
    hsize = (size - 1) // 2
    x = peaks_tbl['x_peak']
    y = peaks_tbl['y_peak']
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) &
            (y < (data.shape[0] - 1 - hsize)))

    stars_tbl = Table()
    stars_tbl['x'] = x[mask]
    stars_tbl['y'] = y[mask]

    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=size)
    epsf_builder = photutils.EPSFBuilder(oversampling=epsf_oversampling,
                                         maxiters=epsf_maxiters,
                                         progress_bar=epsf_progress_bar)
    epsf, fitted_stars = epsf_builder(stars)

    if show_stamps:
        nrows = stamps_nrows
        ncols = stamps_ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                               squeeze=True)
        ax = ax.ravel()
        for i in range(nrows*ncols):
            norm = simple_norm(stars[i], 'log', percent=99.)
            ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

        plt.show()

    return epsf, fitted_stars


def get_fwhm():
    # see also https://photutils.readthedocs.io/en/stable/detection.html
    pass

def find_star():
    # see also https://photutils.readthedocs.io/en/stable/detection.html
    photutils.DAOStarFinder()
    # photutils.IRAFStarFinder()
    pass

def do_photometry():
    # see also https://photutils.readthedocs.io/en/latest/psf.html
    pass


