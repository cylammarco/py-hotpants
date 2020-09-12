import glob
import os
import subprocess
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
from scipy.optimize import curve_fit


def generate_file_list(input_path, output_path, filetype='fits'):
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
               aligned_file_list,
               fmt='%s')

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


def get_stars(data, stars_tbl=None, threshold=100., size=25):
    if stars_tbl is None:
        # detect peaks
        peaks_tbl = photutils.find_peaks(data, threshold=threshold)
        peaks_sort_mask = np.argsort(-peaks_tbl['peak_value'])

        # remove sources near the edge
        hsize = (size - 1) // 2
        x = peaks_tbl['x_peak'][peaks_sort_mask]
        y = peaks_tbl['y_peak'][peaks_sort_mask]
        mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) &
                (y < (data.shape[0] - 1 - hsize)))

        stars_tbl = Table()
        stars_tbl['x'] = x[mask]
        stars_tbl['y'] = y[mask]

    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=size)

    return stars, stars_tbl


def build_psf(stars,
              epsf_oversampling=4,
              epsf_smoothing_kernel='quartic',
              epsf_recentering_maxiters=20,
              epsf_maxiters=10,
              epsf_progress_bar=True,
              epsf_norm_radius=5.5,
              epsf_shift_val=0.5,
              epsf_recentering_boxsize=(5, 5),
              epsf_center_accuracy=0.001,
              show_stamps=False,
              stamps_nrows=5,
              stamps_ncols=5,
              figsize=(10, 10)):
    '''
    data is best background subtracted
    '''
    epsf_builder = photutils.EPSFBuilder(
        oversampling=epsf_oversampling,
        recentering_maxiters=epsf_recentering_maxiters,
        maxiters=epsf_maxiters,
        progress_bar=epsf_progress_bar)
    epsf, fitted_stars = epsf_builder(stars)

    if show_stamps:
        nrows = stamps_nrows
        ncols = stamps_ncols
        fig, ax = plt.subplots(nrows=nrows,
                               ncols=ncols,
                               figsize=figsize,
                               squeeze=True)
        ax = ax.ravel()
        for i in range(nrows * ncols):
            norm = simple_norm(stars[i], 'log', percent=99.)
            try:
                ax[i].imshow(stars[i],
                             norm=norm,
                             origin='lower',
                             cmap='viridis')
            except:
                pass
        plt.show()

    return epsf, fitted_stars


def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function.
    Parameters
    ----------
    x: float or 1-d numpy array
        The data to evaluate the Gaussian over
    a: float
        the amplitude
    b: float
        the constant offset
    x0: float
        the center of the Gaussian
    sigma: float
        the width of the Gaussian
    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def fit_gaussian_for_fwhm(psf, sigma=False):
    '''
    Fit gaussians to the psf
    Patameters
    ----------
    psf: np.ndarray.array
        The 2D PSF profile
    sigma: boolean
        Set to True to return sigma instead of the FWHM
    '''
    psf_x = np.sum(psf.data, axis=0)
    psf_y = np.sum(psf.data, axis=1)
    pguess_x = max(psf_x), 0, len(psf_x) / 2., len(psf_x) / 10.
    pguess_y = max(psf_y), 0, len(psf_y) / 2., len(psf_y) / 10.
    # see also https://photutils.readthedocs.io/en/stable/detection.html
    popt_x, _ = curve_fit(_gaus, np.arange(len(psf_x)), psf_x, p0=pguess_x)
    popt_y, _ = curve_fit(_gaus, np.arange(len(psf_y)), psf_y, p0=pguess_y)
    sigma_x = popt_x[3]
    sigma_y = popt_y[3]
    if sigma:
        return sigma_x, sigma_y
    else:
        return sigma_x * 2.355, sigma_y * 2.355


def get_all_fwhm(file_list,
                 stars_tbl,
                 sigma=False,
                 bkg_box_size=(50, 50),
                 bkg_filter_size=(5, 5),
                 bkg_sigma=3.,
                 stars_threshold=500.,
                 stars_size=25,
                 epsf_oversampling=4,
                 epsf_smoothing_kernel='quartic',
                 epsf_recentering_maxiters=20,
                 epsf_maxiters=20,
                 epsf_progress_bar=True,
                 epsf_norm_radius=5.5,
                 epsf_shift_val=0.5,
                 epsf_recentering_boxsize=(5, 5),
                 epsf_center_accuracy=0.001,
                 show_stamps=False,
                 stamps_nrows=5,
                 stamps_ncols=5,
                 figsize=(10, 10)):
    sigma_x = []
    sigma_y = []
    for filepath in file_list:
        data_i = fits.open(filepath)[0].data
        bkg_i = get_background(data_i,
                               box_size=bkg_box_size,
                               filter_size=bkg_filter_size,
                               sigma=bkg_sigma)
        stars_i, _ = get_stars(data_i - bkg_i.background,
                               stars_tbl=stars_tbl,
                               threshold=stars_threshold,
                               size=stars_size)
        # build the psf using the stacked image
        # see also https://photutils.readthedocs.io/en/stable/epsf.html
        epsf_i, _ = build_psf(
            stars_i,
            epsf_oversampling=epsf_oversampling,
            epsf_smoothing_kernel=epsf_smoothing_kernel,
            epsf_recentering_maxiters=epsf_recentering_maxiters,
            epsf_maxiters=epsf_maxiters,
            epsf_progress_bar=epsf_progress_bar,
            epsf_norm_radius=epsf_norm_radius,
            epsf_shift_val=epsf_shift_val,
            epsf_recentering_boxsize=epsf_recentering_boxsize,
            epsf_center_accuracy=epsf_center_accuracy,
            show_stamps=show_stamps,
            stamps_nrows=stamps_nrows,
            stamps_ncols=stamps_ncols)
        sigma = fit_gaussian_for_fwhm(epsf_i.data, sigma=sigma)
        if sigma:
            sigma_x.append(sigma[0] / epsf_oversampling)
            sigma_y.append(sigma[1] / epsf_oversampling)
        else:
            sigma_x.append(sigma[0])
            sigma_y.append(sigma[1])

    return sigma_x, sigma_y


def find_star():
    # see also https://photutils.readthedocs.io/en/stable/detection.html
    photutils.DAOStarFinder()
    # photutils.IRAFStarFinder()
    pass


def do_photometry():
    # see also https://photutils.readthedocs.io/en/latest/psf.html
    pass


def generate_hotpants_script(ref_path,
                             aligned_file_list,
                             sigma_ref,
                             sigma_list,
                             shell='/bin/bash',
                             hotpants='hotpants',
                             extension='fits',
                             filename='diff_image.sh',
                             overwrite=True,
                             tu=None,
                             tuk=None,
                             tl=None,
                             tg=None,
                             tr=None,
                             tp=None,
                             tni=None,
                             tmi=None,
                             iu=None,
                             iuk=None,
                             il=None,
                             ig=None,
                             ir=None,
                             ip=None,
                             ini=None,
                             imi=None,
                             ki=None,
                             r=None,
                             kcs=None,
                             ft=None,
                             sft=None,
                             nft=None,
                             vmins=None,
                             mous=None,
                             omi=None,
                             gd=None,
                             nrx=None,
                             nry=None,
                             rf=None,
                             rkw=None,
                             nsx=None,
                             nsy=None,
                             ssf=None,
                             cmpfile=None,
                             afssc=None,
                             nss=None,
                             rss=None,
                             savexy=None,
                             n=None,
                             fom=None,
                             sconv=None,
                             ko=None,
                             bgo=None,
                             ssig=None,
                             ks=None,
                             kfm=None,
                             okn=None,
                             fi=None,
                             fin=None,
                             convvar=None,
                             oni=None,
                             ond=None,
                             nim=None,
                             ndm=None,
                             oci=None,
                             cim=None,
                             allm=False,
                             nc=None,
                             hki=None,
                             oki=None,
                             sht=None,
                             obs=None,
                             obz=None,
                             nsht=None,
                             nbs=None,
                             nbz=None,
                             pca=None,
                             v=None):
    '''

    -c and -ng are not available because they are always used based on the
    sigma_ref and sigma_list.

    Version 5.1.11
    Required options:
    [-inim fitsfile]  : comparison image to be differenced
    [-tmplim fitsfile]: template image
    [-outim fitsfile] : output difference image

    Additional options:
    [-tu tuthresh]    : upper valid data count, template (25000)
    [-tuk tucthresh]  : upper valid data count for kernel, template (tuthresh)
    [-tl tlthresh]    : lower valid data count, template (0)
    [-tg tgain]       : gain in template (1)
    [-tr trdnoise]    : e- readnoise in template (0)
    [-tp tpedestal]   : ADU pedestal in template (0)
    [-tni fitsfile]   : input template noise array (undef)
    [-tmi fitsfile]   : input template mask image (undef)
    [-iu iuthresh]    : upper valid data count, image (25000)
    [-iuk iucthresh]  : upper valid data count for kernel, image (iuthresh)
    [-il ilthresh]    : lower valid data count, image (0)
    [-ig igain]       : gain in image (1)
    [-ir irdnoise]    : e- readnoise in image (0)
    [-ip ipedestal]   : ADU pedestal in image (0)
    [-ini fitsfile]   : input image noise array (undef)
    [-imi fitsfile]   : input image mask image (undef)

    [-ki fitsfile]    : use kernel table in image header (undef)
    [-r rkernel]      : convolution kernel half width (10)
    [-kcs step]       : size of step for spatial convolution (2 * rkernel + 1)
    [-ft fitthresh]   : RMS threshold for good centroid in kernel fit (20.0)
    [-sft scale]      : scale fitthresh by this fraction if... (0.5)
    [-nft fraction]   : this fraction of stamps are not filled (0.1)
    [-vmins spread]    : Fraction of kernel half width to spread input mask (1.0)
    [-mous spread]    : Ditto output mask, negative = no diffim masking (1.0)
    [-omi  fitsfile]  : Output bad pixel mask (undef)
    [-gd xmin xmax ymin ymax]
                         : only use subsection of full image (full image)

    [-nrx xregion]    : number of image regions in x dimension (1)
    [-nry yregion]    : number of image regions in y dimension (1)
       -- OR --
    [-rf regionfile]  : ascii file with image regions 'xmin:xmax,ymin:ymax'
       -- OR --
    [-rkw keyword num]: header 'keyword[0->(num-1)]' indicates valid regions

    [-nsx xstamp]     : number of each region's stamps in x dimension (10)
    [-nsy ystamp]     : number of each region's stamps in y dimension (10)
       -- OR --
    [-ssf stampfile]  : ascii file indicating substamp centers 'x y'
       -- OR --
    [-cmp cmpfile]    : .cmp file indicating substamp centers 'x y'

    [-afssc find]     : autofind stamp centers so #=-nss when -ssf,-cmp (1)
    [-nss substamps]  : number of centroids to use for each stamp (3)
    [-rss radius]     : half width substamp to extract around each centroid (15)

    [-savexy file]    : save positions of stamps for convolution kernel (undef)
    [-c  toconvolve]  : force convolution on (t)emplate or (i)mage (undef)
    [-n  normalize]   : normalize to (t)emplate, (i)mage, or (u)nconvolved (t)
    [-fom figmerit]   : (v)ariance, (s)igma or (h)istogram convolution merit (v)
    [-sconv]          : all regions convolved in same direction (0)
    [-ko kernelorder] : spatial order of kernel variation within region (2)
    [-bgo bgorder]    : spatial order of background variation within region (1)
    [-ssig statsig]   : threshold for sigma clipping statistics  (3.0)
    [-ks badkernelsig]: high sigma rejection for bad stamps in kernel fit (2.0)
    [-kfm kerfracmask]: fraction of abs(kernel) sum for ok pixel (0.990)
    [-okn]            : rescale noise for 'ok' pixels (0)
    [-fi fill]        : value for invalid (bad) pixels (1.0e-30)
    [-fin fill]       : noise image only fillvalue (0.0e+00)
    [-convvar]        : convolve variance not noise (0)

    [-oni fitsfile]   : output noise image (undef)
    [-ond fitsfile]   : output noise scaled difference image (undef)
    [-nim]            : add noise image as layer to sub image (0)
    [-ndm]            : add noise-scaled sub image as layer to sub image (0)

    [-oci fitsfile]   : output convolved image (undef)
    [-cim]            : add convolved image as layer to sub image (0)

    [-allm]           : output all possible image layers

    [-nc]             : do not clobber output image (0)
    [-hki]            : print extensive kernel info to output image header (0)

    [-oki fitsfile]   : new fitsfile with kernel info (under)

    [-sht]            : output images 16 bitpix int, vs -32 bitpix float (0)
    [-obs bscale]     : if -sht, output image BSCALE, overrides -inim (1.0)
    [-obz bzero]      : if -sht, output image BZERO , overrides -inim (0.0)
    [-nsht]           : output noise image 16 bitpix int, vs -32 bitpix float (0)
    [-nbs bscale]     : noise image only BSCALE, overrides -obs (1.0)
    [-nbz bzero]      : noise image only BZERO,  overrides -obz (0.0)

    [-ng  ngauss degree0 sigma0 .. degreeN sigmaN]
                         : ngauss = number of gaussians which compose kernel (3)
                         : degree = degree of polynomial associated with gaussian #
                                    (6 4 2)
                         : sigma  = width of gaussian #
                                    (0.70 1.50 3.00)
                         : N = 0 .. ngauss - 1

                         : (3 6 0.70 4 1.50 2 3.00
    [-pca nk k0.fits ... n(k-1).fits]
                         : nk      = number of input basis functions
                         : k?.fits = name of fitsfile holding basis function
                         : Since this uses input basis functions, it will fix :
                         :    hwKernel
                         :
    [-v] verbosity    : level of verbosity, 0-2 (1)
    '''
    # hotpants is the path is to the binary file
    if os.path.exists(filename) and (not overwrite):
        print(filename + ' already exists. Use a different name of set '
              'overwrite to True if you wish to regenerate a new script.')
    else:
        t_sigma = sigma_ref
        with open(filename, "w+") as out_file:
            for i, aligned_file_path in enumerate(aligned_file_list):
                i_sigma = sigma_list[i]
                if i_sigma < t_sigma:
                    c = 'i'
                    ng = None
                else:
                    sigma_match = np.sqrt(i_sigma**2. - t_sigma**2.)
                    c = None
                    ng = '3 6 ' + str(0.5 * sigma_match) + ' 4 ' + str(
                        sigma_match) + ' 2 ' + str(2. * sigma_match)
                out_string = shell + ' -i -c \''
                out_string += hotpants + ' '
                out_string += '-inim ' + aligned_file_path + ' '
                out_string += '-tmplim ' + ref_path + ' '
                out_string += '-outim ' + aligned_file_path.split('.' + extension)[0] + '_diff.' + extension + ' '
                if tu is not None:
                    out_string += '-tu ' + str(tu) + ' '
                if tuk is not None:
                    out_string += '-tuk ' + str(tuk) + ' '
                if tl is not None:
                    out_string += '-tl ' + str(tl) + ' '
                if tg is not None:
                    out_string += '-tg ' + str(tg) + ' '
                if tr is not None:
                    out_string += '-tr ' + str(tr) + ' '
                if tp is not None:
                    out_string += '-tp ' + str(tp) + ' '
                if tni is not None:
                    out_string += '-tni ' + tni + ' '
                if tmi is not None:
                    out_string += '-tmi ' + tmi + ' '
                if iu is not None:
                    out_string += '-iu ' + str(tu) + ' '
                if iuk is not None:
                    out_string += '-iuk ' + str(tuk) + ' '
                if il is not None:
                    out_string += '-il ' + str(tl) + ' '
                if ig is not None:
                    out_string += '-ig ' + str(ig) + ' '
                if ir is not None:
                    out_string += '-ir ' + str(ir) + ' '
                if ip is not None:
                    out_string += '-ip ' + str(ip) + ' '
                if ini is not None:
                    out_string += '-ini ' + ini + ' '
                if imi is not None:
                    out_string += '-imi ' + imi + ' '
                if ki is not None:
                    out_string += '-ki ' + str(ki) + ' '
                if r is not None:
                    out_string += '-r ' + str(r) + ' '
                if kcs is not None:
                    out_string += '-kcs ' + str(kcs) + ' '
                if ft is not None:
                    out_string += '-ft ' + str(ft) + ' '
                if sft is not None:
                    out_string += '-sft ' + str(sft) + ' '
                if nft is not None:
                    out_string += '-nft ' + str(nft) + ' '
                if vmins is not None:
                    out_string += '-vmins ' + str(vmins) + ' '
                if mous is not None:
                    out_string += '-mous ' + str(mous) + ' '
                if omi is not None:
                    out_string += '-omi ' + omi + ' '
                if gd is not None:
                    out_string += '-gd ' + str(gd) + ' '
                if nrx is not None:
                    out_string += '-nrx ' + str(nrx) + ' '
                if nry is not None:
                    out_string += '-nry ' + str(nry) + ' '
                if rf is not None:
                    out_string += '-rf ' + rf + ' '
                if rkw is not None:
                    out_string += '-rkw ' + rkw + ' '
                if nsx is not None:
                    out_string += '-nsx ' + str(nsx) + ' '
                if nsy is not None:
                    out_string += '-nsy ' + str(nsy) + ' '
                if ssf is not None:
                    out_string += '-ssf ' + str(ssf) + ' '
                if cmpfile is not None:
                    out_string += '-cmp ' + cmpfile + ' '
                if afssc is not None:
                    out_string += '-afssc ' + str(afssc) + ' '
                if nss is not None:
                    out_string += '-nss ' + str(nss) + ' '
                if rss is not None:
                    out_string += '-rss ' + str(rss) + ' '
                if savexy is not None:
                    out_string += '-savexy ' + savexy + ' '
                if c is not None:
                    out_string += '-c ' + c + ' '
                if n is not None:
                    out_string += '-n ' + str(n) + ' '
                if fom is not None:
                    out_string += '-fom ' + str(fom) + ' '
                if sconv is not None:
                    out_string += '-sconv ' + str(sconv) + ' '
                if ko is not None:
                    out_string += '-ko ' + str(ko) + ' '
                if bgo is not None:
                    out_string += '-bgo ' + str(bgo) + ' '
                if ssig is not None:
                    out_string += '-ssig ' + str(ssig) + ' '
                if ks is not None:
                    out_string += '-ks ' + str(ks) + ' '
                if kfm is not None:
                    out_string += '-kfm ' + str(kfm) + ' '
                if okn is not None:
                    out_string += '-okn ' + str(okn) + ' '
                if fi is not None:
                    out_string += '-fi ' + str(fi) + ' '
                if fin is not None:
                    out_string += '-fin ' + str(fin) + ' '
                if convvar is not None:
                    out_string += '-convvar ' + str(convvar) + ' '
                if oni is not None:
                    out_string += '-oni ' + oni + ' '
                if ond is not None:
                    out_string += '-ond ' + ond + ' '
                if nim is not None:
                    out_string += '-nim ' + str(nim) + ' '
                if ndm is not None:
                    out_string += '-ndm ' + str(ndm) + ' '
                if oci is not None:
                    out_string += '-oci ' + oci + ' '
                if cim is not None:
                    out_string += '-cim ' + str(cim) + ' '
                if allm:
                    out_string += '-allm '
                if nc is not None:
                    out_string += '-nc ' + str(nc) + ' '
                if hki is not None:
                    out_string += '-hki ' + str(hki) + ' '
                if oki is not None:
                    out_string += '-oki ' + oki + ' '
                if sht is not None:
                    out_string += '-sht ' + str(sht) + ' '
                if obs is not None:
                    out_string += '-obs ' + str(obs) + ' '
                if obz is not None:
                    out_string += '-obz ' + str(obz) + ' '
                if nsht is not None:
                    out_string += '-nsht ' + str(nsht) + ' '
                if nbs is not None:
                    out_string += '-nbs ' + str(nbs) + ' '
                if nbz is not None:
                    out_string += '-nbz ' + str(nbz) + ' '
                if ng is not None:
                    out_string += '-ng ' + ng + ' '
                if pca is not None:
                    out_string += '-pca ' + pca + ' '
                if v is not None:
                    out_string += '-v ' + str(v) + ' '
                out_string = out_string[:-1] + '\'\n'
                out_file.write(out_string)
        os.chmod(filename, 0o755)


def run_hotpants(filename):
    logfile = open(filename.split('.')[0] + '.log', 'w+')
    subprocess.Popen(filename, stdout=logfile, shell=True)

