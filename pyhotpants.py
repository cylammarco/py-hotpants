import glob
import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import ccdproc
import photutils
from astropy import units as u
from astropy.convolution import convolve_fft as convolve
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import CCDData, NDData
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import extract_stars, IntegratedGaussianPRF, DAOGroup
from scipy.optimize import curve_fit

warnings.simplefilter('ignore', category=AstropyWarning)


def generate_file_list(input_folder, output_folder, filetype='fits'):
    '''
    Genearte a text file containing all the files to be processed. It also
    returns the list.


    Parameters
    ----------
    input_folder: str
        Folder containing the files to be processed.
    output_folder: str
        Folder for the outputs. If the folder does not exist, it will be
        created.
    filetype: str (Default: fits)
        Extension of the files to be processed

    Returns
    -------
    file_list: str
        List of filepaths.
    '''

    # Get the file list from the folder
    file_list = glob.glob(os.path.join(input_folder, "*." + filetype))

    # Sort alpha-numerically
    sorted(file_list,
           key=lambda item: (int(item.partition(' ')[0])
                             if item[0].isdigit() else float('inf'), item))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    np.savetxt(os.path.join(output_folder, 'file_list.txt'),
               file_list,
               fmt='%s')

    return file_list


def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros),
                             axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def correlate2d(im1, im2, boundary='fill', nthreads=1, **kwargs):
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
    shift_right = roll_zeropad(image, 1, 1)
    shift_right[:, 0] = 0
    shift_left = roll_zeropad(image, -1, 1)
    shift_left[:, -1] = 0
    shift_down = roll_zeropad(image, 1, 0)
    shift_down[0, :] = 0
    shift_up = roll_zeropad(image, -1, 0)
    shift_up[-1, :] = 0

    shift_up_right = roll_zeropad(shift_up, 1, 1)
    shift_up_right[:, 0] = 0
    shift_down_left = roll_zeropad(shift_down, -1, 1)
    shift_down_left[:, -1] = 0
    shift_down_right = roll_zeropad(shift_right, 1, 0)
    shift_down_right[0, :] = 0
    shift_up_left = roll_zeropad(shift_left, -1, 0)
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
    >>> im1_aligned_to_im2 = roll_zeropad(roll_zeropad(im1,int(yoff),1),int(xoff),0)
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


def align_images(file_list,
                 output_folder,
                 overwrite=True,
                 ccddata_unit=u.ct,
                 ref=None,
                 method='wcs',
                 target_shape=None,
                 order='bilinear',
                 add_keyword=True,
                 fill_value=0.,
                 ra=None,
                 dec=None,
                 size=None,
                 return_combiner=False,
                 combiner_dtype=None,
                 combine_n_max=20,
                 sigma_clip_low=1,
                 sigma_clip_high=1,
                 sigma_clip_func=np.ma.mean,
                 clip_extrema_low_percentile=15.9,
                 clip_extrema_high_percentile=15.9,
                 save_list=True,
                 list_overwrite=True,
                 list_filename='aligned_file_list'):
    '''
    Parameters
    ----------
    file_list : str
        List of file paths of the images to be aligned.
    output_folder: str
        Path to the folder for the outputs. If the folder does not exist, it
        will be created.
    overwrite: boolean (Default: True)
        Set to true to overwrite files if the file already exists. 
    ccddata_unit: astroypy unit (Default: u.ct)
        Unit of the pixels, it is usually u.ct or u.adu.
    ref: str (Deffault: None)
        File path to the reference image, if None is provided, it uses the
        first image in the file list.
    method: (Default: 'wcs')
        Choose from 'wcs' and 'cc', former reproject the image based on its
        WCS to the reference WC; the latter works by cross-correlating
        between the frame with the reference.
    target_shape: (Default: None)
        Pass to the taget_shape parameter of ccdproc.wcs_project().
    order: (Default: 'bilinear')
        Pass to the order parameter of ccdproc.wcs_project().
    add_keyword: (Default: True)
        Pass to the add_keyword parameter of ccdproc.wcs_project().
    fill_value: (Default: 0.)
        The default NaN, which cannot be read by HOTPANTS, filled by
        wcs_project() will be modified to this value.
    xl : int (Default: 0)
        Number of pixels to be trimmed from the left.
    xr : int (Default: 0)
        Number of pixels to be trimmed from the right.
    yb : int (Default: 0)
        Number of pixels to be trimmed from the bottom.
    yt : int (Default: 0)
        Number of pixels to be trimmed from the top.
    return_combiner: boolean (Default: False)
        Set to Tru to return the combiner.
    combiner_dtype: str or numpy.dtype or None (Default: None)
        Allows user to set dtype. See numpy.array dtype parameter
        description. If None, it uses np.float64.
    combine_n_max: int (Default: 20)
        The maximum number of frames to be combined. This overrides the
        following limites.
    sigma_clip_low: int (Default: 1)
        The number of frames with the lowest signal to be clipped.
    sigma_clip_high: int (Default: 1)
        The number of frames with the highest signal to be clipped.
    sigma_clip_func: callable function (Default: np.ma.mean)
        The function to compute the average.
    clip_extrema_low_percentile: float (Default: 15.9)
        The lower rejection limit by percentage.
    clip_extrema_high_percentile: float (Default: 15.9)
        The upper rejction limit by percentage.

    Return
    ------
    aligned_file_list: list of str
        File list of the aligned images.
    combiner: combiner instance or None
        The combiner containing a subset of the aligned images. The subset
        depends on the sigma clipping and combiner parameters.

    '''

    # open the reference image
    if ref is None:
        ref = file_list[0]
    f_ref = CCDData.read(ref, unit=ccddata_unit)
    image_ref = f_ref.data

    # Get the size
    len_x, len_y = np.shape(image_ref.data)
    if size is None:
        width_x = len_x
        width_y = len_y
    else:
        width_x = size
        width_y = size

    # Get the pixel coordinate of the target in the reference frame
    if (ra is not None) and (dec is not None):
        centre_x, centre_y = f_ref.wcs.all_world2pix(ra,
                                                     dec,
                                                     1,
                                                     tolerance=1e-4,
                                                     maxiter=20,
                                                     adaptive=False,
                                                     detect_divergence=True,
                                                     quiet=False)
    else:
        centre_x = int(len_x / 2)
        centre_y = int(len_y / 2)

    print(
        'The centroid of the target in the reference frame is at pixel ({}, {}).'
        .format(centre_x, centre_y))

    # Get the cutout
    image_ref_trimmed = Cutout2D(image_ref.data,
                                 position=(centre_x, centre_y),
                                 size=(width_x, width_y),
                                 wcs=f_ref.wcs)

    # Get the pixel coordinate of the target in the reference frame AFTER
    # cutout
    if (ra is not None) and (dec is not None):
        centre_x, centre_y = image_ref_trimmed.wcs.all_world2pix(
            ra,
            dec,
            1,
            tolerance=1e-4,
            maxiter=20,
            adaptive=False,
            detect_divergence=True,
            quiet=False)
    else:
        centre_x = int(image_ref_trimmed.size[0] / 2)
        centre_y = int(image_ref_trimmed.size[1] / 2)

    print(
        'The centroid of the target in the reference frame CUTOUT is at pixel ({}, {}).'
        .format(centre_x, centre_y))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    aligned_file_list = []

    if return_combiner:
        combiner_list = []

    for i, f_to_align in enumerate(file_list):
        print('Aligning image ' + str(i + 1) + ' of ' + str(len(file_list)) +
              '.')
        outfile_name = f_to_align.split('.')[0] + '_aligned.fits'
        outfile_path = os.path.join(output_folder, outfile_name.split('/')[-1])

        # Check if file exists and whether to overwrite
        if os.path.exists(outfile_path):
            if not overwrite:
                print('Image already aligned, set overwrite to True to '
                      'realign.')
                aligned_file_list.append(outfile_path)
                continue

        f = CCDData.read(f_to_align, unit=ccddata_unit)

        if method == 'wcs':
            # reproject the image to the WCS of the reference frame
            image_aligned_to_ref = ccdproc.wcs_project(
                f,
                target_wcs=image_ref_trimmed.wcs,
                target_shape=target_shape,
                order=order,
                add_keyword=add_keyword)
            nan_mask = np.isnan(image_aligned_to_ref.data)
            image_aligned_to_ref.data[nan_mask] = fill_value

            # Make the cutout
            image_aligned_to_ref_trimmed = Cutout2D(
                image_aligned_to_ref.data,
                position=(centre_x, centre_y),
                size=(width_x, width_y),
                wcs=image_aligned_to_ref.wcs)

            new_mask = Cutout2D(image_aligned_to_ref.mask,
                                position=(centre_x, centre_y),
                                size=(width_x, width_y),
                                wcs=image_aligned_to_ref.wcs).data

            # Update the output
            image_aligned_to_ref.data = image_aligned_to_ref_trimmed.data
            image_aligned_to_ref.wcs = image_aligned_to_ref_trimmed.wcs
            image_aligned_to_ref.mask = new_mask
            image_aligned_to_ref.header['NAXIS1'] = width_x
            image_aligned_to_ref.header['NAXIS2'] = width_y

        elif method == 'cc':
            header = f.header
            image = f.data

            # 2d cross correlate two frames
            yoff, xoff = cross_correlation_shifts(image, image_ref_trimmed)

            # shift the image to the nearest pixel
            image_aligned_to_ref = CCDData(roll_zeropad(
                roll_zeropad(image, int(yoff), 1), int(xoff),
                0).astype('float32'),
                                           header=header,
                                           unit=ccddata_unit)

            image_aligned_to_ref_trimmed = Cutout2D(
                image_aligned_to_ref.data,
                position=(centre_x, centre_y),
                size=(width_x, width_y),
                wcs=image_aligned_to_ref.wcs)

            new_mask = Cutout2D(image_aligned_to_ref.mask,
                                position=(centre_x, centre_y),
                                size=(width_x, width_y),
                                wcs=image_aligned_to_ref.wcs)

            image_aligned_to_ref.data = image_aligned_to_ref_trimmed.data
            image_aligned_to_ref.wcs = image_aligned_to_ref_trimmed.wcs
            image_aligned_to_ref.mask = new_mask
            image_aligned_to_ref.header['NAXIS1'] = width_x
            image_aligned_to_ref.header['NAXIS2'] = width_y

        # append to the combiner if stacking
        if return_combiner:
            combiner_list.append(image_aligned_to_ref)

        # After alignment is done, append the name to list
        aligned_file_list.append(outfile_path)
        if overwrite:
            image_aligned_to_ref.write(outfile_path, overwrite=overwrite)

    if save_list and (not list_overwrite):
        print(
            os.path.join(output_folder, list_filename) + '.txt already '
            'exists. Use a different name or set overwrite to True. EPSFModel '
            'and EPSFStar are not saved to disk.')
    else:
        np.savetxt(os.path.join(output_folder, list_filename) + '.txt',
                   aligned_file_list,
                   fmt='%s')

    # Return the file list for the aligned images and the combiner
    if return_combiner:

        # Collate the sligned images
        combiner = ccdproc.Combiner(combiner_list, dtype=combiner_dtype)
        # Clip the extrema
        combiner.sigma_clipping(low_thresh=sigma_clip_low,
                                high_thresh=sigma_clip_high,
                                func=sigma_clip_func)
        # Get the central (combine_n_high - combine_n_low) or combine_n_max
        # images
        combiner_size = len(combiner.ccd_list)
        combine_n_low = int(clip_extrema_low_percentile / 100. * combiner_size)
        combine_n_high = int(clip_extrema_high_percentile / 100. *
                             combiner_size)
        n = combine_n_high - combine_n_low
        if combine_n_high - combine_n_low > combine_n_max:
            delta = int((n - combine_n_max) / 2)
            combine_n_high -= delta
            combine_n_low += delta

        combiner.clip_extrema(nlow=combine_n_low, nhigh=combine_n_high)

        return aligned_file_list, combiner, (centre_x, centre_y)

    # Return only the file list for the aligned images
    else:

        return aligned_file_list, None, (centre_x, centre_y)


def get_background(data,
                   box_size=25,
                   sigma=3.,
                   sigma_lower=None,
                   sigma_upper=None,
                   maxiters=5,
                   cenfunc='median',
                   stdfunc='std',
                   bkg_estimator=photutils.MedianBackground(),
                   create_figure=True,
                   save_figure=True,
                   save_bkg=True,
                   overwrite=True,
                   output_folder='.',
                   filename='background',
                   **args):
    '''
    Compute the background of the image.

    Parameters
    ----------
    data: CCDData
        Input data.
    box_size: int (Default: 25)
        Size of the sub-images to compute the background.
    sigma: float (Default: 3.)
        Sigma clipping limit to remove bad pixel values.
    sigma_lower: float (Default: None)
        Lower bound of the sligma clipping.
    sigma_upper: float (Default: NonE)
        Upper bound of the sigma clipping.
    maxiters: int (Default: 5)
        Number of iterations in sigma clipping.
    bkg_estimator: callable function (Default: photutils.MedianBackground)
        Method to be used to compute the background.
    cenfunc: str (Default: 'median')
        Function used for centroiding.
    stdfunc: str (Default: 'std')
        Function used to compute the standard deviation.
    **args:
        Extra arguments for photutils.Background2D()

    Return
    ------
    bkg: Background2D object
        The computed background of the image.

    '''

    # Sigma clipping parameters
    sigma_clip = SigmaClip(sigma=sigma,
                           sigma_lower=sigma_lower,
                           sigma_upper=sigma_upper,
                           maxiters=maxiters,
                           cenfunc=cenfunc,
                           stdfunc=stdfunc)

    # Computing the background
    bkg = photutils.Background2D(data,
                                 box_size,
                                 sigma_clip=sigma_clip,
                                 bkg_estimator=bkg_estimator,
                                 **args)
    if create_figure:
        # Plot
        norm = simple_norm(data, 'log', percent=99.)
        norm_bkg = simple_norm(bkg.background, 'log', percent=99.)
        norm_sub = simple_norm(data - bkg.background, 'log', percent=99.)

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(data, norm=norm, origin='lower')
        ax2.imshow(bkg.background, norm=norm_bkg, origin='lower')
        ax3.imshow(data - bkg.background, norm=norm_sub, origin='lower')
        ax4.set_xticklabels('')
        ax4.set_yticklabels('')

        if save_figure:
            plt.savefig(
                os.path.join(output_folder, 'background_subtraction.png'))

    if save_bkg:
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path) or (not overwrite):
            print(
                output_path + '.npy already '
                'exists. Use a different name or set overwrite to True. Background '
                'is not saved to disk.')
        else:
            np.save(output_path, bkg)

    return bkg


def get_good_stars(data,
                   threshold,
                   box_size=15,
                   footprint=None,
                   mask=None,
                   border_width=None,
                   npeaks=100,
                   centroid_func=None,
                   subpixel=True,
                   error=None,
                   wcs=None,
                   stars_tbl=None,
                   edge_size=50,
                   size=25,
                   output_folder='.',
                   save_stars=True,
                   stars_overwrite=True,
                   stars_filename='good_stars',
                   save_stars_tbl=True,
                   stars_tbl_overwrite=True,
                   stars_tbl_filename='good_stars_tbl',
                   **args):
    '''
    Get the centroids of the bright sources to prepare to compute the FWHM.

    Parameters
    ----------
    data: array_like
        The 2D array of the image.
    threshold: float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D threshold must have the same shape as
        data. See photutils.detection.detect_threshold for one way to
        create a threshold image.
    box_size: scalar or tuple, optional (Default: 15)
        The size of the local region to search for peaks at every point in
        data. If box_size is a scalar, then the region shape will be
        (box_size, box_size). Either box_size or footprint must be defined.
        If they are both defined, then footprint overrides box_size.
    footprint: ndarray of bools, optional (Default: None)
        A boolean array where True values describe the local footprint
        region within which to search for peaks at every point in data.
        box_size=(n, m) is equivalent to footprint=np.ones((n, m)). Either
        box_size or footprint must be defined. If they are both defined,
        then footprint overrides box_size.
    mask: array_like, bool, optional (Default: None)
        A boolean mask with the same shape as data, where a True value
        indicates the corresponding element of data is masked.
    border_width: bool, optional (Default: None)
        The width in pixels to exclude around the border of the data.
    npeaks: int, optional (Default: 100)
        The maximum number of peaks to return. When the number of detected
        peaks exceeds npeaks, the peaks with the highest peak intensities
        will be returned.
    centroid_func: callable, optional (Default: None)
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The centroid_func must accept
        a 2D ndarray, have a mask keyword, and optionally an error keyword.
        The callable object must return a tuple of two 1D ndarrays,
        representing the x and y centroids, respectively.
    subpixel: boolean (Defautl: True)
        Perform sub-pixel fitting when computing the centroids. If False,
        it will return the nearest pixel.
    error: array_like, optional (Default: None)
        The 2D array of the 1-sigma errors of the input data. error is used
        only if centroid_func is input (the error array is passed directly
        to the centroid_func).
    wcs: None or WCS object, optional (Default: None)
        A world coordinate system (WCS) transformation that supports the
        astropy shared interface for WCS (e.g., astropy.wcs.WCS,
        gwcs.wcs.WCS). If None, then the sky coordinates will not be
        returned in the output Table.
    stars_tbl: Table, list of Table, optional (Default: None)
        A catalog or list of catalogs of sources to be extracted from the
        input data. To link stars in multiple images as a single source,
        you must use a single source catalog where the positions defined in
        sky coordinates.

        If a list of catalogs is input (or a single catalog with a single
        NDData object), they are assumed to correspond to the list of NDData
        objects input in data (i.e., a separate source catalog for each 2D
        image). For this case, the center of each source can be defined either
        in pixel coordinates (in x and y columns) or sky coordinates (in a
        skycoord column containing a SkyCoord object). If both are specified,
        then the pixel coordinates will be used.

        If a single source catalog is input with multiple NDData objects, then
        these sources will be extracted from every 2D image in the input data.
        In this case, the sky coordinates for each source must be specified as
        a SkyCoord object contained in a column called skycoord. Each NDData
        object in the input data must also have a valid wcs attribute. Pixel
        coordinates (in x and y columns) will be ignored.

        Optionally, each catalog may also contain an id column representing the
        ID/name of stars. If this column is not present then the extracted stars
        will be given an id number corresponding the the table row number
        (starting at 1). Any other columns present in the input catalogs will be
        ignored.
    edge_size: int (Default: 50)
        The number of pixels from the detector edges to be removed.
    size: int or array_like (int), optional (Default: 25)
        The extraction box size along each axis. If size is a scalar then a
        square box of size size will be used. If size has two elements, they
        should be in (ny, nx) order. The size must be greater than or equal to
        3 pixel for both axes. Size must be odd in both axes; if either is even,
        it is padded by one to force oddness.
    **args:
        Extra arguments for astropy.nddata.NDData which holds the input
        data for photutils.psf.extract_stars().

    Return
    ------
    stars: EPSFStars instance
        A photutils.psf.EPSFStars instance containing the extracted stars.
    stars_tbl: Table, list of Table
        A table containing the x and y pixel location of the peaks and their
        values. If centroid_func is input, then the table will also contain the
        centroid position. If no peaks are found then None is returned.

    '''

    if stars_tbl is None:
        # detect peaks and remove sources near the edge
        peaks_tbl = photutils.find_peaks(data[edge_size:-edge_size,
                                              edge_size:-edge_size],
                                         threshold,
                                         box_size=box_size,
                                         footprint=footprint,
                                         mask=mask,
                                         border_width=border_width,
                                         npeaks=npeaks,
                                         centroid_func=centroid_func,
                                         subpixel=subpixel,
                                         error=error,
                                         wcs=wcs)

        peaks_sort_mask = np.argsort(-peaks_tbl['peak_value'])
        x = peaks_tbl['x_peak'][peaks_sort_mask]
        y = peaks_tbl['y_peak'][peaks_sort_mask]

        stars_tbl = Table()
        stars_tbl['x'] = x + edge_size
        stars_tbl['y'] = y + edge_size

        if save_stars_tbl:
            stars_tbl_output_path = os.path.join(output_folder,
                                                 stars_tbl_filename + '.npy')
            if os.path.exists(stars_tbl_output_path) and (
                    not stars_tbl_overwrite):
                print(
                    stars_tbl_output_path + ' already exists. Use a '
                    'different name or set overwrite to True. EPSFModel is not '
                    'saved to disk.')
            else:
                np.save(stars_tbl_output_path, stars_tbl)

    nddata = NDData(data=data, **args)

    # assign npeaks mask again because if stars_tbl is given, the npeaks
    # have to be selected
    stars = extract_stars(nddata, catalogs=stars_tbl[:npeaks], size=size)

    if save_stars:
        stars_output_path = os.path.join(output_folder,
                                         stars_filename + '.pbl')
        if os.path.exists(stars_output_path) and (not stars_overwrite):
            print(stars_output_path + ' already exists. Use a different '
                  'name or set overwrite to True. EPSFStar is not saved to '
                  'disk.')
        else:
            with open(stars_output_path, 'wb+') as f:
                pickle.dump(stars, f)

    return stars, stars_tbl


def build_psf(stars,
              oversampling=None,
              smoothing_kernel='quartic',
              maxiters=10,
              create_figure=False,
              save_figure=True,
              stamps_nrows=None,
              stamps_ncols=None,
              figsize=(10, 10),
              output_folder='.',
              save_epsf_model=True,
              model_overwrite=True,
              model_filename='epsf_model',
              save_epsf_star=True,
              stars_overwrite=True,
              stars_filename='epsf_star',
              **args):
    '''
    data is best background subtracted
    PSF is built using the 'stars' provided, but the stamps_nrows and
    stamps_ncols are only used for controlling the display

    Parameters
    ----------
    stars: EPSFStars instance
        A photutils.psf.EPSFStars instance containing the extracted stars.
    oversampling: int or tuple of two int, optional(Default: None)
        The oversampling factor(s) of the ePSF relative to the input stars
        along the x and y axes. The oversampling can either be a single float
        or a tuple of two floats of the form (x_oversamp, y_oversamp). If
        oversampling is a scalar then the oversampling will be the same for
        both the x and y axes.
    smoothing_kernel: {'quartic', 'quadratic'}, 2D ndarray, or None
                      (Default: quartic')
        The smoothing kernel to apply to the ePSF. The predefined 'quartic'
        and 'quadratic' kernels are derived from fourth and second degree
        polynomials, respectively. Alternatively, a custom 2D array can be
        input. If None then no smoothing will be performed.
    maxiters: int, optional (Default: 10)
        The maximum number of iterations to perform.
    create_figure: boolean (Default: False)
        Display the cutouts of the regions used for building the PSF.
    stamps_nrows: (Default: None)
        Number of rows to display. This does NOT affect the number of stars
        used for building the PSF.
    stamps_ncols: (Default: None)
        Number of columns to display. This does NOT affect the number of stars
        used for building the PSF.
    figsize: (Default: (10, 10))
        Figure size.
    **args
        Extra arguments for photutils.EPSFBuilder.

    Return
    ------
    epsf: EPSFModel object
        The constructed ePSF.
    fitted_stars: EPSFStars object
        The input stars with updated centers and fluxes derived from fitting
        the output epsf.
    oversampling:
        Return the oversampling factor used. It can be different from the
        input because if the input is too large, the factor will be reduced
        automatically.

    '''

    if oversampling is None:
        # Make sure the oversampling factor is sensible for the number
        # of stars available.
        if smoothing_kernel == 'quartic':
            divisor = 4
        elif smoothing_kernel == 'quadratic':
            divisor = 2
        else:
            divisor = 1
        oversampling = int(np.sqrt(len(stars) % divisor)) - 1
        oversampling = oversampling - oversampling % 2
        if oversampling < 2:
            oversampling = 2

    # Build the PSFs
    epsf_builder = photutils.EPSFBuilder(oversampling=oversampling,
                                         smoothing_kernel=smoothing_kernel,
                                         maxiters=maxiters,
                                         **args)
    try:
        epsf, fitted_stars = epsf_builder(stars)
    except:
        return None, None, None

    if create_figure:
        n_star = len(stars)
        # Get the nearest square number to fill if the number of rows and/or
        # columns is/are not provided.
        if (stamps_nrows is None) and (stamps_ncols is None):
            min_sq_number = int(np.ceil(np.sqrt(n_star)))
            stamps_nrows = min_sq_number
            stamps_ncols = min_sq_number
        elif (stamps_nrows is None):
            stamps_nrows = int(np.ceil(n_star / stamps_ncols))
        elif (stamps_ncols is None):
            stamps_ncols = int(np.ceil(n_star / stamps_nrows))
        else:
            pass
        nrows = stamps_nrows
        ncols = stamps_ncols

        # Set up the figure
        fig, ax = plt.subplots(nrows=stamps_nrows,
                               ncols=stamps_ncols,
                               figsize=figsize,
                               squeeze=True)
        ax = ax.ravel()

        for i in range(int(min_sq_number**2)):
            try:
                norm = simple_norm(stars[i], 'log', percent=99.)
                ax[i].imshow(stars[i],
                             norm=norm,
                             origin='lower',
                             cmap='viridis')
                ax[i].tick_params(axis="x", direction="in", pad=-22)
                ax[i].tick_params(axis="y", direction="in", pad=-15)
            except:
                ax[i].set_xticklabels('')
                ax[i].set_yticklabels('')

        if save_figure:
            plt.savefig(os.path.join(output_folder, 'ePSF_stamps.png'))

        epsf_norm = simple_norm(epsf.data, 'log', percent=99.)

        plt.figure()
        plt.imshow(epsf.data, norm=epsf_norm, origin='lower')
        plt.colorbar()

        if save_figure:
            plt.savefig(os.path.join(output_folder, 'ePSF.png'))

    if save_epsf_model:
        model_output_path = os.path.join(output_folder, model_filename)
        if os.path.exists(model_output_path) and (not model_overwrite):
            print(model_output_path + ' already exists. Use a different name '
                  'or set overwrite to True. EPSFModel is not '
                  'saved to disk.')
        else:
            np.save(model_output_path, epsf)

    if save_epsf_star:
        stars_output_path = os.path.join(output_folder, stars_filename)
        if os.path.exists(stars_output_path) and (not stars_overwrite):
            print(stars_output_path + ' already exists. Use a different name '
                  'or set overwrite to True. EPSFStar is not '
                  'saved to disk.')
        else:
            with open(stars_output_path + '.pbl', 'wb+') as f:
                pickle.dump(fitted_stars, f)

    return epsf, fitted_stars, oversampling


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


def fit_gaussian_for_fwhm(psf, fit_sigma=False):
    '''
    Fit gaussians to the psf.

    Patameters
    ----------
    psf: np.ndarray.array
        The 2D PSF profile
    sigma: boolean
        Set to True to return sigma instead of the FWHM

    Return
    ------
    sigma_x/fwhm_x:
        The PSF_x size in number of pixels.
    sigma_y/fwhm_y:
        The PSF_y size in number of pixels.

    '''

    psf_x = np.sum(psf.data, axis=0)
    psf_y = np.sum(psf.data, axis=1)
    pguess_x = max(psf_x), 0, len(psf_x) / 2., len(psf_x) / 10.
    pguess_y = max(psf_y), 0, len(psf_y) / 2., len(psf_y) / 10.
    bound_x = ([0, -np.inf, len(psf_y) / 4.,
                0], [max(psf_x) * 1.5,
                     np.inf,
                     len(psf_x) * 3. / 4.,
                     len(psf_x) / 2.])
    bound_y = ([0, -np.inf, len(psf_y) / 4.,
                0], [max(psf_y) * 1.5,
                     np.inf,
                     len(psf_y) * 3. / 4.,
                     len(psf_y) / 2.])

    # see also https://photutils.readthedocs.io/en/stable/detection.html
    popt_x, _ = curve_fit(_gaus,
                          np.arange(len(psf_x)),
                          psf_x,
                          bounds=bound_x,
                          p0=pguess_x)
    popt_y, _ = curve_fit(_gaus,
                          np.arange(len(psf_y)),
                          psf_y,
                          bounds=bound_y,
                          p0=pguess_y)
    sigma_x = popt_x[3]
    sigma_y = popt_y[3]

    if fit_sigma:
        print('sigma_x = {} and sigma_y = {}'.format(sigma_x, sigma_y))
        return sigma_x, sigma_y
    else:
        print('fwhm_x = {} and fwhm_y = {}'.format(sigma_x * 2.355, sigma_y * 2.355))
        return sigma_x * 2.355, sigma_y * 2.355


def get_all_fwhm(file_list,
                 stars_tbl,
                 fit_sigma=False,
                 sigma=3.,
                 sigma_lower=None,
                 sigma_upper=None,
                 maxiters=5,
                 bkg_estimator=photutils.MedianBackground(),
                 cenfunc='median',
                 stdfunc='std',
                 threshold=1000.,
                 box_size=3,
                 footprint=None,
                 mask=None,
                 border_width=None,
                 npeaks=None,
                 centroid_func=None,
                 subpixel=False,
                 error=None,
                 wcs=None,
                 size=25,
                 save_fwhm=True,
                 overwrite=True,
                 output_folder='.',
                 filename=None,
                 **args):
    '''
    Compute the FWHM of all the frames using the same stars given by the
    stars_tbl. 

    Parameters
    ----------
    file_list: str
        List of file paths of the images to be aligned.
    stars_tbl: Table, list of Table, optional (Default: None)
        A catalog or list of catalogs of sources to be extracted from the
        input data. To link stars in multiple images as a single source,
        you must use a single source catalog where the positions defined in
        sky coordinates.

        If a list of catalogs is input (or a single catalog with a single
        NDData object), they are assumed to correspond to the list of NDData
        objects input in data (i.e., a separate source catalog for each 2D
        image). For this case, the center of each source can be defined either
        in pixel coordinates (in x and y columns) or sky coordinates (in a
        skycoord column containing a SkyCoord object). If both are specified,
        then the pixel coordinates will be used.

        If a single source catalog is input with multiple NDData objects, then
        these sources will be extracted from every 2D image in the input data.
        In this case, the sky coordinates for each source must be specified as
        a SkyCoord object contained in a column called skycoord. Each NDData
        object in the input data must also have a valid wcs attribute. Pixel
        coordinates (in x and y columns) will be ignored.

        Optionally, each catalog may also contain an id column representing the
        ID/name of stars. If this column is not present then the extracted stars
        will be given an id number corresponding the the table row number
        (starting at 1). Any other columns present in the input catalogs will be
        ignored.
    fit_sigma: (Default: False)
        Default is to return the fitted FWHM of the Gaussian, if set to True,
        the standard deviation of the gitted Gaussian is returned instead.
    sigma: float (Default: 3.)
        Sigma clipping limit to remove bad pixel values.
    sigma_lower: float (Default: None)
        Lower bound of the sligma clipping.
    sigma_upper: float (Default: NonE)
        Upper bound of the sigma clipping.
    maxiters: int (Default: 5)
        Number of iterations in sigma clipping.
    cenfunc: str (Default: 'median')
        Function used for centroiding.
    stdfunc: str (Default: 'std')
        Function used to compute the standard deviation.
    bkg_estimator: callable function (Default: photutils.MedianBackground)
        Method to be used to compute the background.
    threshold: float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D threshold must have the same shape as
        data. See photutils.detection.detect_threshold for one way to
        create a threshold image.
    box_size: scalar or tuple, optional (Default: 15)
        The size of the local region to search for peaks at every point in
        data. If box_size is a scalar, then the region shape will be
        (box_size, box_size). Either box_size or footprint must be defined.
        If they are both defined, then footprint overrides box_size.
    footprint: ndarray of bools, optional (Default: None)
        A boolean array where True values describe the local footprint
        region within which to search for peaks at every point in data.
        box_size=(n, m) is equivalent to footprint=np.ones((n, m)). Either
        box_size or footprint must be defined. If they are both defined,
        then footprint overrides box_size.
    mask: array_like, bool, optional (Default: None)
        A boolean mask with the same shape as data, where a True value
        indicates the corresponding element of data is masked.
    border_width: bool, optional (Default: None)
        The width in pixels to exclude around the border of the data.
    npeaks: int, optional (Default: 100)
        The maximum number of peaks to return. When the number of detected
        peaks exceeds npeaks, the peaks with the highest peak intensities
        will be returned.
    centroid_func: callable, optional (Default: None)
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The centroid_func must accept
        a 2D ndarray, have a mask keyword, and optionally an error keyword.
        The callable object must return a tuple of two 1D ndarrays,
        representing the x and y centroids, respectively.
    subpixel: boolean (Defautl: True)
        Perform sub-pixel fitting when computing the centroids. If False,
        it will return the nearest pixel.
    error: array_like, optional (Default: None)
        The 2D array of the 1-sigma errors of the input data. error is used
        only if centroid_func is input (the error array is passed directly
        to the centroid_func).
    wcs: None or WCS object, optional (Default: None)
        A world coordinate system (WCS) transformation that supports the
        astropy shared interface for WCS (e.g., astropy.wcs.WCS,
        gwcs.wcs.WCS). If None, then the sky coordinates will not be
        returned in the output Table.
    edge_size: int (Default: 50)
        The number of pixels from the detector edges to be removed.
    size: int or array_like (int), optional (Default: 25)
        The extraction box size along each axis. If size is a scalar then a
        square box of size size will be used. If size has two elements, they
        should be in (ny, nx) order. The size must be greater than or equal to
        3 pixel for both axes. Size must be odd in both axes; if either is even,
        it is padded by one to force oddness.
    args:
        Extra arguments for build_psf()

    Return
    ------
    sigma_x/fwhm_x:
        The PSF_x size in number of pixels.
    sigma_y/fwhm_y:
        The PSF_y size in number of pixels.

    '''

    sigma_x = []
    sigma_y = []
    n = len(file_list)

    for i, filepath in enumerate(file_list):
        print('Computing FWHM/sigma for frame ' + str(i + 1) + ' of ' + str(n))
        data_i = fits.open(filepath)[0].data
        # Compute the background
        bkg_i = get_background(data_i,
                               sigma=sigma,
                               sigma_lower=sigma_lower,
                               sigma_upper=sigma_upper,
                               maxiters=maxiters,
                               bkg_estimator=bkg_estimator,
                               cenfunc=cenfunc,
                               stdfunc=stdfunc,
                               create_figure=False,
                               save_bkg=False)
        # Extract the bright point sources
        stars_i, _ = get_good_stars(data_i - bkg_i.background,
                                    threshold=threshold,
                                    box_size=box_size,
                                    footprint=footprint,
                                    mask=mask,
                                    border_width=border_width,
                                    npeaks=npeaks,
                                    centroid_func=centroid_func,
                                    subpixel=subpixel,
                                    error=error,
                                    wcs=wcs,
                                    stars_tbl=stars_tbl,
                                    edge_size=50,
                                    size=size,
                                    save_stars=False,
                                    save_stars_tbl=False)

        # build the psf using the stacked image
        # see also https://photutils.readthedocs.io/en/stable/epsf.html
        epsf_i, _, oversampling_factor = build_psf(stars_i,
                                                   save_epsf_model=False,
                                                   save_epsf_star=False,
                                                   create_figure=False,
                                                   **args)

        # If WCS reprojection failed, the entire image can land outside
        # of the frame, in which case no photometry can be performed,
        # (None, None, None) would have been returned.
        if epsf_i is None:
            sigma_x.append(np.inf)
            sigma_y.append(np.inf)
        else:
            try:
                sigma = fit_gaussian_for_fwhm(epsf_i.data, fit_sigma=fit_sigma)
                sigma_x.append(sigma[0] / oversampling_factor)
                sigma_y.append(sigma[1] / oversampling_factor)
            except:
                sigma_x.append(np.inf)
                sigma_y.append(np.inf)

    sigma_x = np.array(sigma_x)
    sigma_y = np.array(sigma_y)

    if save_fwhm:

        if filename is None:
            if fit_sigma:
                filename = 'sigma.txt'
            else:
                filename = 'fwhm.txt'

        output_path = os.path.join(output_folder, filename)
        if os.path.exists(output_path) and (not overwrite):
            print(output_path + ' already exists. Use a different name or set '
                  'overwrite to True. Photometry is not saved to disk.')
        else:
            np.savetxt(output_path, np.column_stack((sigma_x, sigma_y)))

    return sigma_x, sigma_y


def generate_hotpants_script(ref_path,
                             aligned_file_list,
                             sigma_ref,
                             sigma_list,
                             hotpants='hotpants',
                             extension='fits',
                             write_to_file=True,
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
                             v=0):
    '''
    This function is to genereate the shell script for running HOTPANTS, it
    always returns the list of strings that can be executed in shell.

    Parameters
    ----------
    ref_path: str
        The file path to the reference image.
    aligned_file_list: list of str
        The file list of the aligned images.
    sigma_ref: float
        The standard deviation of the fitted Gaussian of the refernce PSF.
    sigma_list: list of float
        The standard deviation of the fitted Gaussian of the image PSF.
    hotpants: str (Default: 'hotpants')
        The path to the HOTPATNS executable. Default assumes it is exported
        in the shell startup file.
    extension: str (Default: 'fits')
        The extension of the output file.
    write_to_file: boolean (Default: True)
        Save the executable shell script.
    filename: str (Default: 'diff_image.sh')
        File name of the shell script.
    overwrite: boolean (Default: True)
        Set to True to overwrite file if it already exists.
    The parameters for HOTPANTS:
        All the flags follow the same as those used in HOTPANTS, with the
        exception of cmp, which is a function in Python, cmpfile is used
        intead.

        -c and -ng are not available because they are always used based on
        the sigma_ref and sigma_list.

        The following is copied from https://github.com/acbecker/hotpants
        -----------------------------------------------------------------
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

    Return
    ------
    script: list of str
        The list of strings that can be executed in shell.

    '''

    # hotpants is the path is to the binary file
    if os.path.exists(filename) and (not overwrite):
        print(filename + ' already exists. Use a different name of set '
              'overwrite to True if you wish to regenerate a new script.')
    else:
        t_sigma = sigma_ref
        script = []
        for i, aligned_file_path in enumerate(aligned_file_list):
            i_sigma = sigma_list[i]
            if i_sigma < t_sigma:
                c = 'i'
                ng = None
            elif i_sigma == t_sigma:
                c = None
                ng = None
            else:
                sigma_match = np.sqrt(i_sigma**2. - t_sigma**2.)
                c = None
                ng = '3 6 ' + str(0.5 * sigma_match) + ' 4 ' + str(
                    sigma_match) + ' 2 ' + str(2. * sigma_match)
            out_string = hotpants + ' '
            out_string += '-inim ' + aligned_file_path + ' '
            out_string += '-tmplim ' + ref_path + ' '
            out_string += '-outim ' + aligned_file_path.split(
                '.' + extension)[0] + '_diff.' + extension + ' '
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
            out_string = out_string[:-1]
            script.append(out_string)

        if write_to_file:
            with open(filename, "w+") as out_file:
                for i in script:
                    out_file.write(i + '\n')

            os.chmod(filename, 0o755)

        return script


def run_hotpants(script, output_folder='.', overwrite=True, shell=None):
    '''
    Compute the difference images with HOTPANTS by running shell commands.

    Parameters
    ----------
    script: list of str
        The list of strings that can be executed in shell.
    shell: str (Default: zsh)
        The shell to run the script.

    Return
    ------
    diff_image_list: list of str
        File list of the aligned images.

    '''

    if shell is None:
        shell = os.environ['SHELL']

    # Pipe the script to a log file.
    with open(os.path.join(output_folder, 'diff_image.log'), 'w+') as out_file:
        for i in script:
            filepath = i.split(' ')[2].split('.')[0] + '_diff.fits'
            if (not os.path.exists(filepath)) or overwrite:
                process = subprocess.run(i,
                                         stdout=subprocess.PIPE,
                                         shell=True,
                                         executable=shell,
                                         universal_newlines=True)
                out_file.write(process.stdout)
            else:
                continue

    diff_image_list = []
    with open(os.path.join(output_folder, 'diff_file_list.txt'),
              'w+') as out_file:
        for i in script:
            filepath = i.split(' ')[2].split('.')[0] + '_diff.fits'
            out_file.write(filepath + '\n')
            diff_image_list.append(filepath)

    return diff_image_list


def find_star(data,
              sigma_clip=3.0,
              finder='dao',
              fwhm=3.,
              minsep_fwhm=1.,
              roundhi=5.,
              roundlo=-5.,
              sharplo=0.,
              sharphi=2.,
              n_threshold=5.,
              x=None,
              y=None,
              radius=50,
              show=False,
              save_figure=True,
              save_source_list=True,
              overwrite=True,
              output_folder='.',
              filename='source_list',
              **args):
    '''
    Note that the get_good_stars() function only pick the best stars for
    compting the FWHM, this function is for the proper star finding. The
    input data should be sky-subtracted.

    Parameters
    ----------
    data: array_like
        Sky-subtracted image for finding sources.
    sigma_clip: float (Default: 3.0)
        Sigma clipping limit to remove bad pixel values.
    finder: {'dao', 'iraf'} (Default: 'dao')
        Choose the photutils star finder.
    fwhm: float float (Default: 3.0)
        Guess FWHM of the data, in unit of number of pixels.
    minsep_fwhm: float (Default: 1.0)
        Minimum separation between sources.
    roundhi: float (Default: 5.0)
        Maximum roundness.
    roundlo: float (Default: -5.0)
        Minimum roundness.
    sharplo: float (Default: 0.0)
        Minimum sharpness.
    sharphi: float (Default: 2.0)
        Maximum sharpness.
    n_threshold: float (Default: 5.)
        Minimum threshold of a detection.
    x: float or None (Default: None)
        Pixel x-cooredinate of the target. None includes everything detected.
    y: float or None (Default: None)
        Pixel y-cooredinate of the target. None includes everything detected.
    radius: float (Default: 50)
        Number of pixels from (x, y) to be included.
    show: boolean (Default: False)
        Set to true to display the finder chart with annotated source ID.
    args:
        Extra arguments to be passed to the star finder

    Return
    ------
    source_list: Table
        Astropy Table of the source positions.

    '''

    mean, median, std = sigma_clipped_stats(data, sigma=sigma_clip)

    # Initialise the star finder
    if finder == 'dao':
        star_finder = photutils.DAOStarFinder(fwhm=fwhm,
                                              threshold=n_threshold * std,
                                              **args)
    elif finder == 'iraf':
        star_finder = photutils.IRAFStarFinder(fwhm=fwhm,
                                               threshold=n_threshold * std,
                                               **args)
    else:
        raise Error('Unknown finder. Please choose from \'dao\' and \'iraf\'.')

    # Find the stars
    source_list = star_finder(data)

    # Only include stars within the given limit
    if (x is not None) and (y is not None):
        distance = np.sqrt((x - source_list['xcentroid'].data)**2. +
                           (y - source_list['ycentroid'].data)**2.)

        source_list = source_list[distance < radius]

    # Format the list
    for col in source_list.colnames:
        source_list[col].info.format = '%.10g'

    # Display the star finder
    if show:

        positions = np.transpose(
            (source_list['xcentroid'], source_list['ycentroid']))
        apertures = photutils.CircularAperture(positions, r=5.)

        # retain -2sigma to +3sigma
        norm = simple_norm(data,
                           stretch='log',
                           min_percent=2.2,
                           max_percent=99.9)
        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap='binary', origin='lower', norm=norm)
        apertures.plot(color='#0547f9', lw=1.5)

        for source in source_list:
            plt.annotate(str(source['id']),
                         (source['xcentroid'], source['ycentroid']))
        plt.xlim(0, data.shape[1] - 1)
        plt.ylim(0, data.shape[0] - 1)
        plt.xlabel('x / pixel')
        plt.ylabel('y / pixel')
        plt.tight_layout()
        plt.grid(color='greenyellow', ls=':', lw=0.5)

        if save_figure:
            plt.savefig(os.path.join(output_folder, 'star_finder.png'))

    if save_source_list:
        output_path = os.path.join(output_folder, filename)
        if os.path.exists(output_path) and (not overwrite):
            print(output_path + ' already exists. Use a different name '
                  'or set overwrite to True. EPSFModel and EPSFStar are not '
                  'saved to disk.')
        else:
            np.save(output_path, source_list)

    return source_list


def do_photometry(diff_image_list,
                  source_list,
                  sigma_list,
                  bkg_estimator=MMMBackground(),
                  fitter=LevMarLSQFitter(),
                  output_folder='.',
                  save_individual=False,
                  individual_overwrite=True,
                  return_tbl=True,
                  save_tbl=True,
                  tbl_overwrite=True,
                  tbl_filename='photometry_tbl'):
    '''
    Perform forced PSF photometry on the list of differenced images based on
    the positions found from the reference/stacked image, the PSF is modelled
    with the sigmas provided.

    see also https://photutils.readthedocs.io/en/latest/psf.html

    Parameters
    ----------
    diff_image_list: list of str
        File list of the differenced images.
    source_list: Table
        Astropy Table of the souurce positions.
    sigma_list: list of float
        The standard deviation of the fitted Gaussian of the image PSF.
    bkg_estimator: callable, optional (Default: MMMBackground())
        Method to be used to compute the background.
    fitter: callable, optional (Default: LevMarLSQFitter())
        Method to be used to fit for the photometry.

    Return
    ------
    result_tab: Table
        Table of the photometry extracted from all the images.

    '''

    if (not return_tbl) & (not save_individual):
        print(
            'You are not saving photometry of individual frames or returning '
            'the combined photometric table. Please set either or both of '
            'return_tbl and save_individual to True.')
        return None

    result_tab = []
    MJD = []

    # Prepare the Table of position of the centroids
    pos = Table(names=['x_0', 'y_0'],
                data=[source_list['xcentroid'], source_list['ycentroid']])

    n = len(diff_image_list)
    for i, diff_image_path in enumerate(diff_image_list):
        print('Doing photometry on frame ' + str(i + 1) + ' of ' + str(n) +'.')
        print(diff_image_path)
        fitsfile = fits.open(diff_image_path)[0]
        image = fitsfile.data

        MJD = float(fitsfile.header['MJD'])
        sigma_i = sigma_list[i]
        fwhm_i = sigma_i * 2.355
        fit_size = int(((fwhm_i * 3) // 2) * 2 + 1)
        daogroup = DAOGroup(fwhm_i)

        # Set to do forced photometry
        print('Sigma = ' + str(sigma_i) + ' pixels.')
        psf_model = IntegratedGaussianPRF(sigma=sigma_i)
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        photometry = photutils.BasicPSFPhotometry(group_maker=daogroup,
                                                  bkg_estimator=bkg_estimator,
                                                  psf_model=psf_model,
                                                  fitter=fitter,
                                                  fitshape=(fit_size,
                                                            fit_size))
        # Store the photometry
        photometry_i = photometry(image=image, init_guesses=pos)
        photometry_i['mjd'] = np.ones(len(photometry_i)) * MJD
        photometry_i['id'] = source_list['id']

        if return_tbl:
            result_tab.append(photometry_i)

        if save_individual:

            individual_output_path = diff_image_path.split(
                '.')[0] + '_photometry'
            if os.path.exists(individual_output_path) and (
                    not individual_overwrite):
                print(individual_output_path + ' already exists. Use a '
                      'different name or set overwrite to True. Photometry is '
                      'not saved to disk.')
            else:
                np.save(individual_output_path, photometry_i)

    if return_tbl:
        if save_tbl:
            output_path = os.path.join(output_folder, tbl_filename)
            if os.path.exists(output_path) and (not tbl_overwrite):
                print(output_path +
                      ' already exists. Use a different name or set '
                      'overwrite to True. Photometry is not saved to disk.')
            else:
                np.save(output_path, result_tab)

        return result_tab


def get_lightcurve(photometry_list,
                   source_id=None,
                   plot=False,
                   use_flux_fit=False,
                   xlabel='MJD',
                   ylabel='Flux / Count',
                   same_figure=True,
                   save_figure=True,
                   overwrite=True,
                   output_folder='.',
                   filename='lightcurves'):
    '''
    Extract the lightcurves from the result table.

    Parameters
    ----------
    photometry_list: Table
        The result table of the photometry.
    source_id: list of int (Default: None)
        To specify source IDs to extract the light curves for. None returns
        everything in the photometry_list.
    plot: boolean (Default: False)
        Set to True to display the lightcurve plot(s).
    use_flux_fit: boolean (Default: False)
        Set to True to use the best fit *model* flux instead of the integrated
        flux.
    xlabel: str (Default: 'MJD')
        The x label, which is usually in the unit of time/phase.
    ylabel: str (Default: 'Flux / Count')
        The y label, which is usually in the unit of count/ADU/magnitude.
    same_figure: boolean (Default: True)
        Set to True to display all lightcurves in one plot.

    Return
    ------
    source_id: numpy.ndarray.array
        Source ID of the targets.
    mjd: numpy.ndarray.array
        MJD of the images.
    flux: numpy.ndarray.array
        Integrated flux of each target in each image.
    flux_err: numpy.ndarray.array
        Uncertainty of flux of each target in each image.
    flux_fit: numpy.ndarray.array
        Fitted flux of each target in each image.

    '''

    flux = []
    flux_err = []
    flux_fit = []
    mjd = []

    if source_id is None:
        source_id = np.arange(len(photometry_list[0])).astype('int') + 1

    for id_i in source_id:
        n = len(photometry_list)
        flux_i = np.zeros(n)
        flux_err_i = np.zeros(n)
        flux_fit_i = np.zeros(n)
        mjd_i = np.zeros(n)

        for j, list_i in enumerate(photometry_list):
            mask = (list_i['id'] == id_i)
            if mask.any():
                flux_i[j] = list_i[mask]['flux_0'].data[0]
                flux_err_i[j] = list_i[mask]['flux_unc'].data[0]
                flux_fit_i[j] = list_i[mask]['flux_fit'].data[0]
                mjd_i[j] = list_i[mask]['mjd'].data[0]

        flux.append(flux_i)
        flux_fit.append(flux_fit_i)
        flux_err.append(flux_err_i)
        mjd.append(mjd_i)

    mjd = np.array(mjd)
    flux = np.array(flux)
    flux_fit = np.array(flux_fit)
    flux_err = np.array(flux_err)

    if plot:
        if use_flux_fit:
            plot_lightcurve(mjd,
                            flux_fit,
                            flux_err,
                            source_id=source_id,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            output_folder=output_folder,
                            same_figure=same_figure,
                            save_figure=save_figure)
        else:
            plot_lightcurve(mjd,
                            flux,
                            flux_err,
                            source_id=source_id,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            output_folder=output_folder,
                            same_figure=same_figure,
                            save_figure=save_figure)

    if save_figure:
        output_path = os.path.join(output_folder, filename)
        if os.path.exists(output_path) and (not overwrite):
            print(output_path + ' already exists. Use a different name or set '
                  'overwrite to True. Lightcurves are not saved to disk.')
        else:
            np.save(output_path,
                    np.column_stack((mjd, flux, flux_err, flux_fit)))

    return source_id, mjd, flux, flux_err, flux_fit


def ensemble_photometry(flux, flux_err):
    '''
    Perform ensemble photometry with all the sources provided.

    Parameters
    ----------
    flux: List of list
        Flux of each target in each image.
    flux_err: List of list
        Uncertainty in flux of each target in each image.

    Return
    ------
    flux_corr: List of list
        Corrected flux of each target in each image.

    '''

    weight = 1. / flux_err**2.

    # length = number of good sources
    mean_flux = np.zeros(len(flux))
    for i in range(len(flux)):
        mean_flux[i] = np.nansum(flux[i] * weight[i]) / np.nansum(weight[i])

    # length = number of epoch
    correction = np.zeros(len(flux[0]))
    for i in range(len(flux[0])):
        correction[i] = np.nansum(
            (flux[:, i] - mean_flux) * weight[:, i]) / np.nansum(weight[:, i])

    # find the corrected magnitude for all sources
    flux_corr = flux.copy()
    for i in range(len(flux)):
        flux_corr[i] = flux[i] - correction

    return flux_corr


def plot_lightcurve(mjd,
                    flux,
                    flux_err,
                    source_id=None,
                    xlabel='MJD',
                    ylabel='Flux / Count',
                    output_folder='.',
                    same_figure=True,
                    save_figure=True):
    '''
    Plot the lightcurves.

    Parameters
    ----------
    mjd:  List of float
        The x-axis, usually the epoch or phase of the image.
    flux: List of float 
        The flux of the targers.
    flux_err: List of float
        The uncertainty in flux of the targets.
    source_id: List of int (Default: None)
        The source IDs to be plotted.
    xlabel: str (Default: 'MJD')
        The x-label.
    ylabel: str (Default: 'Flux / Count')
        The y-label.
    same_figure: boolean (Default: True)
        If True, plot all lightcurves in the same figure.

    '''

    if isinstance(source_id, int):
        source_id = [source_id]

    if np.shape(flux)[0] == 1:

        order = np.argsort(mjd)[0]
        plt.figure(figsize=(8, 8))

        if source_id is None:
            plt.errorbar(mjd[0][order],
                         flux[0][order],
                         yerr=flux_err[0][order],
                         fmt='o-',
                         markersize=3)
        else:
            plt.errorbar(mjd[0][order],
                         flux[0][order],
                         yerr=flux_err[0][order],
                         fmt='o-',
                         markersize=3,
                         label=str(source_id[0]))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(color='grey', ls=':', lw=0.5)
        plt.tight_layout()

        if source_id is not None:
            plt.legend()

        if save_figure:
            plt.savefig(
                os.path.join(output_folder,
                             'lightcurve_' + str(source_id[0]) + '.png'))

    else:

        if same_figure:

            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()

        for i in range(len(flux)):

            if not same_figure:
                fig = plt.figure(i, figsize=(8, 8))
                ax = fig.gca()
            order = np.argsort(mjd[i])

            if source_id is None:
                ax.errorbar(mjd[i][order],
                            flux[i][order],
                            yerr=(flux_err[i][order], flux_err[i][order]),
                            fmt='o-',
                            markersize=5)
            else:
                ax.errorbar(mjd[i][order],
                            flux[i][order],
                            yerr=(flux_err[i][order], flux_err[i][order]),
                            fmt='o-',
                            markersize=5,
                            label=str(source_id[i]))
                if not same_figure:
                    plt.legend()

            if not same_figure:
                ax.set_xlabel('MJD')
                ax.set_ylabel('Flux / Count')
                ax.grid(color='grey', ls=':', lw=1)
                plt.tight_layout()

                if save_figure:
                    plt.savefig(
                        os.path.join(output_folder,
                                     'lightcurve_' + str(i) + '.png'))

        if same_figure:

            ax.set_xlabel('MJD')
            ax.set_ylabel('Flux / Count')
            ax.grid(color='grey', ls=':', lw=1)
            plt.tight_layout()

            if save_figure:
                plt.savefig(os.path.join(output_folder, 'lightcurve.png'))

            if (source_id is not None) and same_figure:
                plt.legend()
