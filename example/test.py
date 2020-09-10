import sys
sys.path.append('..')

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from pyhotpants import *

plt.ion()

input_path = 'test_data'
output_path = 'test_output'

# parameters
overwrite = True

# generate file list
file_list = genrate_file_list(input_path, output_path=output_path)
align_ref = file_list[0]


align_ref_img = fits.open(file_list[0])[0].data
plt.figure(1)
plt.imshow(np.log10(align_ref_img),
           origin='lower',
           vmin=np.nanpercentile(np.log10(align_ref_img), 10),
           vmax=np.nanpercentile(np.log10(align_ref_img), 90))
plt.colorbar()

# cross-correlate to align the images (because wcs fit fails regularly in dense field)
# see also https://image-registration.readthedocs.io/en/latest/index.html
# see also https://ccdproc.readthedocs.io/en/latest/image_combination.html
aligned_file_list, combiner = align_images(align_ref,
                                           file_list=file_list,
                                           output_path=output_path,
                                           overwrite=overwrite,
                                           xl=100,
                                           xr=100,
                                           yb=100,
                                           yt=100,
                                           stack=True)
# can also choose median_combine()
data_stacked = combiner.average_combine()

plt.figure(2)
plt.imshow(np.log10(data_stacked),
           origin='lower',
           vmin=np.nanpercentile(np.log10(data_stacked), 10),
           vmax=np.nanpercentile(np.log10(data_stacked), 90))
plt.colorbar()

# background subtraction
# see also https://photutils.readthedocs.io/en/stable/background.html
bkg = get_background(data_stacked, box_size=(50, 50), filter_size=(3, 3), sigma=3.)
data_stacked_bkg_sub = data_stacked - bkg.background

plt.figure(3)
plt.imshow(np.log10(data_stacked_bkg_sub),
           origin='lower')
plt.colorbar()

# build the psf using the stacked image
# see also https://photutils.readthedocs.io/en/stable/epsf.html
epsf, fitted_stars = build_psf(data_stacked_bkg_sub, threshold=200., show_stamps=True, figsize=(10, 10))
norm = simple_norm(epsf.data, 'log', percent=99.)

plt.figure(4)
plt.imshow(epsf.data, norm=norm, origin='lower')
plt.colorbar()


# Get the FWHM
# see also https://photutils.readthedocs.io/en/latest/psf.html


# Use the FWHM to find stars
# see also https://photutils.readthedocs.io/en/stable/detection.html


# Use the psf and stars to perform photometry on the stacked image
# see also https://photutils.readthedocs.io/en/latest/psf.html

# Use the FWHM to generate script for hotpants
# see also https://github.com/acbecker/hotpants

# run hotpants


# get residual flux from hotpants


# match the Photutils and the difference image photometry from hotpants


# plot lightcurves





