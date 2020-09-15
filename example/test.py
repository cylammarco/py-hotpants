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
file_list = generate_file_list(input_path, output_path=output_path)

# cross-correlate to align the images (because wcs fit fails regularly in dense field)
# see also https://image-registration.readthedocs.io/en/latest/index.html
# see also https://ccdproc.readthedocs.io/en/latest/image_combination.html
aligned_file_list, combiner = align_images(file_list=file_list,
                                           output_path=output_path,
                                           overwrite=overwrite,
                                           xl=150,
                                           xr=150,
                                           yb=150,
                                           yt=150,
                                           return_combiner=True)
# can also choose median_combine()
data_stacked = combiner.average_combine()

fits.HDUList(fits.PrimaryHDU(np.array(data_stacked))).writeto(os.path.join(
    output_path, 'stacked.fits'),
                                                              overwrite=True)

# background subtraction
# see also https://photutils.readthedocs.io/en/stable/background.html
bkg = get_background(data_stacked,
                     maxiters=10,
                     box_size=(50, 50),
                     filter_size=(5, 5)
                     )
data_stacked_bkg_sub = data_stacked - bkg.background

# Plot
align_ref_img = fits.open(aligned_file_list[0])[0].data
norm = simple_norm(data_stacked_bkg_sub, 'log', percent=99.)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(np.log10(align_ref_img),
           origin='lower',
           vmin=np.nanpercentile(np.log10(align_ref_img), 10),
           vmax=np.nanpercentile(np.log10(align_ref_img), 90))
ax2.imshow(np.log10(data_stacked),
           origin='lower',
           vmin=np.nanpercentile(np.log10(data_stacked), 10),
           vmax=np.nanpercentile(np.log10(data_stacked), 90))
ax3.imshow(np.log10(bkg.background))
ax4.imshow(data_stacked_bkg_sub, norm=norm, origin='lower')

# Get the star stamps to build psf
stars, stars_tbl = get_good_stars(data_stacked_bkg_sub, threshold=500., size=25)

# build the psf using the stacked image
# see also https://photutils.readthedocs.io/en/stable/epsf.html
oversampling_factor = 2.
epsf, fitted_stars = build_psf(stars,
                               oversampling=oversampling_factor,
                               smoothing_kernel='quadratic',
                               maxiters=20,
                               show_stamps=True,
                               stamps_nrows=3,
                               stamps_ncols=4)
epsf_norm = simple_norm(epsf.data, 'log', percent=99.)

plt.figure(5)
plt.imshow(epsf.data, norm=epsf_norm, origin='lower')
plt.colorbar()

# Get the FWHM
# for the stack
sigma_x_stack, sigma_y_stack = fit_gaussian_for_fwhm(epsf.data, fit_sigma=True)
sigma_x_stack /= oversampling_factor
sigma_y_stack /= oversampling_factor
# for each frame
sigma_x, sigma_y = get_all_fwhm(aligned_file_list,
                                stars_tbl,
                                fit_sigma=False,
                                sigma=3.,
                                sigma_lower=3.,
                                sigma_upper=3.,
                                recentering_maxiters=20,
                                smoothing_kernel='quadratic',
                                maxiters=20,
                                oversampling=oversampling_factor,
                                norm_radius=5.5,
                                shift_val=0.01,
                                recentering_boxsize=(5, 5),
                                center_accuracy=0.001,
                                show_stamps=False,
                                stamps_nrows=3,
                                stamps_ncols=4,
                                figsize=(10, 10))

# Use the FWHM to generate script for hotpants
# see also https://github.com/acbecker/hotpants
sigma_ref = np.sqrt(sigma_x_stack**2. + sigma_y_stack**2.)
sigma_list = np.sqrt(sigma_x**2. + sigma_y**2.)
diff_image_script = generate_hotpants_script(os.path.join(output_path, 'stacked.fits'),
                         aligned_file_list,
                         sigma_ref,
                         sigma_list,
                         hotpants='hotpants',
                         write_to_file=True,
                         filename=os.path.join(output_path, 'diff_image.sh'),
                         overwrite=True,
                         tu=50000,
                         tg=2.4,
                         tr=12.,
                         iu=50000,
                         ig=2.4,
                         ir=12.
                         )
# run hotpants
diff_image_list = run_hotpants(diff_image_script)

# Use the FWHM to find the stars in the stacked image
# see also https://photutils.readthedocs.io/en/stable/detection.html
source_list = find_star(data_stacked_bkg_sub, fwhm=sigma_x_stack*2.355, n_threshold=5., show=True)

# Use the psf and stars to perform photometry on the stacked image
# see also https://photutils.readthedocs.io/en/latest/psf.html
photometry_list = do_photometry(diff_image_list, source_list, sigma_list)

# get lightcurves
source_id, mjd, flux, flux_err, flux_fit = get_lightcurve(photometry_list)

# plot all lightcurves in the same figure
#plot_lightcurve(mjd, flux, flux_err)

# plot all lightcurves in the separate figure
#plot_lightcurve(mjd, flux, flux_err, same_figure=False)

# plot 1 lightcurve
plot_lightcurve(mjd[48], flux[48], flux_err[48], source_id=48)

# plot a few lightcurves
good_stars = [40, 41, 43, 44, 46, 47, 48, 49, 50, 52, 55, 60]
mjd_good_stars = [mjd[i] for i in good_stars]
flux_good_stars = [flux[i] for i in good_stars]
flux_err_good_stars = [flux_err[i] for i in good_stars]

plot_lightcurve(mjd_good_stars, flux_good_stars, flux_err_good_stars, source_id=good_stars)


