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
return_combiner = True

# generate file list
file_list = generate_file_list(input_path, output_path=output_path)

# cross-correlate to align the images (because wcs fit fails regularly in dense field)
# see also https://image-registration.readthedocs.io/en/latest/index.html
# see also https://ccdproc.readthedocs.io/en/latest/image_combination.html
aligned_file_list, combiner = align_images(file_list=file_list,
                                           output_path=output_path,
                                           overwrite=overwrite,
                                           return_combiner=return_combiner)
if return_combiner:
    # can also choose median_combine()
    data_stacked = combiner.average_combine()
    fits.HDUList(fits.PrimaryHDU(np.array(data_stacked))).writeto(os.path.join(
        output_path, 'stacked.fits'),
                                                                  overwrite=True)
else:
    data_stacked = fits.open(os.path.join(output_path, 'stacked.fits'))[0].data


# background subtraction
# see also https://photutils.readthedocs.io/en/stable/background.html
bkg = get_background(data_stacked,
                     maxiters=10,
                     box_size=(31, 31),
                     filter_size=(7, 7))
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
ax3.imshow(np.log10(bkg.background), origin='lower')
ax4.imshow(data_stacked_bkg_sub, norm=norm, origin='lower')

# Get the star stamps to build psf
stars, stars_tbl = get_good_stars(data_stacked_bkg_sub,
                                  threshold=100.,
                                  box_size=25,
                                  npeaks=225,
                                  edge_size=150)

# build the psf using the stacked image
# see also https://photutils.readthedocs.io/en/stable/epsf.html
epsf, fitted_stars, oversampling_factor = build_psf(
    stars, smoothing_kernel='quadratic', maxiters=20, show_stamps=True)
epsf_norm = simple_norm(epsf.data, 'log', percent=99.)

plt.figure(5)
plt.imshow(epsf.data, norm=epsf_norm, origin='lower')
plt.colorbar()

# Get the FWHM
# for the stack
sigma_x_stack, sigma_y_stack = fit_gaussian_for_fwhm(epsf.data, fit_sigma=True)
sigma_x_stack /= oversampling_factor
sigma_y_stack /= oversampling_factor

# for each frame, get the sigma (note fit_sigma=Ture, meaning it's returning sigma instead of FWHM)
sigma_x, sigma_y = get_all_fwhm(aligned_file_list,
                                stars_tbl,
                                fit_sigma=True,
                                sigma=3.,
                                sigma_lower=3.,
                                sigma_upper=3.,
                                threshold=5000.,
                                box_size=25,
                                recentering_maxiters=10,
                                smoothing_kernel='quadratic',
                                maxiters=10,
                                norm_radius=5.5,
                                npeaks=20,
                                shift_val=0.01,
                                recentering_boxsize=(5, 5),
                                center_accuracy=0.001,
                                show_stamps=False)

# Use the FWHM to generate script for hotpants
# see also https://github.com/acbecker/hotpants
sigma_ref = np.sqrt(sigma_x_stack**2. + sigma_y_stack**2.)
sigma_list = np.sqrt(sigma_x**2. + sigma_y**2.)
diff_image_script = generate_hotpants_script(
    aligned_file_list[np.argmin(sigma_list)],
    aligned_file_list,
    min(sigma_list),
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
    ir=12.)
# run hotpants
diff_image_list = run_hotpants(diff_image_script)

# Use the FWHM to find the stars in the stacked image
# see also https://photutils.readthedocs.io/en/stable/detection.html
source_list = find_star(data_stacked_bkg_sub,
                        fwhm=sigma_x_stack * 2.355,
                        n_threshold=10.,
                        show=True)

# Use the psf and stars to perform photometry on the stacked image
# see also https://photutils.readthedocs.io/en/latest/psf.html
mask = np.argsort(source_list['peak'].data)
photometry_list = do_photometry(diff_image_list, source_list, sigma_list)

# get lightcurves
source_id, mjd, flux, flux_err, flux_fit = get_lightcurve(
    photometry_list, source_list['id'])

# plot all lightcurves in the same figure
#plot_lightcurve(mjd, flux, flux_err)

# plot all lightcurves in the separate figure
#plot_lightcurve(mjd, flux, flux_err, same_figure=False)

# plot 1 lightcurve
target = 62
plot_lightcurve(mjd[np.where(source_id==target)[0]],
                flux[np.where(source_id==target)[0]],
                flux_err[np.where(source_id==target)[0]],
                source_id=[target])
'''
period = 0.0161558
phase = (mjd[np.where(source_id==target)[0]] / period) % 1
plot_lightcurve(phase,
                flux[np.where(source_id==target)[0]],
                flux_err[np.where(source_id==target)[0]],
                source_id=[target])
'''


# plot a few lightcurves
good_stars = [50, 52, 55, 59, 60, 66, 70, 62]
mjd_good_stars = np.array([mjd[i] for i in good_stars])
flux_good_stars = np.array([flux[i] for i in good_stars])
flux_err_good_stars = np.array([flux_err[i] for i in good_stars])

flux_ensemble = ensemble_photometry(flux_good_stars, flux_err_good_stars)

plot_lightcurve(mjd_good_stars,
                flux_good_stars,
                flux_err_good_stars,
                source_id=good_stars)

plot_lightcurve(mjd_good_stars,
                flux_ensemble,
                flux_err_good_stars,
                source_id=good_stars)
