import sys
sys.path.append('..')

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from matplotlib import pyplot as plt

from pyhotpants import *

plt.ion()

# parameters
input_folder = 'test_data'
output_folder = 'test_output'

ra = 253.50009268021
dec = 62.89923147256

overwrite = True
return_combiner = True

# generate file list
file_list = generate_file_list(input_folder, output_folder=output_folder)

# cross-correlate to align the images (because wcs fit fails regularly in dense field)
# see also https://image-registration.readthedocs.io/en/latest/index.html
# see also https://ccdproc.readthedocs.io/en/latest/image_combination.html
aligned_file_list, combiner = align_images(file_list=file_list,
                                           output_folder=output_folder,
                                           overwrite=overwrite,
                                           xl=200,
                                           xr=200,
                                           yb=200,
                                           yt=200,
                                           return_combiner=return_combiner)
if return_combiner:
    # can also choose median_combine()
    data_stacked = combiner.average_combine()
    fits.HDUList(fits.PrimaryHDU(np.array(data_stacked))).writeto(
        os.path.join(output_folder, 'stacked.fits'), overwrite=True)
else:
    data_stacked = fits.open(os.path.join(output_folder,
                                          'stacked.fits'))[0].data

# Get the pixel coordinate of the target in the aligned frame
f_ref = CCDData.read(aligned_file_list[0], unit=u.ct)
x, y = f_ref.wcs.all_world2pix(ra,
                               dec,
                               1,
                               tolerance=1e-4,
                               maxiter=20,
                               adaptive=False,
                               detect_divergence=True,
                               quiet=False)
print('The centroid of the target is at pixel ({}, {}).'.format(x, y))

# background subtraction
# see also https://photutils.readthedocs.io/en/stable/background.html
bkg = get_background(data_stacked,
                     maxiters=10,
                     box_size=(31, 31),
                     filter_size=(7, 7),
                     create_figure=True,
                     output_folder=output_folder)
data_stacked_bkg_sub = data_stacked - bkg.background

# Get the star stamps to build psf
stars, stars_tbl = get_good_stars(data_stacked_bkg_sub,
                                  threshold=100.,
                                  box_size=25,
                                  npeaks=225,
                                  edge_size=150,
                                  output_folder=output_folder)

# build the psf using the stacked image
# see also https://photutils.readthedocs.io/en/stable/epsf.html
epsf, fitted_stars, oversampling_factor = build_psf(
    stars,
    smoothing_kernel='quadratic',
    maxiters=20,
    create_figure=True,
    save_figure=True,
    output_folder=output_folder)

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
                                maxiters=10,
                                norm_radius=5.5,
                                npeaks=20,
                                shift_val=0.01,
                                recentering_boxsize=(5, 5),
                                recentering_maxiters=10,
                                center_accuracy=0.001,
                                smoothing_kernel='quadratic',
                                output_folder=output_folder)

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
    filename=os.path.join(output_folder, 'diff_image.sh'),
    overwrite=True,
    tu=50000,
    tg=2.4,
    tr=12.,
    iu=50000,
    ig=2.4,
    ir=12.)
# run hotpants
diff_image_list = run_hotpants(diff_image_script, output_folder=output_folder)

# Use the FWHM to find the stars in the stacked image
# see also https://photutils.readthedocs.io/en/stable/detection.html
source_list = find_star(data_stacked_bkg_sub,
                        fwhm=sigma_x_stack * 2.355,
                        n_threshold=10.,
                        x=x,
                        y=y,
                        radius=300,
                        show=True,
                        output_folder=output_folder)

# Use the psf and stars to perform forced photometry on the differenced images
# see also https://photutils.readthedocs.io/en/latest/psf.html
photometry_list = do_photometry(diff_image_list,
                                source_list,
                                sigma_list,
                                output_folder=output_folder,
                                save_individual=True)

# get lightcurves
source_id, mjd, flux, flux_err, flux_fit = get_lightcurve(
    photometry_list, source_list['id'], plot=True, output_folder=output_folder)

# plot all lightcurves in the same figure
#plot_lightcurve(mjd, flux, flux_err)

# plot all lightcurves in the separate figure
#plot_lightcurve(mjd, flux, flux_err, same_figure=False)

# Explicitly plot 1 lightcurve
target = 3
plot_lightcurve(mjd[np.where(source_id == target)[0]],
                flux[np.where(source_id == target)[0]],
                flux_err[np.where(source_id == target)[0]],
                source_id=target,
                output_folder=output_folder)

# Explicitly plot a few lightcurves
good_stars = [1, 5, 8, 10, 3]
mjd_good_stars = np.array([mjd[i] for i in good_stars])
flux_good_stars = np.array([flux[i] for i in good_stars])
flux_err_good_stars = np.array([flux_err[i] for i in good_stars])

flux_ensemble = ensemble_photometry(flux_good_stars, flux_err_good_stars)

plot_lightcurve(mjd_good_stars,
                flux_good_stars,
                flux_err_good_stars,
                source_id=good_stars,
                output_folder=output_folder)

plot_lightcurve(mjd_good_stars,
                flux_ensemble,
                flux_err_good_stars,
                source_id=good_stars,
                output_folder=output_folder)
