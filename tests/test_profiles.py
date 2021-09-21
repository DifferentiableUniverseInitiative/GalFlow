import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose

import galflow as gf
import galsim

# Some parameters used for testing Gaussian light profile generation
scale = 0.03      # arcsec
stamp_size = 33   # pixel
sigma = 3.        # arcsec
hlr = 3.          # arcsec
fwhm = 3.         # arcsec
flux = 40.        # photons/cm^2/s

def test_gaussian_profile():
  """This test generates a simple Gaussian light profile with Galsim and GalFlow,
  then checks that the same image stamp is recovered
  """  

  # check sigma input
  obj = galsim.Gaussian(sigma=sigma)
  image_galsim_sigma = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_sigma = gf.lightprofiles.gaussian(sigma=[sigma], nx=stamp_size, ny=stamp_size)[0,...]

  # check batch input
  obj1 = galsim.Gaussian(sigma=sigma)
  obj2 = galsim.Gaussian(sigma=sigma*2)
  image_galsim_batch1 = obj1.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galsim_batch2 = obj2.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galsim_batch = np.stack([image_galsim_batch1, image_galsim_batch2], axis=0)
  image_galflow_batch = gf.lightprofiles.gaussian(sigma=[sigma, sigma*2], nx=stamp_size, ny=stamp_size)

  # check half_light_radius input
  obj = galsim.Gaussian(half_light_radius=hlr)
  image_galsim_hlr = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_hlr = gf.lightprofiles.gaussian(half_light_radius=[hlr], nx=stamp_size, ny=stamp_size)[0,...]

  # check fwhm input
  obj = galsim.Gaussian(fwhm=fwhm)
  image_galsim_fwhm = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_fwhm = gf.lightprofiles.gaussian(fwhm=[fwhm], nx=stamp_size, ny=stamp_size)[0,...]

  # check fwhm input
  obj = galsim.Gaussian(fwhm=fwhm)
  image_galsim_scale = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale, method='no_pixel').array
  image_galflow_scale = gf.lightprofiles.gaussian(fwhm=[fwhm], nx=stamp_size, ny=stamp_size, scale=scale)[0,...]

  # check flux input
  obj = galsim.Gaussian(fwhm=fwhm, flux=flux)
  image_galsim_flux = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_flux = gf.lightprofiles.gaussian(fwhm=[fwhm], flux=[flux], nx=stamp_size, ny=stamp_size)[0,...]

  # check even and odd stamp sizes
  obj = galsim.Gaussian(fwhm=fwhm, flux=flux)
  image_galsim_size = obj.drawImage(nx=stamp_size, ny=stamp_size+1, scale=1., method='no_pixel').array
  image_galflow_size = gf.lightprofiles.gaussian(fwhm=[fwhm], flux=[flux], nx=stamp_size, ny=stamp_size+1)[0,...]

  assert_allclose(image_galsim_sigma, image_galflow_sigma, atol=1e-5)
  assert_allclose(image_galsim_batch, image_galflow_batch, atol=1e-5)
  assert_allclose(image_galsim_hlr, image_galflow_hlr, atol=1e-5)
  assert_allclose(image_galsim_fwhm, image_galflow_fwhm, atol=1e-5)
  assert_allclose(image_galsim_scale, image_galflow_scale, rtol=1e-5)
  assert_allclose(image_galsim_flux, image_galflow_flux, atol=1e-5)
  assert_allclose(image_galsim_size, image_galflow_size, atol=1e-5)

# Some parameters used for testing Sersic light profile generation
scale = 0.03      # arcsec
stamp_size = 55   # pixel
scale_radius = 5  # arcsec
hlr = 30.         # arcsec
n = 2             
flux = 1.         # photons/cm^2/s
trunc = 10.       # arcsec

def test_calculate_b():
  """Test the computation of b_n for Sersic index n on the first 10 integers
  """
  b_m = [1.67834699, 3.67206075, 5.67016119, 7.66924944, 9.66871461, 11.6683632,
         13.6681146,  15.6679295,  17.6677864, 19.6676724]
  m = tf.cast(tf.range(1, 11), tf.float32)
  assert_allclose(gf.lightprofiles.calculate_b(m), b_m, atol=1e-4)

def test_sersic_profile():
  """This test generates a simple Sersic light profile with Galsim and GalFlow,
  then checks that the same image stamp is recovered
  """  

  # check scale_radius input
  obj = galsim.Sersic(n=n, scale_radius=scale_radius)
  image_galsim_scale_radius = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_scale_radius = gf.lightprofiles.sersic(n=[n], scale_radius=[scale_radius], nx=stamp_size)[0,...]

  # check batch input
  obj1 = galsim.Sersic(n=n, scale_radius=scale_radius)
  obj2 = galsim.Sersic(n=n*2, scale_radius=scale_radius*2)
  image_galsim_batch1 = obj1.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galsim_batch2 = obj2.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galsim_batch = np.stack([image_galsim_batch1, image_galsim_batch2], axis=0)
  image_galflow_batch = gf.lightprofiles.sersic(n=[n, n*2.], scale_radius=[scale_radius, scale_radius*2.], nx=stamp_size)


  # check half_light_radius input
  obj = galsim.Sersic(n=n, half_light_radius=hlr, flux=flux)
  image_galsim_hlr = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_hlr = gf.lightprofiles.sersic(n=[n], half_light_radius=[hlr], nx=stamp_size, flux=flux)[0,...]

  # check scale input
  obj = galsim.Sersic(n=n, half_light_radius=hlr, flux=flux)
  image_galsim_scale = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=scale, method='no_pixel').array
  image_galflow_scale = gf.lightprofiles.sersic(n=[n], half_light_radius=[hlr], nx=stamp_size, flux=[flux], scale=scale)[0,...]

  # check even and odd stamp sizes
  obj = galsim.Sersic(n=n, scale_radius=scale_radius)
  image_galsim_size = obj.drawImage(nx=stamp_size, ny=stamp_size+1, scale=1., method='no_pixel').array
  image_galflow_size = gf.lightprofiles.sersic(n=[n], scale_radius=[scale_radius], nx=stamp_size, ny=stamp_size+1)[0,...]

  # check truncated profile, flux_untruncated=Flase
  obj = galsim.Sersic(n=n, scale_radius=scale_radius, trunc=trunc, flux_untruncated=False)
  image_galsim_truncf = obj.drawImage(nx=stamp_size, ny=stamp_size+1, scale=1., method='no_pixel').array
  image_galflow_truncf = gf.lightprofiles.sersic(n=[n], scale_radius=[scale_radius], nx=stamp_size, ny=stamp_size+1, trunc=[trunc], flux_untruncated=[False])[0,...]
  
  # check truncated profile, flux_untruncated=True
  obj = galsim.Sersic(n=n, scale_radius=scale_radius, trunc=trunc, flux_untruncated=True)
  image_galsim_trunct = obj.drawImage(nx=stamp_size, ny=stamp_size+1, scale=1., method='no_pixel').array
  image_galflow_trunct = gf.lightprofiles.sersic(n=[n], scale_radius=[scale_radius], nx=stamp_size, ny=stamp_size+1, trunc=[trunc], flux_untruncated=[True])[0,...]

  assert_allclose(image_galsim_scale_radius, image_galflow_scale_radius, rtol=1e-5)
  assert_allclose(image_galsim_batch, image_galflow_batch, atol=1e-5)
  assert_allclose(image_galsim_hlr, image_galflow_hlr, rtol=1e-5)
  assert_allclose(image_galsim_scale, image_galflow_scale, rtol=1e-5)
  assert_allclose(image_galsim_size, image_galflow_size, rtol=1e-5)
  assert_allclose(image_galsim_truncf, image_galflow_truncf, rtol=1e-5)
  assert_allclose(image_galsim_trunct, image_galflow_trunct, rtol=1e-5)