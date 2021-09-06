import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose

import galflow as gf
import galsim

# Some parameters used for testing Gaussian light profile generation
stamp_size = 33   # pixel
sigma = 3.        # arcsec
hlr = 3.          # arcsec
fwhm = 3.         # arcsec
flux = 40.        # photons/cm^2/s

def test_gaussian_profile():
  """
  This test generates a simple Gaussian light profile with Galsim and GalFlow,
  then checks that the same image stamp is recovered
  """  

  # check sigma input
  obj = galsim.Gaussian(sigma=sigma)
  image_galsim_sigma = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_sigma = gf.lightprofiles.gaussian(sigma=sigma, stamp_size=stamp_size)

  # check half_light_radius input
  obj = galsim.Gaussian(half_light_radius=hlr)
  image_galsim_hlr = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_hlr = gf.lightprofiles.gaussian(half_light_radius=hlr, stamp_size=stamp_size)

  # check fwhm input
  obj = galsim.Gaussian(fwhm=fwhm)
  image_galsim_fwhm = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_fwhm = gf.lightprofiles.gaussian(fwhm=fwhm, stamp_size=stamp_size)

  # check flux input
  obj = galsim.Gaussian(fwhm=fwhm, flux=flux)
  image_galsim_flux = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_flux = gf.lightprofiles.gaussian(fwhm=fwhm, flux=flux, stamp_size=stamp_size)

  assert_allclose(image_galsim_sigma, image_galflow_sigma, atol=1e-5)
  assert_allclose(image_galsim_hlr, image_galflow_hlr, atol=1e-5)
  assert_allclose(image_galsim_fwhm, image_galflow_fwhm, atol=1e-5)
  assert_allclose(image_galsim_flux, image_galflow_flux, atol=1e-5)

# Some parameters used for testing Sersic light profile generation
stamp_size = 55   # pixel
scale_radius = 5  # arcsec
n = 2             
flux = 40         # photons/cm^2/s


def test_sersic_profile():
  """
  This test generates a simple Gaussian light profile with Galsim and GalFlow,
  then checks that the same image stamp is recovered
  """  

  # check scale_radius input
  obj = galsim.Sersic(n=n, scale_radius=scale_radius)
  image_galsim_scale_radius = obj.drawImage(nx=stamp_size, ny=stamp_size, scale=1., method='no_pixel').array
  image_galflow_scale_radius = gf.lightprofiles.sersic(n=n, scale_radius=scale_radius, stamp_size=stamp_size)

  assert_allclose(image_galsim_scale_radius, image_galflow_scale_radius, rtol=1e-4)
