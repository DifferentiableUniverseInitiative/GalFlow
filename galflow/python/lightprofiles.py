# Functions computing light profiles
# based on https://galsim-developers.github.io/GalSim/_build/html/sb.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from tensorflow.python.types.core import Value

__all__ = ["gaussian", "sersic"]

# The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
fwhm_factor = 2*math.sqrt(2*math.log(2))
# The half-light-radius is sqrt(2 ln2) sigma
hlr_factor = math.sqrt(2*math.log(2))

def gaussian(fwhm=None, half_light_radius=None, sigma=None, flux=None, scale=1., nx=None, ny=None, name=None):
  """Function for generating a Gaussian profile:

    :math:`I(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right) / (2 \pi \sigma^2)`
  
  Assuming same stamp size and pixel scale
  Args:
    fwhm: List of `float`, full-width-half-max of the profile.  Typically given in arcsec.
    half_light_radius: List of `float`, half-light radius of the profile.  Typically given in arcsec.
    sigma: List of `float`, sigma of the profile. Typically given in arcsec.
    flux: List of `float`, flux (in photons/cm^2/s) of the profile. [default: 1]
    scale: `float`, the pixel scale to use for the drawn image
    nx: `int`, width of the stamp
    ny: `int`, height of the stamp

  Returns:
    `Tensor` of shape [batch_size, nx, ny] of the centered profile

  Example:
    >>> gaussian(sigma=[3.], nx=55)
  """

  with tf.name_scope(name or "gaussian"):
    if nx is None:
      if ny is None:
        raise ValueError("Either nx or ny or both must be specified")
      else:
        nx = ny
    else:
      if ny is None:
        ny = nx
    if fwhm is not None:
      if half_light_radius is not None or sigma is not None:
        raise ValueError("Only one of sigma, fwhm, and half_light_radius may be specified,\
          fwhm={}, half_light_radius={}, sigma={}".format(fwhm, half_light_radius, sigma))
      else:
        fwhm = tf.convert_to_tensor(fwhm, dtype=tf.float32)
        sigma = fwhm / fwhm_factor
    elif half_light_radius is not None:
      if sigma is not None:
        raise ValueError("Only one of sigma, fwhm, and half_light_radius may be specified,\
          fwhm={}, half_light_radius={}, sigma={}".format(fwhm, half_light_radius, sigma))
      else:
        half_light_radius = tf.convert_to_tensor(half_light_radius, dtype=tf.float32)
        sigma = half_light_radius / hlr_factor
    elif sigma is None:
      raise ValueError("One of sigma, fwhm, and half_light_radius must be specified,\
          fwhm={}, half_light_radius={}, sigma={}".format(fwhm, half_light_radius, sigma))
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    batch_size = sigma.shape[0]
    if flux is None:
      flux = tf.ones(batch_size)
    else:  
      flux = tf.convert_to_tensor(flux, dtype=tf.float32)

    x, y = tf.cast(tf.meshgrid(tf.range(nx), tf.range(ny)), tf.float32)
    x = tf.repeat(tf.expand_dims(x, 0), batch_size, axis=0)
    y = tf.repeat(tf.expand_dims(y, 0), batch_size, axis=0)

    z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32)) * scale
    
    flux = tf.reshape(flux, (batch_size, 1, 1))
    sigma = tf.reshape(sigma, (batch_size, 1, 1))

    gaussian = flux * tf.exp(-z*z / 2 / sigma/sigma) / 2 / math.pi / sigma / sigma  * scale  * scale

    return gaussian

def sersic(n, half_light_radius=None, scale_radius=None, flux=None, trunc=None,
          flux_untruncated=None, scale=1., nx=None, ny=None, name=None):
  """Function for generating a Sersic profile:

    :math:`I(r) = I0 \exp(-\left(r/r_0\right)^{\frac{1}{n}})`
    where
    :math:`I0 = 1 / (2\pi r_0^2 n \Gamma(2n))`
  
  Assuming same stamp size and pixel scale
  Args:
    n: List of `int`, Sersic index.
    half_light_radius: List of `float`, half-light radius of the profile.  Typically given in arcsec.
    scale_radius: List of `float`, scale radius of the profile.  Typically given in arcsec.
    flux: List of `float`, flux (in photons/cm^2/s) of the profile. [default: 1]
    trunc: List of `float`, an optional truncation radius at which the profile is made to drop to zero, in the same units as the size parameter.
    flux_untruncated: List of `boolean`, specifies whether the ``flux`` and ``half_light_radius`` specifications correspond to the untruncated profile (``True``) or to the truncated profile (``False``, default)
    scale: `float`, the pixel scale to use for the drawn image
    nx: `int`, width of the stamp
    ny: `int`, height of the stamp

  Returns:
    `Tensor` of shape [batch_size, nx, ny] of the centered profile

  Example:
    >>> sersic(n=2, scale_radius=5., flux=40., nx=55)
  """
  with tf.name_scope(name or "sersic"):
    if nx is None:
      if ny is None:
        raise ValueError("Either nx or ny or both must be specified")
      else:
        nx = ny
    else:
      if ny is None:
        ny = nx
    
    if n is None:
      raise ValueError("Sersic index n must be specified")
    else:
      n = tf.convert_to_tensor(n, dtype=tf.float32)
      batch_size = n.shape[0]

    if trunc is not None:
      trunc = tf.convert_to_tensor(trunc, dtype=tf.float32)
      if True in (trunc < 0.):
        raise ValueError("Sersic trunc must be > 0.")
    else:
      trunc = tf.zeros(batch_size)

    if flux_untruncated is None:
      flux_untruncated = tf.cast(tf.zeros(batch_size), tf.bool)
    else:  
      flux_untruncated = tf.convert_to_tensor(flux_untruncated, dtype=tf.bool)

    if half_light_radius is not None:
      if scale_radius is not None:
        raise ValueError("Only one of scale_radius and half_light_radius may be specified,\
          scale_radius={}, half_light_radius={}".format(scale_radius, half_light_radius))
      else:
        half_light_radius = tf.convert_to_tensor(half_light_radius, dtype=tf.float32)
        normalize = tf.cast(tf.math.logical_or(trunc==0., flux_untruncated), tf.float32)
        r0 = half_light_radius * (1-normalize) + normalize * half_light_radius / calculateHLRFactor(n, 0.)
    elif scale_radius is not None:
      scale_radius = tf.convert_to_tensor(scale_radius, dtype=tf.float32)
      r0 = scale_radius
    else:
      raise ValueError("Either scale_radius or half_light_radius must be specified for Sersic,\
        half_light_radius={}, scale_radius={}".format(half_light_radius, scale_radius))

    if flux is None:
      flux = tf.ones(batch_size)
    else:  
      flux = tf.convert_to_tensor(flux, dtype=tf.float32)

    flux_fraction = integratedflux(trunc, r0, n)
    normalize = tf.cast(tf.math.logical_and(trunc > 0., tf.math.logical_not(flux_untruncated)), tf.float32)
    flux = flux * (1-normalize) + flux / flux_fraction * normalize

    x, y = tf.cast(tf.meshgrid(tf.range(nx), tf.range(ny)), tf.float32)
    x = tf.repeat(tf.expand_dims(x, 0), batch_size, axis=0)
    y = tf.repeat(tf.expand_dims(y, 0), batch_size, axis=0)

    z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32)) * scale

    n = tf.reshape(n, (batch_size, 1, 1))
    flux = tf.reshape(flux, (batch_size, 1, 1))
    r0 = tf.reshape(r0, (batch_size, 1, 1))
    trunc = tf.reshape(trunc, (batch_size, 1, 1))

    sersic = tf.exp(-tf.math.pow(z/r0, 1/n))  * scale  * scale
    
    trunc = trunc * tf.cast(trunc>0., tf.float32) + tf.math.sqrt(nx*nx*1.+ny*ny*1.) * scale * tf.cast(trunc==0., tf.float32)
    sersic = tf.cast((z<trunc), tf.float32) * sersic

    sersic /= 2 * math.pi * r0 * r0 * n * tf.math.exp(tf.math.lgamma(2.*n))
    sersic *= flux

    return sersic

def integratedflux(trunc, r0, n):
  """ Convenience function to compute the fraction of the total flux enclosed within a given radius
  """
  r = trunc / r0
  z = tf.math.pow(r, 1./n)
  return tf.math.igamma(2.*n, z) * tf.cast(trunc>0., tf.float32) + tf.ones(n.shape[0]) *  tf.cast(trunc==0., tf.float32)

def calculate_b(n):
  """
  Args:
    n: List of `float`

  Return:
    b_n

  Find the solution to gamma(2n; b_n) = Gamma(2n)/2 
  Start with approximation from Ciotti & Bertin, 1999:
  b ~= 2n - 1/3 + 4/(405n) + 46/(25515n^2) + 131/(1148175n^3) - ...
  Then, use a Broyden solver
  """
  n = tf.convert_to_tensor(n, dtype=tf.float32)
  b1 = 2.*n-1./3.
  b2 = b1 + (8./405.)*(1./n) + (46./25515.)*(1./n/n) + (131./1148175.)*(1./n/n/n)

  def func(b):
    return 2.*tf.math.igamma(2*n, b) - 1.

  def update(z, zprev, not_done):
    fp_z = (func(z) - func(zprev)) / (z - zprev)
    return z - func(z)/fp_z * tf.cast(not_done, tf.float32)
  
  def Broyden_solver(b2, b1):
    z_prev, z = b2, update(b2, b1, tf.cast(tf.ones(b2.shape[0]), tf.bool))
    while not tf.reduce_all(func(z) < 1e-5):
      not_done = func(z) > 1e-5
      z_prev, z = z, update(z, z_prev, not_done)
    return z
  
  return Broyden_solver(b2,b1)

def calculateHLRFactor(n, flux_fraction=0.):
  """Calculate the half-light-radius in units of the scale radius.
  """
  b = calculate_b(n)
  return tf.math.pow(b,n)