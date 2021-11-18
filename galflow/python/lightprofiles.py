# Functions computing light profiles
# based on https://galsim-developers.github.io/GalSim/_build/html/sb.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
from tensorflow.python.ops.gen_array_ops import ones_like

from tensorflow.python.types.core import Value

__all__ = ["gaussian", "sersic", "deVaucouleurs"]

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

    x, y = tf.meshgrid(tf.range(nx), tf.range(ny))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    x = tf.repeat(tf.expand_dims(x, 0), batch_size, axis=0)
    y = tf.repeat(tf.expand_dims(y, 0), batch_size, axis=0)

    z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32)) * scale
    
    flux = tf.reshape(flux, (batch_size, 1, 1))
    sigma = tf.reshape(sigma, (batch_size, 1, 1))

    gaussian = flux * tf.exp(-z*z / 2 / sigma/sigma) / 2 / math.pi / sigma / sigma  * scale  * scale

    return gaussian

def exponential(half_light_radius=None, scale_radius=None, flux=None, 
                scale=1., nx=None, ny=None, name=None):
  """Function for generating an Exponential profile:

    :math:`I(r) \sim e^{-r/r_0}`
  
  Assuming same stamp size and pixel scale
  Args:
    half_light_radius: List of `float`, half-light radius of the profile.  Typically given in arcsec.
    scale_radius: List of `float`, scale radius of the profile.  Typically given in arcsec.
    flux: List of `float`, flux (in photons/cm^2/s) of the profile. [default: 1]
    scale: `float`, the pixel scale to use for the drawn image
    nx: `int`, width of the stamp
    ny: `int`, height of the stamp

  Returns:
    `Tensor` of shape [batch_size, nx, ny] of the centered profile

  Example:
    >>> sersic(n=[2.], scale_radius=[5.], flux=[40.], nx=55)
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
    
    _hlr_factor = 1.6783469900166605

    if half_light_radius is not None:
      half_light_radius = tf.convert_to_tensor(half_light_radius, dtype=tf.float32)
      batch_size = half_light_radius.shape[0]
      if scale_radius is not None:
        raise ValueError("Only one of scale_radius and half_light_radius may be specified,\
          scale_radius={}, half_light_radius={}".format(scale_radius, half_light_radius))
      else:
        r0 = half_light_radius / _hlr_factor
    elif scale_radius is not None:
      scale_radius = tf.convert_to_tensor(scale_radius, dtype=tf.float32)
      batch_size = scale_radius.shape[0]
      r0 = scale_radius
    else:
      raise ValueError("Either scale_radius or half_light_radius must be specified for Exponential,\
        half_light_radius={}, scale_radius={}".format(half_light_radius, scale_radius))

    if flux is None:
      flux = tf.ones(batch_size)
    else:  
      flux = tf.convert_to_tensor(flux, dtype=tf.float32)

    x, y = tf.meshgrid(tf.range(nx), tf.range(ny))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    x = tf.repeat(tf.expand_dims(x, 0), batch_size, axis=0)
    y = tf.repeat(tf.expand_dims(y, 0), batch_size, axis=0)

    z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32)) * scale

    flux = tf.reshape(flux, (batch_size, 1, 1))
    r0 = tf.reshape(r0, (batch_size, 1, 1))

    exponential = tf.exp(-tf.math.abs(z/r0))  * scale  * scale

    exponential /=  2 * math.pi * r0 * r0
    
    exponential *= flux
    
    return exponential

def sersic(n, half_light_radius=None, scale_radius=None, flux=None, trunc=None,
          flux_untruncated=None, scale=1., nx=None, ny=None, name=None):
  """Function for generating a Sersic profile:

    :math:`I(r) = I0 \exp(-\left(r/r_0\right)^{\frac{1}{n}})`
    where
    :math:`I0 = 1 / (2\pi r_0^2 n \Gamma(2n))`
  
  Assuming same stamp size and pixel scale
  Args:
    n: List of `float`, Sersic index.
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
    >>> sersic(n=[2.], scale_radius=[5.], flux=[40.], nx=55)
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

    flux = sersic_flux_normalization(flux, trunc, r0, n, flux_untruncated)

    x, y = tf.meshgrid(tf.range(nx), tf.range(ny))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

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

    sersic = sersic_normalization(sersic, n, r0)
    
    sersic *= flux
    
    return sersic

def deVaucouleurs(half_light_radius=None, scale_radius=None, flux=None, trunc=None,
          flux_untruncated=None, scale=1., nx=None, ny=None, name=None):
  """Function for generating a DeVaucoueurs profile:

    :math:`I(r) \sim e^{-(r/r_0)^{1/4}}`

  This is completely equivalent to a Sersic with n=4.
  
  Assuming same stamp size and pixel scale
  Args:
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
    >>> deVaucouleurs(scale_radius=[5.], flux=[40.], nx=55)
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

    if half_light_radius is not None:
      batch_size = len(half_light_radius)
    elif scale_radius is not None:
      batch_size = len(scale_radius)
    else:
      raise ValueError("Either scale_radius or half_light_radius must be specified for Sersic,\
        half_light_radius={}, scale_radius={}".format(half_light_radius, scale_radius))

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

    _b4 = 7.66924944
    _hlr4 = tf.math.pow(_b4, 4.)

    if half_light_radius is not None:
      if scale_radius is not None:
        raise ValueError("Only one of scale_radius and half_light_radius may be specified,\
          scale_radius={}, half_light_radius={}".format(scale_radius, half_light_radius))
      else:
        half_light_radius = tf.convert_to_tensor(half_light_radius, dtype=tf.float32)
        normalize = tf.cast(tf.math.logical_or(trunc==0., flux_untruncated), tf.float32)
        r0 = half_light_radius * (1-normalize) + normalize * half_light_radius / _hlr4

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

    flux = sersic_flux_normalization(flux, trunc, r0, 4., flux_untruncated)

    x, y = tf.meshgrid(tf.range(nx), tf.range(ny))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    x = tf.repeat(tf.expand_dims(x, 0), batch_size, axis=0)
    y = tf.repeat(tf.expand_dims(y, 0), batch_size, axis=0)

    z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32)) * scale

    
    n = 4. * tf.ones((batch_size, 1, 1))
    flux = tf.reshape(flux, (batch_size, 1, 1))
    r0 = tf.reshape(r0, (batch_size, 1, 1))
    trunc = tf.reshape(trunc, (batch_size, 1, 1))

    sersic = tf.exp(-tf.math.pow(z/r0, 1/n))  * scale  * scale

    trunc = trunc * tf.cast(trunc>0., tf.float32) + tf.math.sqrt(nx*nx*1.+ny*ny*1.) * scale * tf.cast(trunc==0., tf.float32)
    sersic = tf.cast((z<trunc), tf.float32) * sersic

    sersic = sersic_normalization(sersic, n, r0)
    
    sersic *= flux
    
    return sersic


def sersic_flux_normalization(flux, trunc, r0, n, flux_untruncated):
  """Convenience function to compute the flux of a Sersic profile
  Need to account cases where n is big and tf.math.exp(tf.math.igamma(n,z)) is inf
  or r0 is 0.
  """
  def check(n, r0, trunc):
    r = trunc / r0
    z = tf.math.pow(r, 1./n)
    return tf.math.igamma(2.*n, z)
  mask = tf.cast(tf.math.is_nan(check(n, r0, trunc)), tf.float32)
  r0 = r0 * (1-mask) + tf.ones_like(r0) * mask
  n = n * (1-mask) + tf.ones_like(n) * mask
  flux_fraction = integratedflux(trunc, r0, n)
  normalize = tf.cast(tf.math.logical_and(trunc > 0., tf.math.logical_not(flux_untruncated)), tf.float32)
  flux = flux * (1-mask) + tf.ones_like(flux)*mask
  flux = flux * (1-normalize) + flux / flux_fraction * normalize
  return flux


def sersic_normalization(sersic, n, r0):
  """Convenience function to normalize the sersic profile
  Need to account cases where n is big and tf.math.exp(tf.math.lgamma(n)) is inf
  """
  check = tf.math.exp(tf.math.lgamma(2.*n))
  isinf = tf.math.is_inf(check)
  isr0 = r0==0.
  mask = tf.cast(tf.math.logical_or(isinf, isr0), tf.float32)
  n = n * (1.-mask) + tf.ones_like(n) * mask
  r0 = r0 * (1.-mask) + tf.ones_like(r0) * mask
  Z = 2 * math.pi * r0 * r0 * n * tf.math.exp(tf.math.lgamma(2.*n))
  isZ = tf.cast(Z==0., tf.float32)
  Z = Z * (1-isZ) + tf.ones_like(Z) * isZ

  sersic = sersic/Z * (1-mask) + tf.zeros_like(sersic) * (mask)
  sersic = sersic * (1-isZ) + tf.zeros_like(sersic) * isZ
  return sersic


def integratedflux(trunc, r0, n):
  """Convenience function to compute the fraction of the total flux enclosed within a given radius
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
    apply = tf.cast(b>=0., tf.float32)
    b = b * apply + tf.ones(b.shape)* (1-apply)
    res = 2.*tf.math.igamma(2*n, b) - 1.
    res = res * apply - 100.*tf.ones(b.shape)* (1-apply)
    return res

  def update(z, zprev, not_done):
    def compute_denom(z, zprev):
      apply = tf.cast(z!=zprev, tf.float32)
      zprev = apply*zprev + -42.*tf.ones(zprev.shape)*(1.-apply)
      res = 1. / (z - zprev)
      res = res * apply - 42.*tf.ones(res.shape)*(1-apply)
      return res

    f_zprez = func(zprev) * tf.cast(func(zprev)!=-100., tf.float32)

    factor = compute_denom(z, zprev)
    apply = tf.cast(factor!=42., tf.float32)
  
    fp_z = (func(z) - f_zprez) * factor
    fp_z = fp_z * apply + tf.ones(fp_z.shape) * (1-apply)
    not_done = tf.math.logical_and(not_done, func(zprev)!=-100.)
    fp_z = fp_z * tf.cast(not_done, tf.float32) + tf.ones(fp_z.shape) * (1-tf.cast(not_done, tf.float32))
  
    return z - func(z)/fp_z * tf.cast(not_done, tf.float32), not_done
  
  def Broyden_solver(b2, b1):
    z_prev, (z, not_done) = b2, update(b2, b1, tf.cast(b1>=0., tf.bool))
    not_done = tf.math.logical_and(not_done, func(z) > 1e-5)

    def condition(z, z_prev, not_done): 
      return tf.reduce_any(not_done)

    def body(z, z_prev, not_done):
      z_prev, (z, not_done) = z, update(z, z_prev, not_done)
      not_done = tf.math.logical_and(not_done, func(z) > 1e-6)
      return z, z_prev, not_done
    
    z, _, _ = tf.while_loop(condition, body, (z, z_prev, not_done))
    return z
  
  return Broyden_solver(b2,b1)

def calculateHLRFactor(n, flux_fraction=0.):
  """Calculate the half-light-radius in units of the scale radius.
  """
  b = calculate_b(n)
  return tf.math.pow(b,n)