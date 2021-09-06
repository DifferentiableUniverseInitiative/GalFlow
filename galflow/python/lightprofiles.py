# Functions computing light profiles
# based on https://galsim-developers.github.io/GalSim/_build/html/sb.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

from tensorflow.python.types.core import Value

# The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
fwhm_factor = 2*math.sqrt(2*math.log(2))
# The half-light-radius is sqrt(2 ln2) sigma
hlr_factor = math.sqrt(2*math.log(2))

def gaussian(fwhm=None, half_light_radius=None, sigma=None, flux=1., nx=None, ny=None):
  """
  I(r) = exp(-r^2 /2/sigma^2)
  """
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
      sigma = fwhm / fwhm_factor
  elif half_light_radius is not None:
    if sigma is not None:
      raise ValueError("Only one of sigma, fwhm, and half_light_radius may be specified,\
        fwhm={}, half_light_radius={}, sigma={}".format(fwhm, half_light_radius, sigma))
    else:
      sigma = half_light_radius / hlr_factor
  elif sigma is None:
    raise ValueError("One of sigma, fwhm, and half_light_radius must be specified,\
        fwhm={}, half_light_radius={}, sigma={}".format(fwhm, half_light_radius, sigma))
    
  x, y = tf.cast(tf.meshgrid(tf.range(nx), tf.range(ny)), tf.float32)
  z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32))
  gaussian = flux * tf.exp(-z*z / 2 / sigma/sigma) / 2 / math.pi / sigma / sigma

  return gaussian

def sersic(n, half_light_radius=None, scale_radius=None, flux=1., trunc=0., flux_untruncated=False, nx=None, ny=None):
  """
  I also want specific parameters, see https://galsim-developers.github.io/GalSim/_build/html/sb.html
  """
  if nx is None:
    if ny is None:
      raise ValueError("Either nx or ny or both must be specified")
    else:
      nx = ny
  else:
    if ny is None:
      ny = nx

  if trunc < 0.:
    raise ValueError("Sersic trunc must be > 0.")
    
  if half_light_radius is not None:
    if scale_radius is not None:
      raise ValueError("Only one of scale_radius and half_light_radius may be specified,\
        scale_radius={}, half_light_radius={}".format(scale_radius, half_light_radius))
    else:
      raise NotImplementedError("to implement, see https://github.com/GalSim-developers/GalSim/blob/6c1eb247df86ec03d1a2c9c8024fe57b1bd4a154/src/SBSersic.cpp#L656")
  elif scale_radius is not None:
    r0 = scale_radius
  else:
    raise ValueError("Either scale_radius or half_light_radius must be specified for Sersic,\
      half_light_radius={}, scale_radius={}".format(half_light_radius, scale_radius))
    
  if trunc > 0.:
    flux_fraction = integratedflux(trunc, r0, n)
    if not flux_untruncated:
      flux /= flux_fraction
  else:
    flux_fraction = 1.

  x, y = tf.cast(tf.meshgrid(tf.range(nx), tf.range(ny)), tf.float32)
  z = tf.sqrt(tf.cast((x+.5-nx/2)**2 + (y+.5-ny/2)**2, tf.float32))

  sersic = tf.exp(-tf.math.pow(z/r0, 1/n))
  if trunc > 0.:
    sersic = tf.cast((z<trunc), tf.float32) * sersic
  sersic /= math.pi * r0 * r0 * math.factorial(2.*n)
  sersic *= flux

  return sersic

def integratedflux(trunc, r0, n):
  r = trunc / r0
  z = tf.math.pow(r, 1./n)
  return tf.math.igamma(2.*n, z)
  