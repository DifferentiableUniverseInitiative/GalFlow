from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.image as tfimage

__all__ = ["convolve"]

def k_wrapping(kimage, wrap_factor=2):
  """
  Wraps kspace image of a real image to decrease its resolution by specified
  factor
  """
  batch_size, Nkx, Nky = tf.shape(kimage)

  # First wrap around the non hermitian dimension
  rkimage = kimage + tf.roll(kimage, shift=Nkx//wrap_factor, axis=1)

  # Now take care of the hermitian part
  revrkimage = tf.reverse(tf.math.conj(tf.reverse(rkimage, axis=[2])), axis=[1])

  # These masks take care of the special case of the 0th frequency
  mask = np.ones((1, Nkx, Nky))
  mask[:,0,:]=0
  mask[:,2*Npix-1,:]=0
  rkimage2 = rkimage + revrkimage*mask
  mask = np.zeros((1, Nkx, Nky))
  mask[:,2*Npix-1,:]=1
  rkimage2 = rkimage2 + tf.roll(revrkimage,shift=-1, axis=1) *mask

  # Now that we have wrapped the image, we can truncate it to the desired size
  kimage = rkimage2[:, :Nkx//wrap_factor, :(Nky-1)//wrap_factor+1]

  return kimage


def convolve(images, kpsf,
             x_interpolant=tfimage.ResizeMethod.BICUBIC,
             zero_padding_factor=2,
             interp_factor=2):
  """
  Convolution of input images with provided k-space psf tensor.

  This function assumes that the k-space PSF is already prodided with the
  stepk and maxk corresponding to the specified interpolation and zero padding
  factors.
  """
  batch_size, Nx, Ny, Nc = tf.shape(images)
  # For now, we only support square, even images, in only one band
  assert Nx == Ny
  assert Nx % 2 == 0
  assert Nc == 1

  # First step is to interpolate the image on a finer grid
  im = tfimage.resize(images,
                      [Nx*interp_factor, Ny*interp_factor],
                      method=x_interpolant,
                      align_corners=False)

  # Apply zero padding to avoid image wrapping
  padding = (zero_padding_factor - 1)*interp_factor*Nx
  im = tf.pad(im[...,0], [[0,0],[0, int(padding)],[0, int(padding)]])

  # Compute DFT
  imk = tf.spectral.rfft2d(im)

  # Perform k-space convolution
  imk = imk * kpsf

  # Apply frequency wrapping to reach target image resolution
  imk = k_wrapping(imk, interp_factor)

  # Perform inverse Fourier Transform
  conv_images = tf.spectral.irfft2d(im)

  # Removes zero padding, and rebuilds channel
  return tf.expand_dims(conv_images[:,:Nx, :Ny], axis=-1)
