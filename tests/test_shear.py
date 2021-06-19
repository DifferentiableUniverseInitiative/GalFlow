import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose

import galflow as gf
import galsim

# Some parameters used for testing shearing transforms
gal_flux = 1.e5    # counts
gal_r0 = 2.7       # arcsec
g1 = 0.1           #
g2 = 0.2           #
pixel_scale = 0.2  # arcsec / pixel

def test_shear_linear_interpolant():
  """
  This test generates a simple galaxy light profile with GalSim, then
  shears it, and checks that under a linear interpolant, the same answer
  is recovered.
  """
  gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)
  # To make sure that GalSim is not cheating, i.e. using the analytic formula of the light profile
  # when computing the affine transformation, it might be a good idea to instantiate the image as
  # an interpolated image.
  # We also make sure GalSim is using the same kind of interpolation as us (bilinear for TF)
  reference_image = gal.drawImage(nx=256,ny=256, scale=pixel_scale)
  gal = galsim.InterpolatedImage(reference_image,
                                 x_interpolant='linear')

  # Apply shear with Galsim
  gal = gal.shear(g1=g1, g2=g2)

  # Draw the image with Galsim
  image_galsim = gal.drawImage(nx=256,ny=256,scale=pixel_scale,method='no_pixel').array

  # Now, same thing with GalFlow
  ##############################

  # We convert the reference image to tensor
  img = tf.reshape(tf.convert_to_tensor(reference_image.array, dtype=tf.float32),
                  [1, 256,256,1])
  # same for shear, to make sure everything is ok
  tfg1 = tf.convert_to_tensor(np.atleast_1d(g1), tf.float32)
  tfg2 = tf.convert_to_tensor(np.atleast_1d(g2), tf.float32)

  # Apply shear
  image_galflow = gf.shear(img, tfg1, tfg2)

  assert_allclose(image_galsim, image_galflow[0,:,:,0], rtol=2e-04)
