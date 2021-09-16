# Functions computing shear related affine transformations
# based on https://github.com/GalSim-developers/GalSim/blob/main/galsim/shear.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from galflow.python.transform import transform

__all__ = ["shear", "shear_transformation"]

def shear_transformation(g1, g2, Fourier=False, name=None):
  """Function to compute the affine transformation corresponding to a given shear.
  This function uses the reduced shear definition:

    :math:`|g| = \frac{a - b}{a + b}`

  If a field is sheared by some shear, s, then the position (x,y) -> (x',y')
  according to:
  .. math::
      \left( \begin{array}{c} x^\prime \\ y^\prime \end{array} \right)
      = S \left( \begin{array}{c} x \\ y \end{array} \right)
  and :math:`S` is the return value of this function ``S = shear.getMatrix()``.
  Specifically, the matrix is
  .. math::
      S = \frac{1}{\sqrt{1-g^2}}
              \left( \begin{array}{cc} 1+g_1 & g_2 \\
                                       g_2 & 1-g_1 \end{array} \right)
  Args: 
    g1: `Tensor`, The first component of the shear in the "reduced shear" definition.
    g2: `Tensor`, The second component of the shear in the "reduced shear" definition.
    Fourier: `boolean`, when doing an interpolation in Fourier space, the center of pixels is on
    integer values, so set to true if transforming in Fourier space.
    name: `string`, name of the operation.
    
  Returns:
    `Tensor` of the transformation matrix of shape [(batch), 3, 3]
  """

  with tf.name_scope(name or "shear_transformation"):
    g1 = tf.convert_to_tensor(g1, dtype=tf.float32)
    g2 = tf.convert_to_tensor(g2, dtype=tf.float32)
    gsqr = g1**2 + g2**2

    # Building a batched jacobian
    jac = tf.stack([ 1. + g1, g2,
                  g2, 1. - g1], axis=1) / tf.expand_dims(tf.sqrt(1.- gsqr),1)
    jac = tf.reshape(jac, [-1,2,2])

    # Inverting these jacobians to follow the TF definition
    if Fourier:
      jac = tf.transpose(jac, [0,2,1])
    else:
      jac = tf.linalg.inv(jac)
    jac = tf.pad(jac, tf.constant([[0, 0], [0, 1],[0,1]]) )
    jac = jac + tf.pad(tf.reshape(tf.ones_like(g1), [-1,1,1]), tf.constant([[0,0],[2,0],[2,0]]))
    return jac

def shear(img, g1, g2):
  """ Convenience function to apply a shear to an input image or kimage.
  """
  transform_matrix = shear_transformation(g1, g2,
                                          Fourier=img.dtype == tf.complex64)
  return transform(img, transform_matrix)
