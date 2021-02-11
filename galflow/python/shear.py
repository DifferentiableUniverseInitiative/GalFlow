# Functions computing shear related affine transformations
# based on https://github.com/GalSim-developers/GalSim/blob/main/galsim/shear.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from galflow.python.transform import transform

__all__ = ["shear", "shear_transformation"]

def shear_transformation(g1, g2):
  """
  Function to compute the affine transformation corresponding to a given shear.
  This function uses the reduced shear definition:

    :math:`|g| = \frac{a - b}{a + b}`

  TODO: better documentation (the following is stolen from galsim)

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
  Parameters
  ----------
    g1: Tensor
      The first component of the shear in the "reduced shear" definition.
    g2: Tensor
     The second component of the shear in the "reduced shear" definition.

  Returns
  -------
    Transformation matrix format, i.e [(batch), 3, 3]
  """
  g1 = tf.convert_to_tensor(g1, dtype=tf.float32)
  g2 = tf.convert_to_tensor(g2, dtype=tf.float32)
  gsqr = g1**2 + g2**2

  # Building a batched jacobian
  jac = tf.stack([ 1. + g1, g2,
                g2, 1. - g1], axis=1) / tf.expand_dims(tf.sqrt(1.- gsqr),1)
  jac = tf.reshape(jac, [-1,2,2])

  # Inverting these jacobians to follow the TF definition
  jac = tf.linalg.inv(jac)
  jac = tf.pad(jac, tf.constant([[0, 0], [0, 1],[0,1]]) )
  jac = jac + tf.pad(tf.ones([16,1,1]), tf.constant([[0,0],[2,0],[2,0]]))
  return jac

def shear(img, g1, g2):
  """ Convenience function to apply a shear to an input image
  """
  transform_matrix = shear_transformation(g1, g2)
  return transform(img, transform_matrix)
