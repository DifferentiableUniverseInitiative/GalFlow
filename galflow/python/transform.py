# Implements various transformation operations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from galflow.python.tfutils import perspective_transform, PixelType

__all__ = ["transform", "compose_transforms", "shift_transformation"]

def compose_transforms(transforms, name=None):
  """Composes the transforms tensors.
  Args:
    transforms: List of `Tensor`, image projective transforms to be composed. Each
      transform is length 8 (single transform) or shape (N, 8) (batched
      transforms). The shapes of all inputs must be equal, and at least one
      input must be given.
    name: `string`, name of the operation.
  Returns:
    `Tensor`, a composed transform tensor. When passed to `transform` op,
      equivalent to applying each of the given transforms to the image in
      order.
  """
  assert transforms, "transforms cannot be empty"
  with tf.name_scope(name or "compose_transforms"):
    composed = transforms[0]
    for tr in transforms[1:]:
      # Multiply batches of matrices.
      composed = tf.matmul(composed, tr)
    return composed

def shift_transformation(nx, ny):
  """ Computes affine transformation for specified pixel shift.
  """
  nx = tf.convert_to_tensor(nx, dtype=tf.float32)
  ny = tf.convert_to_tensor(ny, dtype=tf.float32)
  # Convert to transformation format
  shift = tf.reshape(tf.concat([nx, ny], axis=1), [-1, 2, 1])
  return tf.pad(shift,[[0,0],[0,1],[2,0]]) + tf.expand_dims(tf.eye(3),0)

def transform(img, transform_matrix, keep_center=True, name=None):
  """
  Function for transforming an image or kimage.

  Args:
  -----------
  img: `Tensor`, input image tensor of shape [bacth_size, nx, ny, nchannels]
  transform_matrix: `Tensor`, transormation matrix
  keep_center, `boolean`, when doing an interpolation in Fourier space, the center of pixels is on
    integer values, so set to true if transforming in Fourier space.
  name: `string`, name of the operation.

  Returns:
    `Tensor`, transformed image, tensor of shape [batch_size, nx, ny, nchannels]
  """
  with tf.name_scope(name or "transform"):
    img = tf.convert_to_tensor(img)
    # Extract shape of image
    nb, nx, ny, nc = img.get_shape().as_list()

    if keep_center:
      center = tf.convert_to_tensor([nx/2, ny/2, 1.], dtype=tf.float32)
      center = tf.tile(tf.reshape(center, [1,3,1]),[nb, 1, 1])
      # Compute the shift induced by the requested transform
      shift = center - tf.matmul(transform_matrix, center)
      transform_matrix = compose_transforms([shift_transformation(shift[:,0],
                                                                  shift[:,1]),
                                            transform_matrix])

    if img.dtype == tf.complex64:
      a = perspective_transform(tf.math.real(img),
                                transform_matrix=transform_matrix,
                                pixel_type=PixelType.INTEGER)
      b = perspective_transform(tf.math.imag(img),
                                transform_matrix=transform_matrix,
                                pixel_type=PixelType.INTEGER)
      return tf.complex(a,b)
    else:
      return perspective_transform(img, transform_matrix=transform_matrix)
