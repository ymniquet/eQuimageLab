# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Image utils."""

import numpy as np

from . import params
from . import helpers

#####################
# Image validation. #
#####################

def is_valid_image(image):
  """Return True if the input is a valid image candidate, False otherwise.

  Args:
    image (numpy.ndarray): The image candidate.

  Returns:
    bool: True if the input is a valid image candidate, False otherwise.
  """
  if not issubclass(type(image), np.ndarray): return False
  if image.ndim == 3:
    if image.shape[0] not in [1, 3]: return False
  elif image.ndim != 2:
    return False
  if image.dtype not in [np.float32, np.float64]: return False

##########################
# Image transformations. #
##########################

def clip(image, vmin = 0., vmax = 1.):
  """Clip the input image in the range [vmin, vmax].

  Args:
    image (numpy.ndarray): The input image.
    vmin (float, optional): The lower clip bound (default 0).
    vmax (float, optional): The upper clip bound (default 1).

  Returns:
    numpy.ndarray: The clipped image.
  """
  return np.clip(image, vmin, vmax)

def blend(image1, image2, mixing):
  """Blend two images.

  Returns image1*(1-mixing)+image2*mixing.

  Args:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.
    mixing (float or numpy.ndarray for pixel-dependent mixing): The mixing coefficient(s).

  Returns:
    numpy.ndarray: The blended image image1*(1-mixing)+image2*mixing.
  """
  return image1*(1.-mixing)+image2*mixing

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ##################
  # Image queries. #
  ##################

  def is_out_of_range(self):
    """Return True if the image is out-of-range (data < 0 or > 1 in any channel), False otherwise.

    Returns:
      bool: True if the image is out-of-range, False otherwise.
    """
    return np.any(self.image < -params.IMGTOL) or np.any(self.image > 1.+params.IMGTOL)

  ##############
  # Templates. #
  ##############

  def empty(self):
    """Return an empty image with same size as the object.

    Returns:
      Image: An empty image with the same size as self.
    """
    return np.empty_like(self)

  def black(self):
    """Return a black image with same size as the object.

    Returns:
      Image: An black image with the same size as self.
    """
    return np.zeros_like(self)

  ##############################
  # Clipping & scaling pixels. #
  ##############################

  def clip(self, vmin = 0., vmax = 1.):
    """Clip the image in the range [vmin, vmax].

    Args:
      vmin (float, optional): The lower clip bound (default 0).
      vmax (float, optional): The upper clip bound (default 1).

    Returns:
      Image: The clipped image.
    """
    return np.clip(self, vmin, vmax)

  def scale_pixels(self, source, target, cutoff = params.IMGTOL):
    """Scale all pixels of the image by the ratio target/source. Wherever abs(source) < cutoff, set all channels to target.

    Args:
      source (np.arrray): The source values for scaling (must be the same size as the image).
      target (np.arrray): The target values for scaling (must be the same size as the image).
      cutoff (float, optional): Threshold for scaling. Defaults to `equimage.params.IMGTOL`.

    Returns:
      Image: The scaled image.
    """
    return self.newImage(helpers.scale_pixels(self.image, np.asarray(source), np.asarray(target), cutoff))

  #############
  # Blending. #
  #############

  def blend(self, image, mixing):
    """Blend with the input image.

    Returns self*(1-mixing)+image*mixing.
    The images must share the same shape, color space and color model.

    Args:
      image (Image): The image to blend with.
      mixing (float or numpy.ndarray for pixel-dependent mixing): The mixing coefficient(s).

    Returns:
      Image: The blended image self*(1-mixing)+image*mixing.
    """
    if self.get_shape() != image.get_shape():
      raise ValueError("Error, the images must share the same size & number of channels.")
    if self.colorspace != image.colorspace:
      raise ValueError("Error, the images must share the same color space !")
    if self.colormodel != image.colormodel:
      raise ValueError("Error, the images must share the same color model !")
    return self.newImage(self.image*(1.-mixing)+image.image*mixing)
