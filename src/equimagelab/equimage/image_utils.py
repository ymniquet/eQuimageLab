# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image utils."""

import numpy as np

from . import params
from . import helpers

#####################
# Image validation. #
#####################

def is_valid_image(image):
  """Return True if the input is a valid image candidate, False otherwise."""
  if not issubclass(type(image), np.ndarray): return False
  if image.ndim != 3: return False
  if image.shape[0] != 3: return False
  if image.dtype not in [np.float32, np.float64]: return False

##########################
# Image transformations. #
##########################

def clip(image, vmin = 0., vmax = 1.):
  """Clip the input image in the range [vmin, vmax]."""
  return np.clip(image, vmin, vmax)

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ##################
  # Image queries. #
  ##################

  def is_out_of_range(self):
    """Return True if the image is out-of-range (data < 0 or > 1 in any channel), False otherwise."""
    image = self.image(cls = np.ndarray)
    return np.any(image < -params.IMGTOL) or np.any(image > 1.+params.IMGTOL)

  ##############
  # Templates. #
  ##############

  def empty(self):
    """Return an empty image with the same size."""
    return np.empty_like(self)

  def black(self):
    """Return a black image with the same size."""
    return np.zeros_like(self)

  ##############################
  # Clipping & scaling pixels. #
  ##############################

  def clip(self, vmin = 0., vmax = 1.):
    """Clip the image in the range [vmin, vmax]."""
    return clip(self, vmin, vmax)

  def scale_pixels(self, source, target, cutoff = params.IMGTOL):
    """Scale all pixels by the ratio target/source.
       Wherever abs(source) < cutoff, set all channels to target"""
    return self.newImage_like(self, helpers.scale_pixels(self, source, target, cutoff))
