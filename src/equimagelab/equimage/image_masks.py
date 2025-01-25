# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Image masks."""

import numpy as np
import scipy.ndimage as ndimg
import skimage.morphology as skimo

from . import params

def threshold_mask(filtered, threshold, extend = 0, smooth = 0):
  """Set-up a threshold mask.

  Returns the pixels of the image such that filtered >= threshold.

  See also:
    Image.filter

  Args:
    filtered (numpy.ndarray): The output of a filter (e.g., local average, ...) applied to the image (see Image.filter).
    threshold (float): The threshold for the mask. The mask is 1 wherever filtered >= threshold, and 0 elsewhere.
    extend (int, optional): Once computed, the mask can be extended/eroded by extend pixels (default 0). The mask
      is extended if extend > 0, and eroded if extend < 0.
    smooth (int, optional): Once extended, the edges of the mask can be smoothed over smooth pixels (default 0).

  Returns:
    numpy.ndarray: The mask as an array with the same shape as filtered (*not* converted to a grayscale Image object).
  """
  # Threshold the filter.
  mask = (filtered >= threshold)
  # Extend the mask.
  if extend > 0:
    mask = skimo.isotropic_dilation(mask, extend)
  elif extend < 0:
    mask = skimo.isotropic_erosion(mask, -extend)
  # Smooth the mask.
  mask = mask.astype(params.imagetype)
  if smooth > 0:
    kernel = skimo.disk(smooth, dtype = params.imagetype)
    kernel /= np.sum(kernel)
    mask = ndimg.convolve(mask, kernel, mode = "reflect")
  return mask

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def filter(self, channel, filter, radius, mode = "reflect"):
    """Apply a spatial filter to a selected channel of the image.

    The main purpose of this method is to prepare masks for image processing.

    See also:
      threshold_mask

    Args:
      channel (str): The selected channel:

        - "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
          Apply the filter to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the filter to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the filter to the HSV saturation (RGB and HSV images).
        - "L": Apply the filter to the luma (RGB and grayscale images).
        - "L*": Apply the filter to the CIE lightness L* (RGB and grayscale images).

      filter (str): The filter:

        - "mean": Return the average of the channel within a disk around each pixel.
        - "median": Return the median of the channel within a disk around each pixel.
        - "gaussian": Return the gaussian average of the channel around each pixel.
        - "maximum": Return the maximum of the channel within a disk around each pixel.

      radius (float): The radius of the disk in pixels. The standard deviation for gaussian average is radius/3.
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd -> 0000|abcd|0000).

    Returns:
      numpy.ndarray: A (image height, image width) array with the output of the filter,
      *not* converted to a grayscale Image object.
    """
    if mode == "zero": mode = "constant" # Translate modes.
    data = self.get_channel(channel)
    if filter == "gaussian":
      return ndimg.gaussian_filter(data, sigma = radius/3., mode = mode, cval = 0.)
    elif filter == "mean":
      kernel = skimo.disk(radius, dtype = self.dtype)
      kernel /= np.sum(kernel)
      return ndimg.convolve(data, kernel, mode = mode, cval = 0.)
    elif filter == "median":
      return ndimg.median_filter(data, footprint = skimo.disk(radius), mode = mode, cval = 0.)
    elif filter == "maximum":
      return ndimg.maximum_filter(data, footprint = skimo.disk(radius), mode = mode, cval = 0.)
    else:
      raise ValueError(f"Error, unknown filter '{filter}'.")
