# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.0 / 2025.01.09
# Sphinx OK.

"""Image masks."""

import numpy as np
import scipy.ndimage as ndimg
import skimage.morphology as skimo

from . import params

####################################
# Binary & float masks management. #
####################################

def float_mask(mask):
  """Convert a boolean mask into a float mask.

  Args:
    mask (numpy.ndarray): The input boolean mask.

  Returns:
    numpy.ndarray: A float mask with datatype `equimagelab.equimage.params.imagetype`
      and values 1 where mask is True and 0 where mask is False. If already a float
      array, the input mask is returned as is.
  """
  return np.asarray(mask, dtype = params.imagetype)

def extend_bmask(mask, extend):
  """Extend or erode a boolean mask.

  Args:
    mask (numpy.ndarray): The input boolean mask.
    extend (int): The number of pixels by which the mask is extended.
      The mask is extended if extend > 0, and eroded if extend < 0.

  Returns:
    numpy.ndarray: The extended boolean mask.
  """
  if extend > 0:
    return skimo.isotropic_dilation(mask, extend)
  else:
    return skimo.isotropic_erosion(mask, -extend)

def smooth_mask(mask, radius, mode = "zero"):
  """Smooth a boolean or float mask.

  The input mask is converted into a float mask and convolved with a disk of radius smooth.

  Args:
    mask (numpy.ndarray): The input boolean or float mask.
    radius (float): The smoothing radius in pixels. The edges of the output float mask get
      smoothed over 2*radius pixels.
    mode (str, optional): How to extend the mask across its boundaries for the convolution:

      - "reflect": the mask is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
      - "mirror": the mask is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
      - "nearest": the mask is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
      - "zero" (default): the mask is padded with zeros (abcd -> 0000|abcd|0000).
      - "one": the mask is padded with ones (abcd -> 1111|abcd|1111).

  Returns:
    numpy.ndarray: The smoothed, float mask.
  """
  # Translate modes.
  if mode == "zero":
    mode = "constant"
    cval = 0.
  elif mode == "one":
    mode = "constant"
    cval = 1.
  # Convert into a float mask.
  fmask = float_mask(mask)
  # Smooth the float mask.
  if radius > 0.:
    kernel = skimo.disk(radius, dtype = params.imagetype)
    kernel /= np.sum(kernel)
    fmask = ndimg.convolve(fmask, kernel, mode = mode, cval = cval)
  return fmask

def threshold_bmask(filtered, threshold, extend = 0):
  """Set-up a threshold boolean mask.

  Returns the pixels of the image such that filtered >= threshold as a boolean mask.

  See also:
    Image.filter,
    threshold_fmask

  Args:
    filtered (numpy.ndarray): The output of a filter (e.g., local average, ...) applied to the image (see Image.filter).
    threshold (float): The threshold for the mask. The mask is True wherever filtered >= threshold, and False elsewhere.
    extend (int, optional): Once computed, the mask is extended/eroded by extend pixels (default 0).
      The mask is is extended if extend > 0, and eroded if extend < 0.

  Returns:
    numpy.ndarray: The mask as a boolean array with the same shape as filtered.
  """
  return extend_bmask(filtered >= threshold, extend)

def threshold_fmask(filtered, threshold, extend = 0, smooth = 0., mode = "zero"):
  """Set-up a threshold float mask.

  Returns the pixels of the image such that filtered >= threshold as a float mask.

  See also:
    Image.filter,
    smooth_mask,
    threshold_bmask

  Args:
    filtered (numpy.ndarray): The output of a filter (e.g., local average, ...) applied to the image (see Image.filter).
    threshold (float): The threshold for the mask. The mask is 1 wherever filtered >= threshold, and 0 elsewhere.
    extend (int, optional): Once computed, the mask is extended/eroded by extend pixels (default 0).
      The mask is is extended if extend > 0, and eroded if extend < 0.
    smooth (float, optional): Once extended, the edges of the mask are smoothed over smooth pixels (default 0).
    mode (str, optional): How to extend the mask across its boundaries for smoothing:

      - "reflect": the mask is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
      - "mirror": the mask is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
      - "nearest": the mask is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
      - "zero" (default): the mask is padded with zeros (abcd -> 0000|abcd|0000).
      - "one": the mask is padded with ones (abcd -> 1111|abcd|1111).

  Returns:
    numpy.ndarray: The mask as a float array with the same shape as filtered.
  """
  return smooth_mask(threshold_bmask(filtered, threshold, extend), smooth, mode = mode)

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def filter(self, channel, filter, radius, mode = "reflect"):
    """Apply a spatial filter to a selected channel of the image.

    The main purpose of this method is to prepare masks for image processing.

    See also:
      threshold_bmask
      threshold_fmask

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
