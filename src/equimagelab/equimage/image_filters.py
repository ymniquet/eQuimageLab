# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Image filters."""

import numpy as np
import scipy.ndimage as ndimg

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  def sharpen(self, mode = "reflect", channels = ""):
    """Apply a sharpening (Laplacian) convolution filter to selected channels of the image.

    Args:
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").

      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd -> 0000|abcd|0000).

    Returns:
      Image: The sharpened image.
    """
    # Translate modes.
    if mode == "zero": mode = "constant"
    # Set-up Laplacian kernel.
    kernel = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]], dtype = params.IMGTYPE)
    # Convolve selected channels with the kernel.
    return self.apply_channels(lambda channel: ndimg.convolve(channel, kernel, mode = mode, cval = 0.), channels, multi = False)

  def remove_hot_pixels(self, ratio, mode = "reflect", channels = ""):
    """Remove hot pixels in selected channels of the image.

    All pixels of a selected channel greater than ratio times the eight nearest-neighbors average
    are replaced by this average.

    Args:
      ratio (float): The threshold for hot pixels detection.
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).

      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd -> 0000|abcd|0000).

    Returns:
      Image: The processed image.
    """

    def remove_hot_pixels_channel(channel):
      """Remove hot pixels from the input channel data."""
      avg = ndimg.convolve(channel, kernel, mode = mode, cval = 0.)/nnn
      return np.where(channel > ratio*avg, avg, channel)

    if ratio <= 0.: raise ValueError("Error, ratio must be > 0.")
    # Translate modes.
    if mode == "zero": mode = "constant"
    # Set-up the (unnormalized) kernel for nearest-neighbors average.
    kernel = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype = params.IMGTYPE)
    # Normalize with respect to the actual number of nearest neighbors.
    nnn = ndimg.convolve(np.ones(*self.get_size()), kernel, mode = mode, cval = 0.)
    return self.apply_channels(remove_hot_pixels_channel, channels, multi = False)
