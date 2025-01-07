# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.2 / 2024.12.28
# Sphinx OK.

"""Image filters."""

import numpy as np
import scipy.ndimage as ndimg

from .image_stretch import hms, harmonic_through

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

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
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

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
    kernel = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype = self.image.dtype)
    # Normalize with respect to the actual number of nearest neighbors.
    nnn = ndimg.convolve(np.ones(*self.get_size(), dtype = self.image.dtype), kernel, mode = mode, cval = 0.)
    return self.apply_channels(remove_hot_pixels_channel, channels, multi = False)

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
        - "Ls": Apply the operation to the luma, and protect highlights by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, and protect highlights by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "Ln": Apply the operation to the luma, and protect highlights by normalization.
          (after the operation, the image is normalized so that all pixels fall in the [0, 1] range).
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

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
    kernel = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]], dtype = self.image.dtype)
    # Convolve selected channels with the kernel.
    return self.apply_channels(lambda channel: ndimg.convolve(channel, kernel, mode = mode, cval = 0.), channels, multi = False)

  def LDBS(self, sigma, amount, threshold, channel = "L*", mode = "reflect", full_output = False):
    """Light-dependent blur & sharpen (LDBS).

    Args:
      sigma (float): The standard deviation of the gaussian blur (pixels).
      amount (float): The full strength of the unsharp mask (must be > 0).
      threshold (float): The threshold for sharpening.
        The image is blurred below the threshold, and sharpened above.
      channel (str, optional): The channel for LDBS (can be "L" for luma or "L*" for lightness). Default is "L*".
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd -> 0000|abcd|0000).

      full_output (bool, optional): If True, return the processed image, as well as the original, blurred
        and enhanced channel as grayscale images. If False (default), only return the processed image.

    Returns:
      Image: The processed image(s) (see the full_output argument).
    """
    channel = channel.strip()
    if channel not in ["L", "L*"]: raise ValueError("Error, channel must be L or L*.")
    if amount <= 0.: raise ValueError("Error amount must be > 0.")
    light = self.grayscale(channel)
    blurred = light.gaussian_filter(sigma, mode = mode)
    D = harmonic_through(threshold, 1./(1.+amount))
    enhanced = blurred.blend(light, (1.+amount)*hms(light, D))
    newchannel = enhanced.luma() if channel == "L" else enhanced.lightness()
    output = self.update_channel(channel, newchannel)
    if full_output:
      return output, light, blurred, enhanced
    else:
      return output
