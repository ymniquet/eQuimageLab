# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.2.0 / 2025.02.02
# Sphinx OK.

"""Image filters."""

import numpy as np
import scipy.ndimage as ndimg

from .image_stretch import hms, Dharmonic_through

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
    kernel = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype = self.dtype)
    # Normalize with respect to the actual number of nearest neighbors.
    nnn = ndimg.convolve(np.ones(*self.get_size(), dtype = self.dtype), kernel, mode = mode, cval = 0.)
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
    kernel = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]], dtype = self.dtype)
    # Convolve selected channels with the kernel.
    return self.apply_channels(lambda channel: ndimg.convolve(channel, kernel, mode = mode, cval = 0.), channels, multi = False)

  def LDBS(self, sigma, amount, threshold, channels = "L*", mode = "reflect", full_output = False):
    """Light-dependent blur & sharpen (LDBS).

    Blurs low-brightness and sharpens high-brightness areas.

    The background of astronomical images usually remains noisy. This is the Poisson (photon counting)
    noise typical of low brightness areas. We may want to "blur" this background by applying a "low-pass"
    filter that softens small scale features - such as a convolution with a gaussian:

      blurred = image.gaussian_filter(sigma = 5) # Gaussian blur with a std dev of 5 pixels.

    Yet this operation would also blur the objects of interest (the galaxy, nebula...) !

    As a matter of fact, most of these objects already lack sharpness... We may thus want, on the opposite,
    to apply a "high-pass" filter that enhances small scale features. Since the convolution with a gaussian
    is a low-pass filter, the following operation:

      sharpened = (1 + q) * image - q * blurred, q > 0

    is a high-pass filter known as an "unsharp mask". We can also rewrite this operation as a conventional
    blend with a mixing coefficient m > 1:

      sharpened = (1 - m) * blurred + m * image, m > 1

    Yet such an unsharp mask would also enhance the noise in the background !

    We can meet both requirements by making m dependent on the lightness. Namely, we want m ~ 0 where
    the lightness is "small" (the background), and m > 1 where the lightness is "large" (the object of
    interest). We may use as a starting point:

      m = (1 + a) * image

    where a > 0 controls image sharpening in the bright areas. In practice, we gain flexibility by stretching
    the image to control how fast we switch from blurring to sharpening, e.g.:

      m = (1 + a) * hms(image, D)

    where hms is the harmonic stretch with strength D. The latter can be calculated to switch (m = 1) at a
    given threshold.

    Application of the LDBS to all channels (as in the above equations) can lead to significant "color spilling".
    It is preferable to apply LDBS to the lightness L*, luma L or value V (i.e. setting image = L*, L or V and
    updating that channel with the output of the LDBS).

    Args:
      sigma (float): The standard deviation of the gaussian blur (pixels).
      amount (float): The full strength of the unsharp mask (must be > 0).
      threshold (float): The threshold for sharpening (expected in ]0, 1[).
        The image is blurred below the threshold, and sharpened above.
      channels (str, optional): The channel(s) for LDBS (can be "" for all channels, "V" for Value, "L" for
        luma or "L*" for lightness). Default is "L*".
      mode (str, optional): How to extend the image across its boundaries (for the gaussian blur):

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd -> 0000|abcd|0000).

      full_output (bool, optional): If False (default), only return the processed image. If True, return
        the processed image, as well as:

        - The blurred image if channels = "".
        - The input, blurred and output channel as grayscale images if channels = "V", "L" or "L*".

    Returns:
      Image: The processed image(s) (see the full_output argument).
    """
    channels = channels.strip()
    if channels not in ["", "V", "L", "L*"]: raise ValueError("Error, channels must be '', 'V', 'L' or 'L*'.")
    if amount <= 0.: raise ValueError("Error amount must be > 0.")
    if threshold < .0001 or threshold >= .9999: raise ValueError("Error, threshold must be >= 0.0001 and <= 0.9999.")
    D = Dharmonic_through(threshold, 1./(1.+amount))
    clipped = self.clip() # Clip the image before LDBS.
    if channels == "":
      blurred = clipped.gaussian_filter(sigma, mode = mode)
      output = blurred.blend(clipped, (1.+amount)*hms(clipped, D)).clip()
      if full_output:
        return output, blurred
      else:
        return output
    else:
      cin = clipped.grayscale(channels)
      cblurred = cin.gaussian_filter(sigma, mode = mode)
      cout = cblurred.blend(cin, (1.+amount)*hms(cin, D)).clip()
      output = clipped.set_channel(channels, cout.lightness() if channels == "L*" else cout.image[0])
      if full_output:
        return output, cin, cblurred, cout
      else:
        return output
