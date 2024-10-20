# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image filters."""

import numpy as np
import scipy as sp

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  # TESTED.
  def sharpen(self, channels = ""):
    """Apply a sharpening (Laplacian) convolution filter to selected 'channels' of the image."""
    # Set-up Laplacian kernel.
    kernel = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]], dtype = params.IMGTYPE)
    # Convolve selected channels with the kernel.
    return self.apply_channels(lambda channel: sp.signal.convolve2d(channel, kernel, mode = "same", boundary = "fill", 
                                                fillvalue = 0.), channels, multi = False)

  def remove_hot_pixels(self, ratio, channels = ""):
    """Remove hot pixels in selected 'channels' of the image. 
       All pixels of a channel greater than 'ratio' times the eight nearest-neighbors average are replaced
       by this average.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
       
    def remove_hot_pixels_channel(channel):
      """Remove hot pixels from the input channel data."""
      avg = sp.signal.convolve2d(channel, kernel, mode = "same", boundary = "fill", fillvalue = 0.)/nnn
      return np.where(channel > ratio*avg, avg, channel)
      
    if ratio <= 0.: raise ValueError("Error, ratio must be > 0.")
    # Set-up the (unnormalized) kernel for nearest-neighbors average.
    kernel = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype = params.IMGTYPE)
    # Normalize with respect to the number of nearest neighbors.
    nnn = sp.signal.convolve2d(np.ones(self.height(), self.width()), kernel, mode = "same", boundary = "fill", fillvalue = 0.)
    return self.apply_channels(remove_hot_pixels_channel, channels, multi = False)
