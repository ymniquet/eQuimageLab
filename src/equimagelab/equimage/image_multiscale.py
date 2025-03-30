# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.1 / 2025.03.26
# Doc OK.

"""Multiscale transformations."""

import numpy as np
import pywt

from .image import Image
from .helpers import is_valid_image

class WaveletTransform:
  """Wavelet transform object."""

  def iwt(self):
    """
    """
    if wt.type == "dwt":
      data = pywt.waverec2(self.coeffs, wavelet = self.wavelet, mode = self.mode, axes = (-2, -1))
    elif wt.type == "swt":
      data = pywt.iswt2(self.coeffs, wavelet = self.wavelet, norm = self.norm, axes = (-2, -1))
      height, width = self.size
      ptop, pleft = self.padding
      data = data[..., ptop:ptop+height, pleft:pleft+width]
    else:
      raise ValueError(f"Unknown wavelet transform type '{wt.type}'.")
    return Image(data, self.colorspace, self.colormodel) if self.isImage else data

def swt(image, level, wavelet = "coif4", mode = "symmetric", start = 0):
  """
  """
  isImage = issubclass(image, Image)
  if isImage:
    width, height = image.get_size()
    data = image.image
  elif is_valid_image(image):
    width, height = image.shape[-1], image.shape[-2]
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Pad the image so that the width and height of the image are powers of 2.
  pwidth  = int(2**(np.ceil(np.log2(width))))
  pheight = int(2**(np.ceil(np.log2(height))))
  pleft = (pwidth-width)//2 ; pright = pwidth-width-pleft
  ptop = (pheight-height)//2 ; pbottom = pheight-height-ptop
  padding = (data.ndim-2)*((0, 0),)+((ptop, pbottom), (pleft, pright))
  if mode == "zero": # Translate pywt boundary modes.
    mode = "constant"
  elif mode == "constant":
    mode = "edge"
  elif mode == "periodic":
    mode = "wrap"
  elif mode not in ["symmetric", "reflect"]:
    raise ValueError(f"Error, unknown boundary mode '{mode}'.")
  padded = np.pad(data, padding, mode = mode)
  # Compute the stationary wavelet transform.
  wt = WaveletTransform()
  wt.type = "swt"
  wt.wavelet = wavelet
  wt.norm = True
  wt.coeffs = pywt.swt2(padded, wavelet = wavelet, level = level, start_level = start, trim_approx = True, norm = True, axes = (-2, -1))
  wt.size = (height, width)
  wt.padding = (ptop, pleft)
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

def dwt(image, level, wavelet = "coif4", mode = "symmetric"):
  """
  """
  isImage = issubclass(image, Image)
  if isImage:
    data = image.image
  elif is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Compute the discrete wavelet transform.
  wt = WaveletTransform()
  wt.type = "dwt"
  wt.wavelet = wavelet
  wt.mode = mode
  wt.coeffs = pywt.wavedec2(data, wavelet = wavelet, level = level, mode = mode, axes = (-2, -1))
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def swt(self, level, wavelet = "coif4", mode = "symmetric", start = 0):
    """
    """
    return swt(self.image, level, wavelet = wavelet, mode = mode, start = start)

  def dwt(self, level, wavelet = "coif4", mode = "symmetric"):
    """
    """
    return dwt(self.image, level, wavelet = wavelet, mode = mode)
