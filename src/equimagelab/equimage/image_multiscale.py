# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.1 / 2025.03.26
# Doc OK.

"""Multiscale transformations."""

import pywt
import numpy as np
from copy import deepcopy

from . import params
from . import image as img
from . import image_utils as imgutils

class WaveletTransform:
  """Wavelet transform class."""

  def iwt(self):
    """
    """
    if self.type == "dwt":
      data = pywt.waverec2(self.coeffs, wavelet = self.wavelet, mode = self.mode, axes = (-2, -1))
    elif self.type == "swt":
      data = pywt.iswt2(self.coeffs, wavelet = self.wavelet, norm = self.norm, axes = (-2, -1))
      height, width = self.size
      ptop, pleft = self.padding
      data = data[..., ptop:ptop+height, pleft:pleft+width]
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return img.Image(data, colorspace = self.colorspace, colormodel = self.colormodel) if self.isImage else data

  def scale_levels(self, mult, inplace = False):
    """
    """
    if isinstance(mult, dict):
      m = np.ones(self.levels)
      for key, value in mult.items():
        if not isinstance(key, int): raise ValueError("Error, mult dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        m[key] = value
    else:
      m = np.asarray(mult)
      if m.ndim != 1: raise ValueError("Error, mult must be a dictionary or be mappable to a 1D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    for level in range(min(self.levels, m.size)):
      cH, cV, cD = output.coeffs[level+1]
      cH *= m[level]
      cV *= m[level]
      cD *= m[level]
    return output

def swt(image, levels, wavelet = "default", mode = "symmetric", start = 0):
  """
  """
  isImage = issubclass(type(image), img.Image)
  if isImage:
    width, height = image.get_size()
    data = image.image
  elif imgutils.is_valid_image(image):
    width, height = image.shape[-1], image.shape[-2]
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Pad the image so that the width and height are multiples of 2**level.
  length = 2**levels
  pwidth  = int(np.ceil(width /length))*length
  pheight = int(np.ceil(height/length))*length
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
  if wavelet == "default": wavelet = params.defwavelet
  wt = WaveletTransform()
  wt.type = "swt"
  wt.wavelet = wavelet
  wt.levels = levels
  wt.start = start
  wt.norm = True
  wt.coeffs = pywt.swt2(padded, wavelet = wavelet, level = levels, start_level = start, trim_approx = True, norm = True, axes = (-2, -1))
  wt.size = (height, width)
  wt.padding = (ptop, pleft)
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

def dwt(image, levels, wavelet = "default", mode = "symmetric"):
  """
  """
  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Compute the discrete wavelet transform.
  if wavelet == "default": wavelet = params.defwavelet
  wt = WaveletTransform()
  wt.type = "dwt"
  wt.wavelet = wavelet
  wt.levels = levels
  wt.start = 0
  wt.mode = mode
  wt.coeffs = pywt.wavedec2(data, wavelet = wavelet, level = levels, mode = mode, axes = (-2, -1))
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

  def swt(self, levels, wavelet = "default", mode = "symmetric", start = 0):
    """
    """
    return swt(self, levels, wavelet = wavelet, mode = mode, start = start)

  def dwt(self, levels, wavelet = "default", mode = "symmetric"):
    """
    """
    return dwt(self, levels, wavelet = wavelet, mode = mode)
