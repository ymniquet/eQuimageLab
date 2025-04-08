# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.1 / 2025.03.26
# Doc OK.

"""Multiscale transformations.

The following symbols are imported in the equimage/equimagelab namespaces for convenience:
  "dwt", "swt", "slt".
"""

__all__ = ["dwt", "swt", "slt"]

import pywt
import numpy as np
import scipy.ndimage as ndimg
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
    elif self.type == "slt":
      data = np.sum(self.coeffs, axis = 0)
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return img.Image(data, colorspace = self.colorspace, colormodel = self.colormodel) if self.isImage else data

  def scale_levels(self, mult, inplace = False):
    """
    """
    if isinstance(mult, dict):
      ms = np.ones(self.levels)
      for key, value in mult.items():
        if not isinstance(key, int): raise ValueError("Error, mult dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ms[key] = value
    else:
      ms = np.asarray(mult)
      if ms.ndim != 1: raise ValueError("Error, mult must be a dictionary or be mappable to a 1D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    if self.type in ["dwt", "swt"]:
      for level in range(min(self.levels, ms.size)):
        if (m := ms[level]) == 1.: continue
        cH, cV, cD = output.coeffs[-(level+1)]
        cH *= m
        cV *= m
        cD *= m
    elif self.type == "slt":
      for level in range(min(self.levels, ms.size)):
        if (m := ms[level]) == 1.: continue
        cA = output.coeffs[-(level+1)]
        cA *= m
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

  def threshold_levels(self, threshold, mode = "soft", substitute = 0., inplace = False):
    """
    """
    if isinstance(threshold, dict):
      ts = np.zeros(self.levels)
      for key, value in threshold.items():
        if not isinstance(key, int): raise ValueError("Error, mult dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key] = value
    else:
      ts = np.asarray(threshold)
      if ts.ndim != 1: raise ValueError("Error, threshold must be a dictionary or be mappable to a 1D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    if self.type in ["dwt", "swt"]:
      for level in range(min(self.levels, ts.size)):
        if (t := ts[level]) <= 0.: continue
        cH, cV, cD = output.coeffs[-(level+1)]
        cH[...] = pywt.threshold(cH, t, mode = mode, substitute = substitute)
        cV[...] = pywt.threshold(cV, t, mode = mode, substitute = substitute)
        cD[...] = pywt.threshold(cD, t, mode = mode, substitute = substitute)
    elif self.type == "slt":
      for level in range(min(self.levels, ts.size)):
        if (t := ts[level]) <= 0.: continue
        cA = output.coeffs[-(level+1)]
        cA[...] = pywt.threshold(cA, t, mode = mode, substitute = substitute)
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

  def threshold_firm_levels(self, thresholds, inplace = False):
    """
    """
    if isinstance(thresholds, dict):
      ts = np.zeros(self.levels)
      for key, value in thresholds.items():
        if not isinstance(key, int): raise ValueError("Error, mult dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key] = value
    else:
      ts = np.asarray(thresholds)
      if ts.ndim != 2: raise ValueError("Error, thresholds must be a dictionary or be mappable to a 2D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    if self.type in ["dwt", "swt"]:
      for level in range(min(self.levels, ts.shape[0])):
        t = ts[level]
        cH, cV, cD = output.coeffs[-(level+1)]
        cH[...] = pywt.threshold_firm(cH, t[0], t[1])
        cV[...] = pywt.threshold_firm(cV, t[0], t[1])
        cD[...] = pywt.threshold_firm(cD, t[0], t[1])
    elif self.type == "slt":
      for level in range(min(self.levels, ts.shape[0])):
        t = ts[level]
        cA = output.coeffs[-(level+1)]
        cA[...] = pywt.threshold_firm(cA, t[0], t[1])
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

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
  wt.mode = mode
  wt.norm = True
  wt.coeffs = pywt.swt2(padded, wavelet = wavelet, level = levels, start_level = start, trim_approx = True, norm = wt.norm, axes = (-2, -1))
  wt.size = (height, width)
  wt.padding = (ptop, pleft)
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

def slt(image, levels, starlet = "cubic", mode = "symmetric"):
  """
  """

  def convolve_starlet(data, step):
    """
    """
    # Set-up convolution kernel.
    if starlet == "linear":
      size = 2*step+1
      kernel = np.zeros(size, dtype = data.dtype)
      kernel[   0] = 1/4.
      kernel[step] = 1/2.
      kernel[  -1] = 1/4.
    elif starlet == "cubic":
      size = 4*step+1
      kernel = np.zeros(size, dtype = data.dtype)
      kernel[     0] = 1/16.
      kernel[  step] = 1/4.
      kernel[2*step] = 3/8.
      kernel[3*step] = 1/4.
      kernel[    -1] = 1/16.
    else:
      raise ValueError("Error, starlet must be 'linear' or 'cubic'.")
    # Convolve data with the kernel along the last two axes.
    output = data
    for axis in (-2, -1):
      output = ndimg.convolve1d(output, kernel, axis = axis, mode = mode, cval = 0.)
    return output

  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Translate pywt boundary modes.
  if mode == "zero":
    mode = "constant"
  elif mode == "constant":
    mode = "nearest"
  elif mode == "periodic":
    mode = "wrap"
  elif mode == "symmetric":
    mode = "reflect"
  elif mode == "reflect":
    mode = "mirror"
  else:
    raise ValueError(f"Error, unknown boundary mode '{mode}'.")
  # Compute the starlet transform.
  step = 1
  coeffs = []
  for level in range(levels):
    convolved = convolve_starlet(data, step)
    coeffs.append(data-convolved)
    data = convolved
    step *= 2
  coeffs.append(convolved)
  wt = WaveletTransform()
  wt.type = "slt"
  wt.wavelet = starlet
  wt.levels = levels
  wt.start = 0
  wt.mode = mode
  wt.coeffs = tuple(reversed(coeffs))
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

# def slt(image, levels, starlet = "cubic", mode = "symmetric"):
#   """
#   """
#   isImage = issubclass(type(image), img.Image)
#   if isImage:
#     width, height = image.get_size()
#     data = image.image
#   elif imgutils.is_valid_image(image):
#     width, height = image.shape[-1], image.shape[-2]
#     data = image
#   else:
#     raise ValueError("Error, the input image is not valid.")
#   # Pad the image so that the width and height are multiples of 2**level.
#   length = 2**levels
#   pwidth  = int(np.ceil(width /length))*length
#   pheight = int(np.ceil(height/length))*length
#   pleft = (pwidth-width)//2 ; pright = pwidth-width-pleft
#   ptop = (pheight-height)//2 ; pbottom = pheight-height-ptop
#   padding = (data.ndim-2)*((0, 0),)+((ptop, pbottom), (pleft, pright))
#   if mode == "zero": # Translate pywt boundary modes.
#     mode = "constant"
#   elif mode == "constant":
#     mode = "edge"
#   elif mode == "periodic":
#     mode = "wrap"
#   elif mode not in ["symmetric", "reflect"]:
#     raise ValueError(f"Error, unknown boundary mode '{mode}'.")
#   padded = np.pad(data, padding, mode = mode)
#   # Compute the starlet transform.
#   if starlet == "linear": # Only the first (low pass) filter is relevant here.
#     wavelet = pywt.Wavelet("linear", filter_bank = [[1/4, 1/2, 1/4], [-1/4, 1/2, -1/4], [0., 0., 0.], [0., 0., 0.]])
#   elif starlet == "cubic":
#     wavelet = pywt.Wavelet("cubic", filter_bank = [[1/16, 1/4, 3/8, 1/4, 1/16], [-1/16, -1/8, 3/8, -1/8, -1/16], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
#   else:
#     raise ValueError("Error, starlet must be 'linear' or 'cubic'.")
#   coeffs = pywt.swt2(padded, wavelet = wavelet, level = levels, start_level = 0, trim_approx = False, norm = False, axes = (-2, -1))
#   wt = WaveletTransform()
#   wt.type = "slt"
#   wt.wavelet = starlet
#   wt.levels = levels
#   wt.start = 0
#   wt.mode = mode
#   wt.coeffs = (coeffs[0][0][..., ptop:ptop+height, pleft:pleft+width],)
#   for level in range(levels-1):
#     delta = coeffs[level+1][0]-coeffs[level][0]
#     wt.coeffs += (delta[..., ptop:ptop+height, pleft:pleft+width],)
#   delta = image-coeffs[levels-1][0]
#   wt.coeffs += (delta[..., ptop:ptop+height, pleft:pleft+width],)
#   wt.isImage = isImage
#   if isImage:
#     wt.colorspace = image.colorspace
#     wt.colormodel = image.colormodel
#   return wt

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def dwt(self, levels, wavelet = "default", mode = "symmetric"):
    """
    """
    return dwt(self, levels, wavelet = wavelet, mode = mode)

  def swt(self, levels, wavelet = "default", mode = "symmetric", start = 0):
    """
    """
    return swt(self, levels, wavelet = wavelet, mode = mode, start = start)

  def slt(self, levels, starlet = "cubic", mode = "symmetric"):
    """
    """
    return slt(self, levels, starlet = starlet, mode = mode)

