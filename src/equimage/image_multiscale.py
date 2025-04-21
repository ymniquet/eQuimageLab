# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.1 / 2025.03.26
# Doc OK.

"""Multiscale transforms.

The following symbols are imported in the equimage/equimagelab namespaces for convenience:
  "dwt", "swt", "slt".
"""

__all__ = ["dwt", "swt", "slt"]

import pywt
import numpy as np
import scipy.ndimage as ndimg
from copy import deepcopy

from . import params
from . import helpers
from . import image as img
from . import image_utils as imgutils

class WaveletTransform:
  """Wavelet transform class."""

  def iwt(self):
    """Inverse wavelet transform.

    Returns:
      Image or numpy.ndarray: The inverse wavelet transform of the object.
    """
    if self.type == "dwt":
      data = pywt.waverec2(self.coeffs, wavelet = self.wavelet, mode = self._mode, axes = (-2, -1))
    elif self.type == "swt":
      data = pywt.iswt2(self.coeffs, wavelet = self.wavelet, norm = self.norm, axes = (-2, -1))
      height, width = self.size
      ptop, pleft = self.padding
      data = data[..., ptop:ptop+height, pleft:pleft+width]
    elif self.type == "slt":
      data = self.coeffs[0]+np.sum(self.coeffs[1:], axis = 0)[0]
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return img.Image(data, colorspace = self.colorspace, colormodel = self.colormodel) if self.isImage else data

  def scale_levels(self, mult, inplace = False):
    """Scale wavelet levels.

    Args:
      mult (numpy.ndarray or dict): The scaling factor for each wavelet level. Level 0 is the smallest
        scale. If a dictionary, must be of the form {level: scaling factor, ...} (e.g. {0: .8, 1: 1.5}).
        Default scaling factor is 1 for all unspecified wavelet levels.
      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
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
    if self.type in ["dwt", "swt", "slt"]:
      for level in range(min(self.levels, ms.size)):
        if (m := ms[level]) == 1.: continue
        output.coeffs[-(level+1)] = [m*c for c in output.coeffs[-(level+1)]]
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

  def threshold_levels(self, threshold, mode = "soft", substitute = 0., inplace = False):
    """Threshold wavelet levels.

    See also:
      :py:meth:`threshold_firm_levels <WaveletTransform.threshold_firm_levels>`

    Args:
      threshold (numpy.ndarray or dict): The threshold for each wavelet level. Level 0 is the smallest
        scale. If a dictionary, must be of the form {level: threshold, ...} (e.g. {0: 1.e-2, 1: 1.e-3}).
        Default threshold is 0 for all unspecified wavelet levels.
      mode (string, optional): The thresholding mode:

        - "soft" (default): Wavelet coefficients with absolute value < threshold are replaced by
          substitute, while those with absolute value >= threshold are shrunk toward zero by
          threshold.
        - "hard": Wavelet coefficients with absolute value < threshold are replaced by substitute,
          while those with absolute value >= threshold are left unchanged.
        - "garrote": Non-negative Garotte threshold (soft for small wavelet coefficients, and hard
          for large wavelet coefficients).
        - "greater": Wavelet coefficients < threshold are replaced by substitute.
        - "less": Wavelet coefficients > threshold are replaced by substitute.

      substitute (float, optional): The substitute value for thresholded wavelet coefficients
        (default 0).
      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """
    if isinstance(threshold, dict):
      ts = np.zeros(self.levels)
      for key, value in threshold.items():
        if not isinstance(key, int): raise ValueError("Error, threshold dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key] = value
    else:
      ts = np.asarray(threshold)
      if ts.ndim != 1: raise ValueError("Error, threshold must be a dictionary or be mappable to a 1D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    if self.type in ["dwt", "swt", "slt"]:
      for level in range(min(self.levels, ts.size)):
        t = ts[level]
        output.coeffs[-(level+1)] = [pywt.threshold(c, t, mode = mode, substitute = substitute) for c in output.coeffs[-(level+1)]]
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

  def threshold_firm_levels(self, thresholds, inplace = False):
    """Firm threshold of wavelet levels.

    Firm threshold behaves the same as soft-thresholding for wavelet coefficients below threshold_low
    and the same as hard-thresholding for wavelet coefficients above threshold_high. For intermediate
    values, the outcome is in between that of soft and hard thresholding.

    See also:
      :py:meth:`threshold <WaveletTransform.threshold>`

    Args:
      thresholds (numpy.ndarray or dict): The thresholds for each wavelet level. Level 0 is the smallest
        scale. If a dictionary, must be of the form {level: (threshold_low, threshold_high), ...}
        (e.g. {0: (1.e-2, 5e-2), 1: (1.e-3, 5e-3)}). If an array, the first column is threshold_low
        and the second column is threshold_high. Default thresholds are (0, 0) for all unspecified
        wavelet levels.
      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """
    if isinstance(thresholds, dict):
      ts = np.zeros((self.levels, 2))
      for key, value in thresholds.items():
        if not isinstance(key, int): raise ValueError("Error, thresholds dictionary keys must be integers.")
        if key < 0 or key >= self.levels: raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key, :] = value[0], value[1]
    else:
      ts = np.asarray(thresholds)
      if ts.ndim != 2: raise ValueError("Error, thresholds must be a dictionary of tuples or be mappable to a 2D array.")
    if inplace:
      output = self
    else:
      output = deepcopy(self)
    if self.type in ["dwt", "swt", "slt"]:
      for level in range(min(self.levels, ts.shape[0])):
        t = ts[level]
        output.coeffs[-(level+1)] = [pywt.threshold_firm(c, t[0], t[1]) for c in output.coeffs[-(level+1)]]
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
    return output

  def noise_scale_factors(self, numerical = False, size = None, repeat = 1):
    if not numerical:
      if self.type == "dwt":
        return np.ones((self.levels, 3))
      elif self.type == "swt":
        return np.array([[0.5**(level+1)]*3 for level in range(self.levels)])
    if size is None: size = self.coeffs[0].shape[-2:]
    rng = np.random.default_rng(12345) # Ensure reproducibility.
    scale_factors = 0.
    for n in range(repeat):
      image = rng.normal(size = (1, size[-2], size[-1]))
      if self.type == "dwt":
        wt = dwt(image, levels = self.levels, wavelet = self.wavelet, mode = self.mode)
      elif self.type == "swt":
        wt = swt(image, levels = self.levels, wavelet = self.wavelet, mode = self.mode, start = self.start)
      elif self.type == "slt":
        wt = slt(image, levels = self.levels, starlet = self.wavelet, mode = self.mode)
      else:
        raise ValueError(f"Unknown wavelet transform type '{self.type}'.")
      scale_factors += np.array([[np.median(abs(c))/0.6744897501960817 for c in wt.coeffs[-(level+1)]] for level in range(wt.levels)])
    return scale_factors/repeat

  def estimate_noise0(self, clip = None):
    abscoeffs = helpers.at_least_3D(np.abs(self.coeffs[-1][-1]))
    sigma = np.median(abscoeffs, axis = (-2, -1))/0.6744897501960817
    if clip is not None:
      for ic in range(abscoeffs.shape[0]):
        oldset = np.zeros_like(abscoeffs[ic], dtype = bool)
        while True:
          newset = abscoeffs[ic] < clip*sigma[ic]
          if np.all(newset == oldset): break
          sigma[ic] = np.median(abscoeffs[ic][newset])/0.6744897501960817
          oldset = newset
    return sigma

  def estimate_noise(self, scale_factors = None, clip = None):
    if scale_factors is None: scale_factors = self.noise_scale_factors()
    sigma0 = self.estimate_noise0(clip = clip)
    norm = scale_factors[0][-1]
    sigmas = np.array([[sigma0*f/norm for f in s] for s in scale_factors])
    return sigmas, sigma0/norm

def dwt(image, levels, wavelet = "default", mode = "reflect"):
  """Discrete wavelet transform of an image.

  Args:
    image (Image or numpy.ndarray): The input image.
    levels (int): The number of wavelet levels.
    wavelet (string or pywt.Wavelet object, optional): The wavelet used for the transformation.
      Default is "default" for `equimage.params.defwavelet`.
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
      - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
      - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The discrete wavelet transform of the input image.
  """
  if levels < 1: raise ValueError("Error, levels must be > 1.")
  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Translate boundary mode for pywt.
  if mode == "reflect":
    _mode = "symmetric"
  elif mode == "mirror":
    _mode = "reflect"
  elif mode == "nearest":
    _mode = "constant"
  elif mode == "wrap":
    _mode = "periodic"
  elif mode == "zero":
    _mode = "zero"
  else:
    raise ValueError(f"Error, unknown boundary mode '{mode}'.")
  # Compute the discrete wavelet transform.
  if wavelet == "default": wavelet = params.defwavelet
  wt = WaveletTransform()
  wt.type = "dwt"
  wt.wavelet = wavelet
  wt.levels = levels
  wt.start = 0
  wt.mode = mode
  wt._mode = _mode
  wt.coeffs = pywt.wavedec2(data, wavelet = wavelet, level = levels, mode = _mode, axes = (-2, -1))
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

def swt(image, levels, wavelet = "default", mode = "reflect", start = 0):
  """Stationary wavelet transform (also known as undecimated or "à trous" transform) of an image.

  Args:
    image (Image or numpy.ndarray): The input image.
    levels (int): The number of wavelet levels.
    wavelet (string or pywt.Wavelet object, optional): The wavelet used for the transformation.
      Default is "default" for `equimage.params.defwavelet`.
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
      - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
      - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The stationary wavelet transform of the input image.
  """
  if levels < 1: raise ValueError("Error, levels must be > 1.")
  isImage = issubclass(type(image), img.Image)
  if isImage:
    width, height = image.get_size()
    data = image.image
  elif imgutils.is_valid_image(image):
    width, height = image.shape[-1], image.shape[-2]
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Translate boundary mode.
  if mode == "zero":
    _mode = "constant"
  elif mode == "nearest":
    _mode = "edge"
  elif mode in ["reflect", "mirror", "wrap"]:
    _mode = mode
  else:
    raise ValueError(f"Error, unknown boundary mode '{mode}'.")
  # Pad the image so that the width and height are multiples of 2**level.
  length = 2**levels
  pwidth  = int(np.ceil(width /length))*length
  pheight = int(np.ceil(height/length))*length
  pleft = (pwidth-width)//2 ; pright = pwidth-width-pleft
  ptop = (pheight-height)//2 ; pbottom = pheight-height-ptop
  padding = (data.ndim-2)*((0, 0),)+((ptop, pbottom), (pleft, pright))
  padded = np.pad(data, padding, mode = _mode)
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

def slt(image, levels, starlet = "cubic", mode = "reflect"):
  """Starlet (isotropic undecimated wavelet) transform of the input image.

  Args:
    image (Image or numpy.ndarray): The input image.
    levels (int): The number of starlet levels.
    starlet (string, optional): The starlet used for the transformation ("linear" for the 3x3
      linear spline or "cubic" for the 5x5 cubic spline). Default is "cubic".
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
      - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
      - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The starlet transform of the input image.
  """

  def convolve_starlet(data, step):
    """Convolve the input data with the starlet at scale step/2-1."""
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
      output = ndimg.convolve1d(output, kernel, axis = axis, mode = _mode, cval = 0.)
    return output

  if levels < 1: raise ValueError("Error, levels must be > 1.")
  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  # Translate boundary mode.
  if mode == "zero":
    _mode = "constant"
  elif mode in ["reflect", "mirror", "nearest", "wrap"]:
    _mode = mode
  else:
    raise ValueError(f"Error, unknown boundary mode '{mode}'.")
  # Compute the starlet transform.
  step = 1
  coeffs = []
  for level in range(levels):
    convolved = convolve_starlet(data, step)
    coeffs.append([data-convolved])
    data = convolved
    step *= 2
  coeffs.append(convolved)
  wt = WaveletTransform()
  wt.type = "slt"
  wt.wavelet = starlet
  wt.levels = levels
  wt.start = 0
  wt.mode = mode
  wt.coeffs = list(reversed(coeffs))
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

  def dwt(self, levels, wavelet = "default", mode = "reflect"):
    """Discrete wavelet transform of the image.

    Args:
      levels (int): The number of wavelet levels.
      wavelet (string or pywt.Wavelet object, optional): The wavelet used for the transformation.
        Default is "default" for `equimage.params.defwavelet`.
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
        - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

    Returns:
      WaveletTransform: The discrete wavelet transform of the image.
    """
    return dwt(self, levels, wavelet = wavelet, mode = mode)

  def swt(self, levels, wavelet = "default", mode = "reflect", start = 0):
    """Stationary wavelet transform (also known as undecimated or "à trous" transform) of the image.

    Args:
      levels (int): The number of wavelet levels.
      wavelet (string or pywt.Wavelet object, optional): The wavelet used for the transformation.
        Default is "default" for `equimage.params.defwavelet`.
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
        - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

    Returns:
      WaveletTransform: The stationary wavelet transform of the image.
    """
    return swt(self, levels, wavelet = wavelet, mode = mode, start = start)

  def slt(self, levels, starlet = "cubic", mode = "reflect"):
    """Starlet (isotropic undecimated wavelet) transform of the image.

    Args:
      levels (int): The number of starlet levels.
      starlet (string, optional): The starlet used for the transformation ("linear" for the 3x3
        linear spline or "cubic" for the 5x5 cubic spline). Default is "cubic".
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
        - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
        - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
        - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
        - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

    Returns:
      WaveletTransform: The starlet transform of the image.
    """
    return slt(self, levels, starlet = starlet, mode = mode)
