# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.4.1 / 2025.05.30
# Doc OK.

"""Multiscale transforms.

The following symbols are imported in the equimage/equimagelab namespaces for convenience:
  "dwt", "swt", "slt", "anscombe", "inverse_anscombe".
"""

__all__ = ["dwt", "swt", "slt", "anscombe", "inverse_anscombe"]

import pywt
import numpy as np
import scipy.ndimage as ndimg
from copy import deepcopy

from . import params
from . import helpers
from . import image as img
from . import image_utils as imgutils

###############
# Statistics. #
###############

def std_centered(data, std, **kwargs):
  """Return the standard deviation of a centered data set.

  Args:
    data (numpy.ndarray): The data set (whose average must be zero).
    std (str): The method used to compute the standard deviation:

      - "variance": std_centered = sqrt(numpy.mean(data**2))
      - "median": std_centered = numpy.median(abs(data))/0.6744897501960817.
        This estimate is more robust to outliers.

    kwargs: Optional keyword arguments are passed to the numpy.mean (std = "variance") or
      numpy.median (std = "median") functions.

  Returns:
    float: The standard deviation of data.
  """
  if std == "variance":
    return np.sqrt(np.mean(data**2, **kwargs))
  elif std == "median":
    return np.median(abs(data), **kwargs)/0.6744897501960817
  else:
    raise ValueError("Error, unknown method '{std}'.")

def anscombe(data, gain = 1., average = 0., sigma = 0.):
  """Return the generalized Anscombe transform (gAt) of the input data.

  The gAt transforms the sum data = gain*P+N of a white Poisson noise P and a white Gaussian noise
  N (characterized by its average and standard deviation sigma) into an approximate white Gaussian
  noise with variance 1.

  For gain = 1, average = 0 and sigma = 0 (default), the gAt is the original Anscombe transform
  T(data) = 2*sqrt(data+3/8).

  See also:
    :func:`inverse_anscombe`

  Args:
    data (numpy.ndarray): The input data.
    gain (float, optional): The gain (default 1).
    average (float, optional): The average of the Gaussian noise (default 0).
    sigma (float, optional): The standard deviation of the Gaussian noise (default 0).

  Returns:
    numpy.ndarray: The generalized Anscombe transform of the input data.
  """
  return 2.*np.sqrt(gain*data+3.*gain**2/8.+sigma**2-gain*average)/gain

def inverse_anscombe(data, gain = 1., average = 0., sigma = 0.):
  """Return the inverse generalized Anscombe transform of the input data.

  See also:
    :func:`anscombe`

  Args:
    data (numpy.ndarray): The input data.
    gain (float, optional): The gain (default 1).
    average (float, optional): The average of the Gaussian noise (default 0).
    sigma (float, optional): The standard deviation of the Gaussian noise (default 0).

  Returns:
    numpy.ndarray: The inverse generalized Anscombe transform of the input data.
  """
  return ((gain*data/2.)**2-3.*gain**2/8.-sigma**2+gain*average)/gain

###########################
# WaveletTransform class. #
###########################

class WaveletTransform:
  """Wavelet transform class."""

  def iwt(self, asarray = False):
    """Inverse wavelet transform.

    Args:
      asarray (bool, optional): If True, return the inverse wavelet transform as a numpy.ndarray.
        If False (default), return the inverse wavelet transform as an Image object if the original
        image was an Image object, and as a numpy.ndarray otherwise.

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
    return img.Image(data, colorspace = self.colorspace, colormodel = self.colormodel) if self.isImage and not asarray else data

  def copy(self):
    """Return a (deep) copy of the object.

    Returns:
      WaveletTransform: A copy of the object.
    """
    return deepcopy(self)

  def apply_same_transform(self, image):
    """Apply the wavelet transform of the object to the input image.

    Args:
      image (Image or numpy.ndarray): The input image.

    Returns:
      WaveletTransform: The wavelet transform of the input image.
    """
    if self.type == "dwt":
      return dwt(image, levels = self.levels, wavelet = self.wavelet, mode = self.mode)
    elif self.type == "swt":
      return swt(image, levels = self.levels, wavelet = self.wavelet, mode = self.mode, start = self.start)
    elif self.type == "slt":
      return slt(image, levels = self.levels, starlet = self.wavelet, mode = self.mode)
    else:
      raise ValueError(f"Unknown wavelet transform type '{self.type}'.")

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
      ms = [0.]*self.levels
      for key, value in mult.items():
        if not isinstance(key, int):
          raise TypeError("Error, mult dictionary keys must be integers.")
        if key < 0 or key >= self.levels:
          raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ms[key] = value
    else:
      ms = np.asarray(mult)
      if ms.ndim != 1:
        raise ValueError("Error, mult must be a dictionary or be mappable to a 1D array.")
    output = self if inplace else self.copy()
    for level in range(min(self.levels, len(ms))):
      if (m := ms[level]) == 1.: continue
      output.coeffs[-(level+1)] = [m*c for c in self.coeffs[-(level+1)]]
    return output

  def threshold_levels(self, threshold, mode = "soft", inplace = False):
    """Threshold wavelet levels.

    See also:
      :meth:`WaveletTransform.threshold_firm_levels`

    Args:
      threshold (numpy.ndarray or dict): The threshold for each wavelet level. Level 0 is the
        smallest scale. Can be a 1D array [threshold for each level], a 2D array [threshold for
        each level (rows) & channel (columns)], or a dictionary of the form {level: threshold, ...}
        or of the form {level: (threshold channel #1, threshold channel #2, ...), ...} (e.g.,
        {0: 1.e-2, 1: 1.e-3}). Default threshold is 0 for all unspecified wavelet levels.
      mode (string, optional): The thresholding mode:

        - "soft" (default): Wavelet coefficients with absolute value < threshold are replaced by 0,
          while those with absolute value >= threshold are shrunk toward 0 by the value of threshold.
        - "hard": Wavelet coefficients with absolute value < threshold are replaced by 0, while
          those with absolute value >= threshold are left unchanged.
        - "garrote": Non-negative Garrote threshold (soft for small wavelet coefficients, and hard
          for large wavelet coefficients).
        - "greater": Wavelet coefficients < threshold are replaced by 0.
        - "less": Wavelet coefficients > threshold are replaced by 0.

      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """
    if isinstance(threshold, dict):
      ts = [0.]*self.levels
      for key, value in threshold.items():
        if not isinstance(key, int):
          raise TypeError("Error, threshold dictionary keys must be integers.")
        if key < 0 or key >= self.levels:
          raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key] = value
    else:
      ts = np.asarray(threshold)
      if ts.ndim > 2:
        raise ValueError("Error, threshold must be a dictionary or be mappable to a 1D or 2D array.")
    output = self if inplace else self.copy()
    for level in range(min(self.levels, len(ts))):
      t = ts[level]
      scalar = np.isscalar(t)
      if not scalar:
        if len(t) != self.nc:
          raise ValueError("Error, the length of the threshold must match the number of channels.")
        if self.nc == 1:
          t = t[0]
          scalar = True
      if scalar:
        if t == 0.: continue
        output.coeffs[-(level+1)] = [pywt.threshold(c, t, mode = mode) for c in self.coeffs[-(level+1)]]
      else:
        output.coeffs[-(level+1)] = [np.array([pywt.threshold(c[ic], t[ic], mode = mode) \
          for ic in range(self.nc)]) for c in self.coeffs[-(level+1)]]
    return output

  def threshold_firm_levels(self, thresholds, inplace = False):
    """Firm threshold of wavelet levels.

    Firm thresholding behaves the same as soft-thresholding for wavelet coefficients with absolute
    values below threshold_low, and the same as hard-thresholding for wavelet coefficients with
    absolute values above threshold_high. For intermediate values, the outcome is in between soft
    and hard thresholding.

    See also:
      :meth:`WaveletTransform.threshold`

    Args:
      thresholds (numpy.ndarray or dict): The thresholds for each wavelet level. Level 0 is the
        smallest scale. Can be a 2D array [threshold_low (column 1) and threshold_high (column 2)
        for each level (rows)], a 3D array [threshold_low and threshold_high (second axis) for each
        level (first axis) & channel (third axis)], or a dictionary of the form
        {level: (threshold_low, threshold_high), ...} or of the form {level: ((threshold_low
        channel #1, threshold_low channel #2, ...), (threshold_high channel #1, threshold_high
        channel #2, ...)), ...} (e.g. {0: (1.e-2, 5e-2), 1: (1.e-3, 5e-3)}). Default thresholds
        are (threshold_low = 0, threshold_high = 0) for all unspecified wavelet levels.
      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """
    if isinstance(thresholds, dict):
      ts = [(0., 0.)]*self.levels
      for key, value in thresholds.items():
        if not isinstance(key, int):
          raise TypeError("Error, thresholds dictionary keys must be integers.")
        if key < 0 or key >= self.levels:
          raise ValueError(f"Error, wavelet levels must be >= 0 and < {self.levels}.")
        ts[key] = value
    else:
      ts = np.asarray(thresholds)
      if ts.ndim not in [2, 3]: raise ValueError("Error, thresholds must be a dictionary or be mappable to a 2D or 3D array.")
    output = self if inplace else self.copy()
    for level in range(min(self.levels, len(ts))):
      tlow, thigh = ts[level][0], ts[level][1]
      scalartlow, scalarthigh = np.isscalar(tlow), np.isscalar(thigh)
      if not scalartlow and len(tlow) != self.nc:
        raise ValueError("Error, the length of threshold_low must match the number of channels.")
      if not scalarthigh and len(thigh) != self.nc:
        raise ValueError("Error, the length of threshold_high must match the number of channels.")
      if scalartlow != scalarthigh:
        raise ValueError("Error, threshold_low and threshold_high must both be either scalars or arrays.")
      if not scalartlow and nc == 1:
        tlow, thigh = tlow[0], thigh[0]
        scalartlow = True
      if scalartlow:
        if (tlow, thigh) == (0., 0.): continue
        output.coeffs[-(level+1)] = [pywt.threshold_firm(c, tlow, thigh) for c in self.coeffs[-(level+1)]]
      else:
        output.coeffs[-(level+1)] = [np.array([pywt.threshold_firm(c[ic], tlow[ic], thigh[ic]) \
          for ic in range(self.nc)]) for c in self.coeffs[-(level+1)]]
    return output

  def noise_scale_factors(self, std = "median", numerical = False, size = None, samples = 1):
    """Compute the standard deviation of a white gaussian noise with variance 1 at all wavelet levels.

    This method returns the partition of a white gaussian noise with variance 1 across all wavelet
    levels. It does so analytically when the distribution of the variance is known for the object
    transformation & wavelet. If not, it does so numerically by transforming random images with
    white gaussian noise and computing the standard deviation of the wavelet coefficients at all
    scales.

    Args:
      std (str, optional): The method used to compute standard deviations. Can be "variance"
        or "median" (default). See :func:`std_centered` for details.
      numerical (bool, optional): If False (default), use analytical results when known. If True,
        always compute the standard deviations numerically.
      size (tuple of int, optional): The size (height, width) of the random images used to compute
        the standard deviations numerically. If None, defaults to the object image size.
      samples (int, optional): The number of random images used to compute the standard deviations
        numerically. The standard deviations of all random images are averaged at each scale.

    Returns:
      numpy.ndarray: The standard deviation of a white gaussian noise with variance 1 at all wavelet
      levels. Level #0 is the smallest scale.
    """
    if self.type != "slt": numerical = numerical or not pywt.Wavelet(self.wavelet).orthogonal
    # Analytical noise partition.
    if not numerical:
      if self.type == "dwt":
        return np.ones(self.levels)
      elif self.type == "swt":
        return np.array([0.5**(level+1) for level in range(self.levels)])
      elif self.type == "slt" and self.levels <= 10:
        # See misc/wavelets_noise.py.
        if self.wavelet == "linear":
          return np.array([0.8003905296791061, 0.2728788936964528, 0.11977928208173409, \
                           0.057766478495319795, 0.028616328288766757, 0.014274750590965335, \
                           0.007133197029613843, 0.0035660761823069166, 0.001782972798050199, \
                           0.0008914782373390341])[:self.levels]
        elif self.wavelet == "cubic":
          return np.array([0.8907963102787584, 0.20066385102441897, 0.08550750475336993, \
                           0.0412174443743162, 0.02042496659278143, 0.010189759249213292, \
                           0.005092046680819309, 0.0025456694579151255, 0.001272790500997916, \
                           0.0006363897222337067])[:self.levels]
    # Numerical estimate of the noise partition.
    if size is None: size = self.size
    rng = np.random.default_rng(12345) # Ensure reproducibility.
    scale_factors = 0.
    for n in range(samples):
      image = rng.normal(size = (size[0], size[1]))
      wt = self.apply_same_transform(image)
      scale_factors += np.array([std_centered(wt.coeffs[-(level+1)][-1], std) for level in range(wt.levels)])
    return scale_factors/samples

  def estimate_noise0(self, std = "median", clip = None, eps = 1.e-3, maxit = 8):
    """Estimate noise as the standard deviation of the wavelet coefficients at the smallest scale.

    This method estimates the noise of the image as the standard deviation sigma0 of the (diagonal)
    wavelet coefficients at the smallest scale (level #0). If the clip kwarg is provided, it then
    excludes wavelets whose absolute coefficients are greater than clip*sigma0, and iterates until
    sigma0 is converged.

    See also:
      :meth:`WaveletTransform.estimate_noise`

    Args:
      std (str, optional): The method used to compute standard deviations. Can be "variance"
        or "median" (default). See :func:`std_centered` for details.
      clip (float, optional): If not None (default), exclude wavelets whose absolute coefficients
        are greater than clip*sigma0 and iterate until sigma0 is converged (see the eps and maxit
        kwargs).
      eps (float, optional): If clip is not None, iterate until abs(delta sigma0) < eps*sigma0, where
        delta sigma0 is the variation of sigma0 between two successive iterations. Default is 1e-3.
      maxit (int, optional): Maximum number of iterations if clip is not None. Default is 8.

    Returns:
      numpy.ndarray: The noise sigma0 in each channel.
    """
    coeffs = helpers.at_least_3D(self.coeffs[-1][-1])
    sigma = std_centered(coeffs, std, axis = (-2, -1))
    if clip is not None and maxit > 0:
      for ic in range(self.nc):
        # print(f"Channel #{ic+1}:")
        # print(f"Iteration #0: σ = {sigma[ic]:.6e}.")
        for it in range(maxit):
          oldsigma = sigma[ic]
          cset = abs(coeffs[ic]) <= clip*sigma[ic]
          sigma[ic] = std_centered(coeffs[ic][cset], std)
          # print(f"Iteration #{it+1}: σ = {sigma[ic]:.6e}.")
          if converged := abs(sigma[ic]-oldsigma) <= eps*sigma[ic]: break
        if not converged:
          print(f"Channel #{ic+1}: After {maxit} iterations, σ = {sigma[ic]:.6e} but |Δσ| = {abs(sigma[ic]-oldsigma):.6e} > {eps:.3e}σ.")
    return sigma

  def estimate_noise(self, std = "median", clip = None, eps = 1.e-3, maxit = 8, scale_factors = None):
    """Estimate noise at each wavelet level.

    This method first estimates the noise at the smallest scale as the standard deviation sigma0 of
    the (diagonal) wavelet coefficients at level #0. It then extrapolates sigma0 to all wavelet
    levels assuming the noise is white and gaussian.

    See also:
      :meth:`WaveletTransform.estimate_noise0`,
      :meth:`WaveletTransform.noise_scale_factors`

    Args:
      std (str, optional): The method used to compute standard deviations. Can be "variance"
        or "median" (default). See :func:`std_centered` for details.
      clip (float, optional): If not None (default), exclude level #0 wavelets whose absolute
        coefficients are greater than clip*sigma0 and iterate until sigma0 is converged (see the
        eps and maxit kwargs).
      eps (float, optional): If clip is not None, iterate until abs(delta sigma0) < eps*sigma0, where
        delta sigma0 is the variation of sigma0 between two successive iterations. Default is 1e-3.
      maxit (int, optional): Maximum number of iterations if clip is not None. Default is 8.
      scale_factors (numpy.ndarray): The expected standard deviation of a white gaussian noise
        with variance 1 at each wavelet level. If None (default), this method calls
        :meth:`WaveletTransform.noise_scale_factors` to compute these factors.

    Returns:
      numpy.ndarray, numpy.ndarray: The noise in each channel (columns) and wavelet level (rows),
      and the total noise in each channel.
    """
    if scale_factors is None: scale_factors = self.noise_scale_factors(std = std)
    sigma0 = self.estimate_noise0(std = std, clip = clip, eps = eps, maxit = maxit)
    norm = scale_factors[0]
    sigmas = np.array([sigma0*factor/norm for factor in scale_factors])
    return sigmas, sigma0/norm

  def VisuShrink_clip(self):
    """Return the VisuShrink clip factor.

    The VisuShrink method computes the thresholds for the wavelet coefficients from the standard
    deviation sigma of the noise in each level as threshold = clip*sigma, with clip =
    sqrt(2*log(npixels)) and npixels the number of pixels in the image.

    Note:
      Borrowed from scikit-image. See L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
      wavelet shrinkage", Biometrika 81, 425 (1994) (DOI:10.1093/biomet/81.3.425).

    See also:
      :meth:`WaveletTransform.VisuShrink`

    Returns:
      float: The VisuShrink clip factor clip = sqrt(2*log(npixels)).
    """
    return np.sqrt(2.*np.log(self.size[0]*self.size[1]))

  def VisuShrink(self, sigmas, mode = "soft", inplace = False):
    """Threshold wavelet coefficients using the VisuShrink method.

    The VisuShrink method computes the thresholds for the wavelet coefficients from the standard
    deviations sigma of the noise in each level as threshold = clip*sigma, with clip =
    sqrt(2*log(npixels)) and npixels the number of pixels in the image.

    VisuShrink produces softer images than BayesShrink (see :meth:`WaveletTransform.BayesShrink`),
    but may oversmooth and loose many details.

    Note:
      Borrowed from scikit-image. See L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by
      wavelet shrinkage", Biometrika 81, 425 (1994) (DOI:10.1093/biomet/81.3.425).

    See also:
      :meth:`WaveletTransform.VisuShrink_clip`
      :meth:`WaveletTransform.BayesShrink`

    Args:
      sigmas (numpy.ndarray): The noise in each channel (columns) and wavelet level (rows).
      mode (string, optional): The thresholding mode:

        - "soft" (default): Wavelet coefficients with absolute value < threshold are replaced by 0,
          while those with absolute value >= threshold are shrunk toward 0 by the value of threshold.
        - "hard": Wavelet coefficients with absolute value < threshold are replaced by 0, while those
          with absolute value >= threshold are left unchanged.
        - "garrote": Non-negative Garrote threshold (soft for small wavelet coefficients, and hard
          for large wavelet coefficients).

      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """
    clip = self.VisuShrink_clip()
    print(f"VisuShrink: threshold = {clip:.5f}σ.")
    return self.threshold_levels(clip*sigmas, mode = mode, inplace = inplace)

  def BayesShrink(self, sigmas, std = "median", mode = "soft", inplace = False):
    """Threshold wavelet coefficients using the BayeShrink method.

    This method computes the thresholds for the wavelet coefficients from the standard deviation
    sigma of the noise in each level as threshold = <c²>/sqrt(<c²>-sigma²), where <c²> is the
    variance of the wavelet coefficients.

    This level-dependent strategy preserves more details than the VisuShrink method (see
    :meth:`WaveletTransform.VisuShrink`).

    Note:
      Borrowed from scikit-image. See Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
      thresholding for image denoising and compression", IEEE Transactions on Image Processing 9,
      1532 (2000) (DOI:10.1109/83.862633).

    See also:
      :meth:`WaveletTransform.VisuShrink`

    Args:
      sigmas (numpy.ndarray): The noise in each channel (columns) and wavelet level (rows).
      std (str, optional): The method used to compute standard deviations. Can be "variance"
        or "median" (default). See :func:`std_centered` for details.
      mode (string, optional): The thresholding mode:

        - "soft" (default): Wavelet coefficients with absolute value < threshold are replaced by 0,
          while those with absolute value >= threshold are shrunk toward 0 by the value of threshold.
        - "hard": Wavelet coefficients with absolute value < threshold are replaced by 0, while those
          with absolute value >= threshold are left unchanged.
        - "garrote": Non-negative Garrote threshold (soft for small wavelet coefficients, and hard
          for large wavelet coefficients).

      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """

    def shrink(c, sigma):
      """Process wavelets coefficients c with noise sigma."""
      eps = np.finfo(c.dtype).eps # Floating point accuracy.
      threshold = sigma**2/np.sqrt(max(std_centered(c, std = std)**2-sigma**2, eps))
      return pywt.threshold(c, threshold, mode = mode)

    output = self if inplace else self.copy()
    for level in range(self.levels):
      if self.nc == 1:
        output.coeffs[-(level+1)] = [shrink(c, sigmas[level, 0]) for c in self.coeffs[-(level+1)]]
      else:
        output.coeffs[-(level+1)] = [np.array([shrink(c[ic], sigmas[level, ic]) \
          for ic in range(self.nc)]) for c in self.coeffs[-(level+1)]]
    return output

  def iterative_noise_reduction(self, std = "median", clip = 3., eps = 1.e-3, maxit = 8, scale_factors = None):
    """Iterative noise reduction.

    This method first estimates the noise sigma in each channel and wavelet level (using
    :meth:`WaveletTransform.estimate_noise`), then clips the wavelet coefficients whose
    absolute values are smaller than clip*sigma. It then computes the inverse wavelet transform
    I0 and the difference D0 = I-I0 with the original image I.

    It next computes the wavelet transform of D0, estimates the noise sigma_D in each channel
    and wavelet level, clips the wavelet coefficients whose absolute values are smaller than
    clip*sigma_D, calculates the inverse wavelet transform dD0, and a new image I1 = I0+dD0
    that contains the significant residual structures thus identified in D0.

    It then repeats this procedure with D1 = I-I1, D2 = I-I2... until sigma_D is converged (which
    means that no further residual structure can be indentified in Dn).

    The method returns the denoised image In and the noise Dn = I-In. Dn shall hence be (almost)
    structureless.

    Note:
      See: Image processing and data analysis: The multiscale approach, Jean-Luc Starck, Fionn Murtagh,
      and Albert Bijaoui, Cambridge University Press (1998).
      https://www.researchgate.net/publication/220688988_Image_Processing_and_Data_Analysis_The_Multiscale_Approach

    See also:
      :meth:`WaveletTransform.estimate_noise`,
      :meth:`WaveletTransform.noise_scale_factors`

    Args:
      std (str, optional): The method used to compute standard deviations. Can be "variance"
        or "median" (default). See :func:`std_centered` for details.
      clip (float, optional): Clip wavelets whose absolute coefficients are smaller than clip*sigma,
        where sigma is the estimated noise at that wavelet level. Default is 3.
      eps (float, optional): Iterate until abs(delta sigma_D) < eps*sigma_D, where delta sigma_D is
        the variation of sigma_D between two successive iterations. Default is 1e-3.
      maxit (int, optional): Maximum number of iterations. Default is 8.
      scale_factors (numpy.ndarray): The expected standard deviation of a white gaussian noise
        with variance 1 at each wavelet level. If None (default), this method calls
        :meth:`WaveletTransform.noise_scale_factors` to compute these factors.

    Returns:
      Image or numpy.ndarray: The denoised image In and the noise Dn = I-In.
    """
    if scale_factors is None: scale_factors = self.noise_scale_factors(std = std)
    original = helpers.at_least_3D(self.iwt(asarray = True))
    sigmas, sigmat = self.estimate_noise(std = std, clip = clip, eps = eps, maxit = maxit, scale_factors = scale_factors)
    denoised = helpers.at_least_3D(self.threshold_levels(clip*sigmas, mode = "hard").iwt(asarray = True))
    for ic in range(self.nc):
      print(f"Channel #{ic+1}:")
      print(f"Initial estimate: σ = {sigmat[ic]:.6e}.")
      it = 0
      while True:
        D = original[ic]-denoised[ic]
        Dwt = self.apply_same_transform(D)
        sigmaDs, sigmaDt = Dwt.estimate_noise(std = std, clip = clip, eps = eps, maxit = maxit, scale_factors = scale_factors)
        sigmaDt = sigmaDt[0]
        print(f"Iteration #{it}: σ_D = {sigmaDt:.6e}.")
        converged = it > 0 and abs(sigmaDt-oldsigmaDt) <= eps*sigmaDt
        if converged: break
        it += 1
        if it > maxit: break
        denoised[ic] += Dwt.threshold_levels(clip*sigmaDs, mode = "hard", inplace = True).iwt()
        oldsigmaDt = sigmaDt
      if converged:
        print(f"Converged in {it} iterations.")
      elif maxit > 0:
        print(f"After {maxit} iterations, σ_D = {sigmaDt:.6e} but |Δσ_D| = {abs(sigmaDt-oldsigmaDt):.6e} > {eps:.3e}σ_D.")
    diff = original-denoised
    if self.isImage:
      denoised = img.Image(denoised, colorspace = self.colorspace, colormodel = self.colormodel)
      diff = img.Image(diff, colorspace = self.colorspace, colormodel = self.colormodel)
    return denoised, diff

  def enhance_details(self, alphas, betas = 1., thresholds = 0., alphaA = 1., betaA = 1., inplace = False):
    """Enhance the detail coefficients of a starlet transformation.

    This method (only implemented for starlet transformations at present) enhances the details
    coefficients c → f(abs(c))*c of each wavelet level, with:

      - f(x) = 1 if x <= threshold,
      - f(x) = (cmax/x)*((x-c0)/(cmax-c0))**alpha if x > threshold,

    where cmax = beta*max(abs(c)) and c0 is computed to ensure continuity at x = threshold.

    With alpha < 1 this transformations enhances the detail coefficients whose absolute values are
    within [threshold, cmax], and softens detail coefficients whose absolute values are above cmax
    (dynamic range compression).

    Args:
      alphas (float): The alpha exponent for each wavelet level (expected < 1). Can be a scalar
        (same alpha for all scales) or a list/tuple/array (level #0 is the smallest scale).
        If alpha = 1, the wavelet level is not enhanced.
      betas (float, optional): The beta factor for each wavelet level (expected < 1). Can be
        a scalar (same beta for all scales) or a list/tuple/array (level #0 is the smallest scale).
      thresholds (float, optional): The threshold for each wavelet level. Can be a scalar (same
        threshold for all scales) or a list/tuple/array (level #0 is the smallest scale).
      alphaA (float, optional): The alpha exponent for the approximation coefficients (default 1 =
        not enhanced).
      betaA (float, optional): The beta factor for the approximation coefficients (default 1).
      inplace (bool, optional): If True, update the object "in place"; if False (default), return a
        new WaveletTransform object.

    Returns:
      WaveletTransform: The updated WaveletTransform object.
    """

    def enhance(c, cmin, cmax, alpha):
      """Enhance the input coefficients c."""
      r = (cmin/cmax)**(1./alpha)
      c0 = (cmax*r-cmin)/(r-1.)
      cout = np.empty_like(c)
      cset = c <= cmin
      cout[ cset] = c[cset]
      cout[~cset] = cmax*((c[~cset]-c0)/(cmax-c0))**alpha
      return cout

    if self.type != "slt": raise NotImplementedError("Error, only implemented for starlet transforms.")
    if np.isscalar(alphas): alphas = [alphas]*self.levels
    if np.isscalar(betas): betas = [betas]*self.levels
    if np.isscalar(thresholds): thresholds = [thresholds]*self.levels
    output = self if inplace else self.copy()
    for level in range(self.levels):
      alpha = alphas[level]
      if alpha == 1.: continue
      beta = betas[level]
      c = self.coeffs[-(level+1)][0]
      absc = abs(c)
      cmin = thresholds[level]
      cmax = beta*absc.max()
      if cmax <= cmin: raise ValueError(f"Error, threshold > cmax at level #{level}. Decrease threshold or increase beta.")
      output.coeffs[-(level+1)][0] = np.sign(c)*enhance(absc, cmin, cmax, alpha)
    if alphaA != 1.:
      c = self.coeffs[0][0]
      output.coeffs[0][0] = enhance(c, 0., betaA*c.max(), alphaA)
    return output

#######################
# Wavelet transforms. #
#######################

def dwt(image, levels, wavelet = "default", mode = "reflect"):
  """Discrete wavelet transform of an image.

  Args:
    image (Image or numpy.ndarray): The input image.
    levels (int): The number of wavelet levels.
    wavelet (string or pywt.Wavelet object, optional): The wavelet used for the transformation.
      Default is "default", a shortcut for `equimage.params.defwavelet`.
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
      - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
      - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The discrete wavelet transform of the input image.
  """
  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  height, width = data.shape[-2], data.shape[-1]
  maxlevels = pywt.dwtn_max_level((height, width), wavelet)
  if levels < 1: raise ValueError("Error, levels must be >= 1.")
  if levels > maxlevels: raise ValueError(f"Error, levels must be <= {maxlevels} for an image with size ({height}, {width}).")
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
  wt.size = (height, width)
  wt.nc = 1 if data.ndim == 2 else data.shape[0]
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
      Default is "default", a shortcut for `equimage.params.defwavelet`.
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "mirror": the image is reflected about the center of the last pixel (abcd → dcb|abcd|cba).
      - "nearest": the image is padded with the value of the last pixel (abcd → aaaa|abcd|dddd).
      - "zero": the image is padded with zeros (abcd → 0000|abcd|0000).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The stationary wavelet transform of the input image.
  """
  isImage = issubclass(type(image), img.Image)
  if isImage:
    data = image.image
  elif imgutils.is_valid_image(image):
    data = image
  else:
    raise ValueError("Error, the input image is not valid.")
  height, width = data.shape[-2], data.shape[-1]
  maxlevels = pywt.dwtn_max_level((height, width), wavelet)
  if levels < 1: raise ValueError("Error, levels must be >= 1.")
  if levels > maxlevels: raise ValueError(f"Error, levels must be <= {maxlevels} for an image with size ({height}, {width}).")
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
  wt.nc = 1 if data.ndim == 2 else data.shape[0]
  wt.padding = (ptop, pleft)
  wt.isImage = isImage
  if isImage:
    wt.colorspace = image.colorspace
    wt.colormodel = image.colormodel
  return wt

def slt(image, levels, starlet = "cubic", mode = "reflect"):
  """Starlet (isotropic undecimated wavelet) transform of the input image.

  Note:
    See: Image processing and data analysis: The multiscale approach, Jean-Luc Starck, Fionn Murtagh,
    and Albert Bijaoui, Cambridge University Press (1998).
    https://www.researchgate.net/publication/220688988_Image_Processing_and_Data_Analysis_The_Multiscale_Approach

  Args:
    image (Image or numpy.ndarray): The input image.
    levels (int): The number of starlet levels.
    starlet (string, optional): The starlet used for the transformation ("linear" for the 3x3
      linear spline or "cubic" for the 5x5 cubic spline). Default is "cubic".
    mode (str, optional): How to extend the image across its boundaries:

      - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
      - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

  Returns:
    WaveletTransform: The starlet transform of the input image.
  """

  def convolve_starlet(data, step):
    """Convolve the input data with the starlet at scale step/2-1."""
    # Set-up convolution kernel.
    if starlet == "linear":
      kernel_size = 2*step+1
      kernel = np.zeros(kernel_size, dtype = data.dtype)
      kernel[   0] = 1/4.
      kernel[step] = 1/2.
      kernel[  -1] = 1/4.
    elif starlet == "cubic":
      kernel_size = 4*step+1
      kernel = np.zeros(kernel_size, dtype = data.dtype)
      kernel[     0] = 1/16.
      kernel[  step] = 1/4.
      kernel[2*step] = 3/8.
      kernel[3*step] = 1/4.
      kernel[    -1] = 1/16.
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
  height, width = data.shape[-2], data.shape[-1]
  if starlet == "linear":
    kernel_size = 3
  elif starlet == "cubic":
    kernel_size = 5
  else:
    raise ValueError("Error, starlet must be 'linear' or 'cubic'.")
  maxlevels = pywt.dwt_max_level(min(height, width), kernel_size)
  if levels < 1: raise ValueError("Error, levels must be >= 1.")
  if levels > maxlevels: raise ValueError(f"Error, levels must be <= {maxlevels} for an image with size ({height}, {width}).")
  # Check boundary mode.
  # These are the only modes that ensure that the average of the wavelet coefficients is zero at all scales.
  if mode not in ["reflect", "wrap"]: raise ValueError(f"Error, unknown boundary mode '{mode}'.")
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
  wt.size = (height, width)
  wt.nc = 1 if data.ndim == 2 else data.shape[0]
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
        Default is "default", a shortcut for `equimage.params.defwavelet`.
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
        Default is "default", a shortcut for `equimage.params.defwavelet`.
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

    Note:
      See: Image processing and data analysis: The multiscale approach, Jean-Luc Starck, Fionn Murtagh,
      and Albert Bijaoui, Cambridge University Press (1998).
      https://www.researchgate.net/publication/220688988_Image_Processing_and_Data_Analysis_The_Multiscale_Approach

    Args:
      levels (int): The number of starlet levels.
      starlet (string, optional): The starlet used for the transformation ("linear" for the 3x3
        linear spline or "cubic" for the 5x5 cubic spline). Default is "cubic".
      mode (str, optional): How to extend the image across its boundaries:

        - "reflect" (default): the image is reflected about the edge of the last pixel (abcd → dcba|abcd|dcba).
        - "wrap": the image is periodized (abcd → abcd|abcd|abcd).

    Returns:
      WaveletTransform: The starlet transform of the image.
    """
    return slt(self, levels, starlet = starlet, mode = mode)

  def anscombe(self, gain = 1., average = 0., sigma = 0.):
    """Return the generalized Anscombe transform (gAt) of the image.

    The gAt transforms the sum gain*P+N of a white Poisson noise P and a white Gaussian noise N
    (characterized by its average and standard deviation sigma) into an approximate white Gaussian
    noise with variance 1.

    For gain = 1, average = 0 and sigma = 0 (default), the gAt is the original Anscombe transform
    T(image) = 2*sqrt(image+3/8).

    See also:
      :meth:`Image.inverse_anscombe <.inverse_anscombe>`

    Args:
      gain (float, optional): The gain (default 1).
      average (float, optional): The average of the Gaussian noise (default 0).
      sigma (float, optional): The standard deviation of the Gaussian noise (default 0).

    Returns:
      Image: The generalized Anscombe transform of the image.
    """
    return anscombe(self, gain = gain, average = average, sigma = sigma)

  def inverse_anscombe(self, gain = 1., average = 0., sigma = 0.):
    """Return the inverse generalized Anscombe transform of the image.

    See also:
      :meth:`Image.anscombe <.anscombe>`

    Args:
      gain (float, optional): The gain (default 1).
      average (float, optional): The average of the Gaussian noise (default 0).
      sigma (float, optional): The standard deviation of the Gaussian noise (default 0).

    Returns:
      Image: The inverse generalized Anscombe transform of the image.
    """
    return inverse_anscombe(self, gain = gain, average = average, sigma = sigma)
