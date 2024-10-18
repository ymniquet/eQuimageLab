# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Interface with scikit-image."""

import numpy as np
import skimage as skim

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ############
  # Filters. #
  ############

  # TESTED.
  def gaussian_filter(self, sigma, mode = "reflect"):
    """Convolve the image with a gaussian of standard deviation 'sigma' (pixels).
       The image is extended across its boundaries according to the boundary 'mode':
         - Reflect: the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
         - Mirror: the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
         - Nearest: the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
         - Zero: the image is padded with zeros (abcd -> 0000|abcd|0000)."""
    if mode == "zero": # Translate modes.
      mode = "constant"
    output = skim.filters.gaussian(self, channel_axis = 0, sigma = sigma, mode = mode, cval = 0.)
    return self.newImage_like(self, output)

  # TESTED.
  def bilateral_filter(self, sigma_space, sigma_color = .1, mode = "reflect"):
    """Bilateral filter.
       Convolve the image IMG with a gaussian gs of standard deviation 'sigma_space' weighted by
       a gaussian gc in color space (with standard deviation 'sigma_color'):

         OUT(r) = Sum_{r'} IMG(r') x gs(|r-r'|) x gc(|IMG(r)-IMG(r')|)

       The image is extended across its boundaries according to the boundary 'mode':
         - Reflect: the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
         - Mirror: the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
         - Nearest: the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
         - Zero: the image is padded with zeros (abcd -> 0000|abcd|0000)."""
    if mode == "mirror": # Translate modes.
      mode = "symmetric"
    elif mode == "nearest":
      mode = "edge"
    elif mode == "zero":
      mode = "constant"
    output = skim.restoration.denoise_bilateral(self, channel_axis = 0, sigma_spatial = sigma_space,
                                                sigma_color = sigma_color, mode = mode, cval = 0.)
    return self.newImage_like(self, output)

  # TESTED.
  def butterworth_filter(self, cutoff, order = 2, padding = 0):
    """Butterworth low-pass filter in the frequency domain:

         H(f) = 1/(1+(f/fc^(2n))

       where n = 'order' is the order of the filter and fc = (1-c)fs/2 is the cut-off frequency, with fs the
       sampling frequency and c = 'cutoff' in [0, 1] the normalized cut-off frequency.
       If the filter leaves visible artifacts on the edges, the image may be padded with 'padding' pixels (set to
       the edge values) prior to Fourier transform."""
    output = skim.filters.butterworth(self, channel_axis = 0, cutoff_frequency_ratio = (1.-cutoff)/2., order = order,
                                      npad = padding, squared_butterworth = True)
    return self.newImage_like(self, output)

  ##############
  # Denoising. #
  ##############

  # TESTED.
  def estimate_noise(self, average = True):
    """Estimate the rms noise of the image (in each channel, or on average if 'average' is True)."""
    return skim.restoration.estimate_sigma(self, channel_axis = 0, average_sigmas = average)

  # TESTED.
  def non_local_means(self, size = 7, dist = 11, h = .01, sigma = 0., fast = True):
    """Non-local means filter for denoising:

         OUT(r) ~ Sum_{r'} f(r, r') x IMG(r')

       where:

         f(r, r') = exp[-(M(r)-M(r'))²/h²]

      and M(r) is an average of the pixels in a patch around r. The filter is controlled by:
        - The size 'size' of the (square) patch used to compute M(r). The pixels within the patch are uniformly
          averaged if 'fast' is true, weighted by a gaussian if not (better yet slower).
        - The maximal distance 'dist' between the patches.
        - The cut-off 'h' in gray levels (the filter is applied to all channels independently).
      The standard deviation 'sigma' of the noise may be provided and subtracted out when computing f(r, r').
      This can lead to a modest improvement in image quality.
      The non-local means filter can restore textures that would be blurred by other denoising algorithms."""
    output = skim.restoration.denoise_nl_means(self, channel_axis = 0, patch_size = size, patch_distance = dist,
                                               h = h, sigma = sigma, fast_mode = fast)
    return self.newImage_like(self, output)

  # TESTED.
  def total_variation(self, weight = .1, algorithm = "Chambolle"):
    """Total variation denoising.
       Given a noisy image f, find an image u with less total variation than f under the constraint that u
       remains similar to f. This can be expressed as the Rudin–Osher–Fatemi (ROF) minimization problem:

         minmize Sum_{r} |grad u(r)|+(lambda/2)[f(r)-u(r)]²

       where 'weight' = 1/lambda controls denoising (the larger the weight, the stronger the denoising
       at the expense of image fidelity).
       The minimization can either be performed with the Chambolle ('algorithm' = "Chambolle") or
       Split Bregman algorithms ('algorithm' = "Bregman", respectively).
       Total variation denoising tends to produce cartoon-like (piecewise-constant) images."""
    if algorithm == "Chambolle":
      output = skim.restoration.denoise_tv_chambolle(self, channel_axis = 0, weight = weight)
    elif algorithm == "Bregman":
      output = skim.restoration.denoise_tv_bregman(self, channel_axis = 0, weight = 1./(2.*weight))
    else:
      raise ValueError(f"Error, unknown algorithm {algorithm} (must be 'Chambolle' or 'Bregman').")
    return self.newImage_like(self, output)
