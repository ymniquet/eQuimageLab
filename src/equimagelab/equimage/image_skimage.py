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
  def gaussian(self, sigma, mode = "reflect"):
    """Convolve the image with a gaussian of standard deviation sigma (pixels).
       The image is extended across its boundaries according to the boundary mode:
         - Reflect: the image is reflected about the edge of the last pixel (abcd -> dcba|abcd|dcba).
         - Mirror: the image is reflected about the center of the last pixel (abcd -> dcb|abcd|cba).
         - Nearest: the image is padded with the value of the last pixel (abcd -> aaaa|abcd|dddd).
         - Zero: the image is padded with zeros (abcd -> 0000|abcd|0000)."""    
    if mode == "zero": # Translate modes.
      mode = "constant" 
    image = self.image(cls = np.ndarray)
    filtered = skim.filters.gaussian(image, channel_axis = 0, sigma = sigma, mode = mode, cval = 0.)
    return self.newImage_like(self, filtered)

  # TESTED.
  def bilateral(self, sigmaspace, sigmacolor = .1, mode = "reflect"):
    """Bilateral filter.
       Convolve the image IMG with a gaussian gs of standard deviation sigmaspace weighted by
       a gaussian gc in color space (with standard deviation sigmacolor):

         OUT(r) = Sum_{r'} IMG(r') x gs(|r-r'|) x gc(|IMG(r)-IMG(r')|)

       The image is extended across its boundaries according to the boundary mode:
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
    image = self.image(cls = np.ndarray)      
    filtered = skim.restoration.denoise_bilateral(image, channel_axis = 0, sigma_spatial = sigmaspace, sigma_color = sigmacolor, mode = mode, cval = 0.)
    return self.newImage_like(self, filtered)

  # TESTED.
  def total_variation(self, weight = .1, algorithm = "Chambolle"):
    """Total variation denoising.
       Given a noisy image f, find an image u with less total variation than f under the constraint that u  
       remains similar to f. This can be expressed as the Rudin–Osher–Fatemi (ROF) minimization problem:

         minmize Sum_{r} |grad u(r)|+(lambda/2)[f(r)-u(r)]²

       where the weight 1/lambda controls denoising (the larger the weight, the stronger the denoising at the 
       expense of image fidelity). The minimization can either be performed with the Chambolle or Split Bregman 
       algorithms (algorithm = "Chambolle" or "Bregman", respectively). 
       Total variation denoising tends to produce cartoon-like (piecewise-constant) images."""
    image = self.image(cls = np.ndarray)      
    if algorithm == "Chambolle":
      filtered = skim.restoration.denoise_tv_chambolle(image, channel_axis = 0, weight = weight)
    elif algorithm == "Bregman":
      filtered = skim.restoration.denoise_tv_bregman(image, channel_axis = 0, weight = 1./(2.*weight))
    else:
      raise ValueError(f"Error, unknown algorithm {algorithm} (must be 'Chambolle' or 'Bregman').")
    return self.newImage_like(self, filtered)
