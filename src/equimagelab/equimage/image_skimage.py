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
