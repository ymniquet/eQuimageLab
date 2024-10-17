# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image filters."""

import numpy as np
from scipy.signal import convolve2d

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  # TESTED.
  def sharpen(self):
    """Apply a sharpening (Laplacian) convolution filter to the image."""
    image = self.image()
    filtered = self.empty()
    kernel = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]], dtype = params.IMGTYPE)
    for ic in range(3):
      filtered[ic] = convolve2d(image[ic], kernel, mode = "same", boundary = "fill", fillvalue = 0.)
    return filtered
