# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2.1.0 / 2025.09.21

"""High dynamic range transformations."""

import numpy as np

from .image_utils import blend
from .image_stretch import mts
from . import image_multiscale as multiscale

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def HDRMT(self, transform = "cubic", lmin = 0, lmax = 5, alpha = 2.5, gains = 1., beta = 1., gamma = 1., niter = 1, channels = ""):
    """HDRMT. Experimental."""

    # Check inputs.
    channels = channels.strip()
    if channels == "":
      if self.colormodel == "RGB":
        channels = "RGB"
      elif self.colormodel == "gray":
        channels = "L"
      elif self.colormodel == "HSV":
        channels = "V"
      elif self.colormodel == "HSL":
        channels = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        channels = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormodel}.")
    if channels == "RGB": self.check_color_model("RGB")
    if channels not in ["RGB", "V", "L'", "L", "Ls", "Ln", "L*", "L*ab", "L*uv", "L*sh"]:
      raise ValueError("""Error, channels must be "RGB", "V", "L'", "L", "Ls", "Ln", "L*", "L*ab", "L*uv" or "L*sh".""")
    print(f"HDRMT (transform = '{transform}') on channel(s) {channels}...")
    if np.isscalar(gains): gains = np.full(lmax+1, gains)
    # HDRMT algorithm.
    image = self.image if channels == "RGB" else self.get_channel(channel = channels)
    # Normalize image.
    image -= np.min(image)
    image /= np.max(image)
    median0 = np.median(image)
    # Iterate starlet/median transforms.
    for iiter in range(niter):
      if niter > 1: print(f"Iteration {iiter+1}/{niter}.")
      # Copy image.
      original = image.copy()
      # Compute starlet/median transform.
      if transform == "median":
        wt = multiscale.mmt(image, levels = lmax+1)
      else:
        wt = multiscale.slt(image, levels = lmax+1, starlet = transform)
      # Use approximation as compression/fusion mask.
      mask = wt.coeffs[0].copy()
      # Compress the dynamic range of each level.
      alpham = alpha*np.maximum(mask**beta, 1.e-4) if beta > 0. else alpha # Compress bright more than dark areas.
      for level in range(lmin, lmax+1):
        alphag = gains[level]*alpham
        wt.coeffs[-(level+1)][0] = np.asinh(alphag*wt.coeffs[-(level+1)][0])/np.asinh(alphag)
      # Compute the inverse starlet/median transform.
      image = wt.inverse()
      # Normalize image.
      image -= np.min(image)
      image /= np.max(image)
      median = np.median(image)
      image = mts(image, mts(median, median0))
      # Blend original and compressed image using fusion mask.
      if gamma > 0.: image = blend(original, image, mask**gamma)
    return self.newImage(image) if channels == "RGB" else self.set_channel(channel = channels, data = image)
