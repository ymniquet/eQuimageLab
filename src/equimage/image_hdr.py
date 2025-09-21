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

  def HDRwt1(self, starlet = "cubic", lmin = 1, lmax = 5, rstrength = .1, mstrength = 2., target = "bright", niter = 1, channels = "", maskchannel = ""):
    """HDRWT v1. Experimental."""
    # Check inputs.
    if target not in ["bright", "dark"]: raise ValueError("Error, target must be 'bright' or 'dark'.")
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
    if channels not in ["RGB", "V", "L'", "L", "Ls", "Ln", "L*", "L*ab", "L*uv", "L*sh"]:
      raise ValueError("""Error, channels must be "RGB", "V", "L'", "L", "Ls", "Ln", "L*", "L*ab", "L*uv" or "L*sh".""")
    if channels == "RGB": self.check_color_model("RGB")
    maskchannel = maskchannel.strip()
    if maskchannel == "":
      if self.colormodel in ["RGB", "gray"]:
        maskchannel = "L"
      elif self.colormodel == "HSV":
        maskchannel = "V"
      elif self.colormodel == "HSL":
        maskchannel = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        maskchannel = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormodel}.")
    if maskchannel not in ["V", "L'", "L", "L*"]:
      raise ValueError("""Error, maskchannel must be "V", "L'", "L" or "L*".""")
    print(f"HDRWT on channel(s) {channels} with mask channel {maskchannel}...")
    # HDRWT algorithm.
    image = self.copy()
    # median0 = np.median(image)
    for iiter in range(niter): # HDRWT iterations.
      if niter > 1: print(f"Iteration {iiter+1}/{niter}.")
      for level in range(lmin, lmax+1): # Iterate over wavelet levels.
        print(f"Processing level #{level}: ", end = "")
        # Compute the approximation of the mask channel at that wavelet level.
        lightness = image.get_channel(channel = maskchannel)
        approx = multiscale.slt(lightness, levels = level, starlet = starlet).coeffs[0] if level > 0 else mc
        # Compute the midtone transformation and HDR fusion mask.
        median = np.median(approx)
        if target == "bright":
          mask = np.clip(   approx, 0., 1.)**mstrength
          midtone = max(mts(median, (1.-rstrength)*median), .5)
        else:
          mask = np.clip(1.-approx, 0., 1.)**mstrength
          midtone = min(mts(median, (1.+rstrength)*median), .5)
        print(f"midtone = {midtone:.5f}.")
        # Apply the midtone transformation and blend with the original image.
        stretched = image.midtone_stretch(channels = channels, midtone = midtone, trans = False)
        image = image.blend(stretched, mask)
        # image = image.midtone_stretch(midtone = mts(np.median(image), median0), channels = "L", trans = False)
    return image

  def HDRwt2(self, starlet = "cubic", lmin = 0, lmax = 5, alpha = .9, gamma = 1., niter = 1, channels = ""):
    """HDRWT v2. Experimental."""
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
    print(f"HDRWT on channel(s) {channels}...")
    # HDRWT algorithm.
    image = self.image if channels == "RGB" else self.get_channel(channel = channels)
    for iiter in range(niter): # HDRWT iterations.
      if niter > 1: print(f"Iteration {iiter+1}/{niter}.")
      # Compute wavelet transform.
      wt = multiscale.slt(image, levels = lmax+1, starlet = starlet)
      # Compress the dynamic range of each wavelet level.
      cwt = wt.enhance_details(alphas = alpha, alphaA = alpha)
      # Blend the compressed wavelet levels with the original ones,
      # using the approximation at the next scale as fusion mask.
      approx = wt.coeffs[0].copy()
      wt.coeffs[0] = cwt.coeffs[0]
      for level in range(lmax, lmin-1, -1):
        c = wt.coeffs[-(level+1)][0].copy()
        wt.coeffs[-(level+1)][0] = blend(c, cwt.coeffs[-(level+1)][0], approx**gamma)
        approx += c
      # Compute the inverse wavelet transform and renormalize.
      image = wt.inverse()
      image -= np.min(image)
      image /= np.max(image)
    return self.newImage(image) if channels == "RGB" else self.set_channel(channel = channels, data = image)
