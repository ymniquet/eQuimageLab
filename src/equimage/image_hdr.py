# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2.0.0 / 2025.07.13

"""High dynamic range transformations."""

import numpy as np

from .image_stretch import mts
from . import image_multiscale as multiscale

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def HDRwt1(self, starlet = "cubic", lmin = 1, lmax = 5, rstrength = .1, mstrength = 2., target = "bright", niter = 1, channels = "", lightchannel = ""):
    """HDRWT."""
    # Check inputs.
    if target not in ["bright", "dark"]: raise ValueError("Error, target must be 'bright' or 'dark'.")
    channels = channels.strip()
    lightchannel = lightchannel.strip()
    if channels == "":
      if self.colormodel in ["RGB", "gray"]:
        channels = "Ls" if target == "dark" else "L"
      elif self.colormodel == "HSV":
        channels = "V"
      elif self.colormodel == "HSL":
        channels = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        channels = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormdel}.")
    if lightchannel == "":
      if self.colormodel in ["RGB", "gray"]:
        lightchannel = "L"
      elif self.colormodel == "HSV":
        lightchannel = "V"
      elif self.colormodel == "HSL":
        lightchannel = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        lightchannel = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormdel}.")
    print(f"HDRWT on channel(s) {channels} with (pseudo-)lightness channel {lightchannel}...")
    # HDRWT algorithm.
    image = self.copy()
    omed = np.median(image)
    for iiter in range(niter): # HDRWT iterations.
      if niter > 1: print(f"Iteration {iiter+1}/{niter}.")
      for level in range(lmin, lmax+1): # Iterate over wavelet levels.
        print(f"Processing level #{level}: ", end = "")
        # Compute the approximation of the (pseudo-)lightness at that wavelet level.
        lightness = image.get_channel(channel = lightchannel)
        if level > 0:
          approx = multiscale.slt(lightness, levels = level, starlet = starlet).coeffs[0]
        else:
          approx = lightness
        # Compute the midtone transformation and HDR fusion mask.
        amed = np.median(approx)
        if target == "bright":
          mask = np.clip(   approx, 0., 1.)**mstrength
          midtone = max(mts(amed, (1.-rstrength)*amed), .5)
        else:
          mask = np.clip(1.-approx, 0., 1.)**mstrength
          midtone = min(mts(amed, (1.+rstrength)*amed), .5)
        print(f"midtone = {midtone:.5f}.")
        # Apply the midtone transformation and blend with the original image.
        stretched = image.midtone_stretch(channels = channels, midtone = midtone, trans = False)
        image = image.blend(stretched, mask)
        image = image.midtone_stretch(midtone = mts(np.median(image), omed), channels = "L", trans = False)
      # image = image.midtone_stretch(midtone = mts(np.median(image), omed), channels = "L", trans = False)
    return image

  def HDRwt2(self, starlet = "cubic", lmin = 1, lmax = 5, rstrength = .1, mstrength = 2., target = "bright", niter = 1, channels = "", lightchannel = ""):
    """HDRWT."""
    # Check inputs.
    if target not in ["bright", "dark"]: raise ValueError("Error, target must be 'bright' or 'dark'.")
    channels = channels.strip()
    lightchannel = lightchannel.strip()
    if channels == "":
      if self.colormodel in ["RGB", "gray"]:
        channels = "Ls" if target == "dark" else "L"
      elif self.colormodel == "HSV":
        channels = "V"
      elif self.colormodel == "HSL":
        channels = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        channels = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormdel}.")
    if lightchannel == "":
      if self.colormodel in ["RGB", "gray"]:
        lightchannel = "L"
      elif self.colormodel == "HSV":
        lightchannel = "V"
      elif self.colormodel == "HSL":
        lightchannel = "L'"
      elif self.colormodel in ["Lab", "Luv", "Lch", "Lsh"]:
        lightchannel = "L*"
      else:
        raise ValueError(f"Error, unknown color model {self.colormdel}.")
    print(f"HDRWT on channel(s) {channels} with (pseudo-)lightness channel {lightchannel}...")
    # HDRWT algorithm.
    image = self.copy()
    omed = np.median(image)
    for iiter in range(niter): # HDRWT iterations.
      if niter > 1: print(f"Iteration {iiter+1}/{niter}.")
      for level in range(lmin, lmax+1): # Iterate over wavelet levels.
        print(f"Processing level #{level}...")
        # Compute the approximation of the (pseudo-)lightness at that wavelet level.
        lightness = image.get_channel(channel = lightchannel)
        if level > 0:
          mst = multiscale.slt(lightness, levels = level, starlet = starlet)
          approx = mst.coeffs[0]
        else:
          approx = lightness
        # Compress the dynamical range of the approximation.
        amax = np.max(approx)
        approx /= amax
        amed = np.median(approx)
        a = -(rstrength/amax)/(amed*(amed-1.))
        b = -a*(amed+1.)
        c = 1.-a-b
        compressed = amax*approx*(a*approx**2+b*approx+c)
        # compressed = mts(amax*approx, .5+rstrength)
        if level > 0:
          mst.coeffs[0] = compressed
          lightness = mst.inverse()
        else:
          lightness = compressed
        image.set_channel(channel = lightchannel, data = lightness, inplace = True)
        # image = image.midtone_stretch(midtone = mts(np.median(image), omed), channels = "L", trans = False)
    return image
