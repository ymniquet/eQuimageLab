# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Histogram stretch."""

import numpy as np

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ### TODO : Add Value & Saturation for HSV images.

  def midtone_correction(self, midtone = .5, channels = "L"):
    """Apply midtone correction to selected channels of the input RGB image.
       channels can be "V" (value), "L" (luma) or any combination of "R" (red) "G" (green), and "B" (blue)."""
    self.check_color_model("RGB")
    if midtone <= 0.: raise ValueError("Error, midtone must be >= 0.")
    if channels in ["V", "L"]:
      channel = self.value() if channels == "V" else self.luma()
      clipped = np.clip(channel, 0., 1.)
      stretched = (midtone-1.)*clipped/((2.*midtone-1.)*clipped-midtone)
      return self.scale_pixels(channel, stretched)
    else:
      stretched = self.copy()
      for ic, key in ((0, "R"), (1, "G"), (2, "B")):
        if key in channels:
          clipped = np.clip(stretched[ic], 0., 1.)
          stretched[ic] = (midtone-1.)*clipped/((2.*midtone-1.)*clipped-midtone)
      return stretched
