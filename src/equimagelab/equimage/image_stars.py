# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.0 / 2025.03.08
# Doc OK.

"""Stars management."""

import os
import shutil
import numpy as np

from .image_stretch import Dharmonic_through

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def starnet(self, midtone = .5, starmask = False):
    """Remove the stars from the image with StarNet++.

    See: https://www.starnetastro.com/

    The image is saved as a TIFF file (16 bits integer per channel); the stars are removed from this
    TIFF file with StarNet++, and the starless image is finally reloaded in eQuimageLab and returned.

    The command "starnet++" must be in the PATH.

    Args:
      midtone (float, optional): If different from 0.5 (default), apply a midtone stretch to the
        input image before running StarNet++, then apply the inverse stretch to the output starless.
        This can help StarNet++ find stars on low contrast, linear RGB images.
        See :meth:`Image.midtone_stretch() <.midtone_stretch>`; midtone can either be "auto" (for
        automatic stretch) or a float in ]0, 1[.
      starmask (bool, optional): If True, return both the starless image and the star mask.
        If False (default), only return the starless image [the star mask being the difference
        between the original image (self) and the starless].

    Returns:
      Image: The starless image if starmask is False, and both the starless image and star mask if
      starmask is True.
    """
    # We need to cwd to the starnet++ directory to process the image.
    cmdpath = shutil.which("starnet++")
    if cmdpath is None: raise FileNotFoundError("Error, starnet++ executable not found in the PATH.")
    path, cmd = os.path.split(cmdpath)
    # Stretch the input image if needed.
    if midtone == "auto":
      avgmedian = np.mean(np.median(self.image, axis = (-1, -2)))
      midtone = 1./(Dharmonic_through(avgmedian, .33)+2.)
    if midtone != .5:
      image = self.midtone_stretch(midtone)
    else:
      image = self
    # Run starnet++.
    starless = image.edit_with("starnet++ $FILE$ $FILE$", export = "tiff", depth = 16, editor = "StarNet++", interactive = False, cwd = path)
    # "Unstretch" the starless if needed.
    if midtone != .5:
      starless = starless.midtone_stretch(midtone, inverse = True)
    # Return starless/star masks as appropriate.
    return starless, self-starless if starmask else starless

  def synthetic_stars_siril(self):
    """Resynthetize stars with Siril.

    This method runs Siril to find the stars on the image and resynthetize them with "perfect"
    Gaussian or Moffat shapes. This can be used to correct coma and other aberrations.

    Note:
      Star resynthesis works best on a star mask produced by :meth:`Image.starnet() <.starnet>`.
      The synthetic star mask and the starless image must then be blended together.

    The command "siril-cli" must be in the PATH.

    Returns:
      Image: The edited image, with stars resynthetized by Siril.
    """
    script = ("requires 1.2.0"
              "load $FILE$"
              "setfindstars -roundness=0.10"
              "findstar"
              "synthstar"
              "save $FILE$")
    return self.edit_with("siril-cli -s $SCRIPT$", export = "fits", depth = 32, script = script, editor = "Siril", interactive = False)

