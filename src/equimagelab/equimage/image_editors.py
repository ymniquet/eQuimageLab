# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""External image editors."""

import os
import shutil
import tempfile
import subprocess
import numpy as np

from . import params

from .image_io import load_image_as_array
from .image_stretch import harmonic_through

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def edit_with(self, command, export = "tiff", depth = 16, editor = "<Editor>", interactive = True, cwd = None):
    """Edit the image with an external tool.

    The image is saved on disk; the editor command is then run on this file, which is finally
    reloaded in eQuimageLab and returned.

    The user/editor must simply overwrite the edited file when leaving.

    Args:
      command (str): The command to be run (e.g., "gimp -n $").
        Any "$" is replaced by the name of the image file to be opened by the editor.
      export (str, optional): The format used to export the image. Can be:

        - "png": PNG file with depth = 8 or 16 bits integer per channel.
        - "tiff" (default): TIFF file with depth = 8, 16 or 32 bits integer per channel.
        - "fits": FITS file width depth = 32 bits float per channel.

      depth (int, optional): The color depth (bits per channel) used to export the image (see above; default 16).
      editor (str, optional): The name of the editor (for pretty-print purposes; default "<Editor>").
      interactive (bool, optional): If True (default), the editor is interactive (awaits commands from the user);
        if False, the editor processes the image autonomously and does not require inputs from the user.
      cwd (str, optional): If not None (default), change working directory to cwd before running the editor.

    Returns:
      Image: The edited image.
    """
    if export not in ["png", "tiff", "fits"]: raise ValueError(f"Error, unknown export format '{export}'.")
    if depth not in [8, 16, 32]: raise ValueError("Error, depth must be 8, 16 or 32 bpc.")
    # Edit image.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors = True) as tmpdir:
      # Set tmp file name.
      filename = "eQuimageLab."+export
      filepath = os.path.join(tmpdir, filename)
      # Process command.
      splitcmd = []
      filefound = False
      for item in command.strip().split(" "):
        if item == "$":
          filefound = True
          splitcmd.append(filepath)
        else:
          splitcmd.append(item)
      if not filefound: raise ValueError("Error, no place holder for the image file ($) found in the command.")
      # Save image.
      print(f"Writing file {filepath} with depth = {depth} bpc...")
      self.save(filepath, depth = depth, compress = 0, verbose = False) # Don't compress image to ensure compatibility with the editor.
      ctime = os.path.getmtime(filepath)
      # Run editor.
      print(f"Running {editor}...")
      if interactive: print(f"Overwrite file {filepath} when leaving {editor}.")
      subprocess.run(splitcmd, cwd = cwd)
      # Load and return edited image.
      mtime = os.path.getmtime(filepath)
      if mtime == ctime:
        print(f"The image has not been modified by {editor}; Returning the original...")
        return self.copy()
      print(f"Reading file {filepath}...")
      image, meta = load_image_as_array(filepath, verbose = False)
      return self.newImage(image)

  def edit_with_gimp(self, export = "tiff", depth = 16):
    """Edit the image with Gimp.

    The image is saved on disk; Gimp is then run on this file, which is finally
    reloaded in eQuimageLab and returned.

    The user must simply overwrite the edited file when leaving Gimp.

    The command "gimp" must be in the PATH.

    Args:
      export (str, optional): The format used to export the image. Can be:

        - "png": PNG file with depth = 8 or 16 bits integer per channel.
        - "tiff" (default): TIFF file with depth = 8, 16 or 32 bits integer per channel.
        - "fits": FITS file width depth = 32 bits float per channel.

      depth (int, optional): The color depth (bits per channel) used to export the image (see above; default 16).

    Returns:
      Image: The edited image.
    """
    return self.edit_with("gimp -n $", export = export, depth = depth, editor = "Gimp", interactive = True)

  def edit_with_siril(self):
    """Edit the image with Siril.

    The image is saved as a FITS file (32 bits float per channel) on disk; Siril is then run on
    this file, which is finally reloaded in eQuimageLab and returned.

    The user must simply overwrite the edited file when leaving Siril.

    The command "siril" must be in the PATH.

    Returns:
      Image: The edited image.
    """
    return self.edit_with("siril $", export = "fits", depth = 32, editor = "Siril", interactive = True)

  def starnet(self, midtone = .5, starmask = False):
    """Remove the stars from the image with StarNet++.

    See: https://www.starnetastro.com/

    The image is saved as a TIFF file (16 bits integer per channel) on disk; the stars on this
    TIFF file are removed with StarNet++, and the starless image is finally reloaded in eQuimageLab
    and returned.

    The command "starnet++" must be in the PATH.

    Args:
      midtone (float, optional): If different from 0.5 (default), apply a midtone stretch to the
        image before running StarNet++, then apply the inverse stretch to the output starless.
        This can help StarNet++ find stars on low contrast, linear RGB images.
        See Image.midtone_stretch; midtone can either be "auto" (for automatic stretch) or a
        float in ]0, 1[.
      starmask (bool, optional): If True, return both the starless image and the star mask.
        If False (default), only return the starless image [the star mask being the difference
        between the original image (self) and the starless].

    Returns:
      Image: The starless image if starmask is False, and a tuple (starless image, star mask)
      if starmask is True.
    """
    # We need to cwd to the starnet++ directory to process the image.
    cmdpath = shutil.which("starnet++")
    if cmdpath is None: raise FileNotFoundError("Error, starnet++ executable not found in the PATH.")
    path, cmd = os.path.split(cmdpath)
    # Stretch the image if needed.
    if midtone == "auto":
      avgmedian = np.mean(np.median(self.image, axis = (-1, -2)))
      midtone = 1./(harmonic_through(avgmedian, .33)+2.)
    if midtone != .5:
      image = self.midtone_stretch(midtone)
    else:
      image = self
    # Run starnet++.
    starless = image.edit_with("starnet++ $ $", export = "tiff", depth = 16, editor = "StarNet++", interactive = False, cwd = path)
    # "Unstretch" the starless if needed.
    if midtone != .5:
      starless = starless.midtone_stretch(midtone, inverse = True)
    # Return starless/star masks as appropriate.
    return (starless, self-starless) if starmask else starless
