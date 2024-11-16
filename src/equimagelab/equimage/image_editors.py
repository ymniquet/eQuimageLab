# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""External image editors."""

import os
import tempfile
import subprocess
import numpy as np

from . import params

from .image_io import load_image_as_array

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  def edit_with(self, command, export = "tiff", depth = 16, editor = "<Editor>"):
    """Edit the image with an external tool.

    The image is saved on disk; the editor command is then run on this file, which is finally
    reloaded in eQuimageLab and returned.

    The user must simply overwrite the edited file when leaving the editor.

    Args:
      command (str): The command to be run (e.g., "gimp -n $"). Any "$" is replaced by the name of the image
        file to be opened by the editor.
      export (str, optional): The format used to export the image. Can be:
        - "png": PNG file with depth = 8 or 16 bits integer per channel.
        - "tiff" (default): TIFF file with depth = 8, 16 or 32 bits integer per channel.
        - "fits": FITS file width depth = 32 bits float per channel.
      depth (int, optional): The color depth (bits per channel) used to export the image (see above; default 16).
      editor (str, optional): The name of the editor (for pretty-print purposes; default "<Editor>").

    Returns:
      Image: The edited image.
    """
    if export not in ["png", "tiff", "fits"]: raise ValueError(f"Error, unknown export format '{export}'.")
    if depth not in [8, 16, 32]: raise ValueError("Error, depth must be 8, 16 or 32 bpc.")
    # Edit image.
    with tempfile.TemporaryDirectory() as tmpdir:
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
      self.save(filepath, depth = depth, verbose = False)
      ctime = os.path.getmtime(filepath)
      # Run editor.
      print(f"Running {editor}...")
      print(f"Overwrite file {filepath} when leaving {editor}.")
      subprocess.run(splitcmd)
      # Load and return edited image.
      mtime = os.path.getmtime(filepath)
      if mtime == ctime:
        print(f"The image has not been modified by {editor}; Returning the original...")
        return self.copy()
      print(f"Reading file {filepath}...")
      image, meta = load_image_as_array(filepath, verbose = False)
      return self.newImage(image)

  def edit_with_gimp(self, export = "tiff", depth = 16):
    """Edit the image with GIMP.

    The image is saved on disk; GIMP is then run on this file, which is finally
    reloaded in eQuimageLab and returned.

    The user must simply overwrite the edited file when leaving GIMP.

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
    return self.edit_with("gimp -n $", export = export, depth = depth, editor = "GIMP")

  def edit_with_siril(self):
    """Edit the image with SIRIL.

    The image is saved as a FITS file (32 bits float per channel) on disk; SIRIL is then run on
    this file, which is finally reloaded in eQuimageLab and returned.

    The user must simply overwrite the edited file when leaving SIRIL.

    The command "siril" must be in the PATH.

    Returns:
      Image: The edited image.
    """
    return self.edit_with("siril $", export = "fits", depth = 32, editor = "SIRIL")
