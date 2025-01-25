# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Image geometry management."""

import numpy as np
from PIL import Image as PILImage

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  ################################
  # Geometrical transformations. #
  ################################

  def flip_height(self):
    """Flip the image along its height.

    Returns:
      Image: The flipped image.
    """
    return np.flip(self, axis = 1)

  def flip_width(self):
    """Flip the image along its width.

    Returns:
      Image: The flipped image.
    """
    return np.flip(self, axis = 2)

  ##################
  # Resize & Crop. #
  ##################

  def resize(self, width, height, method = "lanczos"):
    """Resize the image.

    Args:
      width (int): New image width (pixels).
      height (int): New image height (pixels).
      method (str, optional): Resampling method:

        - "nearest": Nearest neighbor interpolation.
        - "bilinear": Linear interpolation.
        - "bicubic": Cubic spline interpolation.
        - "lanczos": Lanczos (truncated sinc) filter (default).
        - "box": Box average (equivalent to "nearest" for upscaling).
        - "hamming": Hamming (cosine bell) filter.

    Returns:
      Image: The resized image.
    """
    if width < 1 or width > 32768: raise ValueError("Error, width must be >= 1 and <= 32768 pixels.")
    if height < 1 or height > 32768: raise ValueError("Error, height must be >= 1 and <= 32768 pixels.")
    if width*height > 2**26: raise ValueError("Error, can not resize to > 64 Mpixels.")
    if method == "nearest":
      method = PILImage.Resampling.NEAREST
    elif method == "bilinear":
      method = PILImage.Resampling.BILINEAR
    elif method == "bicubic":
      method = PILImage.Resampling.BICUBIC
    elif method == "lanczos":
      method = PILImage.Resampling.LANCZOS
    elif method == "box":
      method = PILImage.Resampling.BOX
    elif method == "hamming":
      method = PILImage.Resampling.HAMMING
    else:
      raise ValueError(f"Error, unknown resampling method '{method}'.")
    nc = self.get_nc()
    output = np.empty((nc, height, width), dtype = params.imagetype)
    for ic in range(nc): # Resize each channel using PIL.
      # Convert to np.float32 while resizing.
      PILchannel = PILImage.fromarray(np.asarray(self.image[ic], dtype = np.float32), "F").resize((width, height), method)
      output[ic] = np.asarray(PILchannel, dtype = params.imagetype)
    return self.newImage(output)

  def rescale(self, scale, method = "lanczos"):
    """Rescale the image.

    Args:
      scale (float): Scaling factor.
      method (str, optional): Resampling method:

        - "nearest": Nearest neighbor interpolation.
        - "bilinear": Linear interpolation.
        - "bicubic": Cubic spline interpolation.
        - "lanczos": Lanczos (truncated sinc) filter (default).
        - "box": Box average (equivalent to "nearest" for upscaling).
        - "hamming": Hamming (cosine bell) filter.

    Returns:
      Image: The rescaled image.
    """
    if scale <= 0. or scale > 16.: raise ValueError("Error, scale must be > 0 and <= 16.")
    width, height = self.get_size()
    newwidth, newheight = int(round(scale*width)), int(round(scale*height))
    return self.resize(newwidth, newheight, method)

  def crop(self, xmin, xmax, ymin, ymax):
    """Crop the image.

    Args:
      xmin, xmax, ymin, ymax (integer or float): Crop from x = xmin to x = xmax and from y = ymin to y = ymax.

    Returns:
      Image: The cropped image.
    """
    if xmax <= xmin: raise ValueError("Error, xmax <= xmin.")
    if ymax <= ymin: raise ValueError("Error, ymax <= ymin.")
    width, height = self.get_size()
    xmin = max(int(np.floor(xmin))  , 0)
    xmax = min(int(np.ceil (xmax))+1, width)
    ymin = max(int(np.floor(ymin))  , 0)
    ymax = min(int(np.ceil (ymax))+1, height)
    return self.newImage(self.image[:, ymin:ymax, xmin:xmax])
