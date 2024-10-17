# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image geometry management."""

import numpy as np
from PIL import Image as PILImage

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ##################
  # Image queries. #
  ##################

  def width(self):
    """Return the width of the image in pixels."""
    return self.shape[2]

  def height(self):
    """Return the height of the image in pixels."""
    return self.shape[1]

  def width_height(self):
    """Return the width and height of the image in pixels."""
    return self.shape[2], self.shape[1]

  ################################
  # Geometrical transformations. #
  ################################

  # TESTED.
  def flip_height(self):
    """Flip the input image along its height."""
    return np.flip(self, axis = 1)

  # TESTED.
  def flip_width(self):
    """Flip the input image along its width."""
    return np.flip(self, axis = 2)

  ##################
  # Resize & Crop. #
  ##################

  # TESTED.
  def resample(self, width, height, method = "lanczos"):
    """Resize image to width 'width' and height 'height' using resampling method 'method',
       which can be:
         - "nearest": Nearest neighbor interpolation.
         - "bilinear": Linear interpolation.
         - "bicubic": Cubic spline interpolation.
         - "lanczos": Lanczos (truncated sinc) filter.
         - "box": Box average (equivalent to "nearest" for upscaling).
         - "hamming": Hamming (cosine bell) filter."""
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
    image = self.view()
    resized = np.empty((3, height, width), dtype = params.IMGTYPE)
    for ic in range(3): # Resize each channel using PIL.
      PILchannel = PILImage.fromarray(np.float32(image[ic]), "F").resize((width, height), method) # Convert to np.float32 while resizing.
      resized[ic] = np.asarray(PILchannel, dtype = params.IMGTYPE)
    return self.newImage_like(self, resized)

  # TESTED.
  def rescale(self, scale, method = "lanczos"):
    """Rescale image by a factor 'scale' using resampling method 'method',
       which can be:
         - "nearest": Nearest neighbor interpolation.
         - "bilinear": Linear interpolation.
         - "bicubic": Cubic spline interpolation.
         - "lanczos": Lanczos (truncated sinc) filter.
         - "box": Box average (equivalent to "nearest" for upscaling).
         - "hamming": Hamming (cosine bell) filter."""
    if scale <= 0. or scale > 16.: raise ValueError("Error, scale must be > 0 and <= 16.")
    width, height = self.width_height()
    newwidth, newheight = int(round(scale*width)), int(round(scale*height))
    return self.resample(newwidth, newheight, method)

  # TESTED.
  def crop(self, xmin, xmax, ymin, ymax):
    """Crop image from x = xmin to x = xmax and from y = ymin to y = ymax."""
    if xmax <= xmin: raise ValueError("Error, xmax <= xmin.")
    if ymax <= ymin: raise ValueError("Error, ymax <= ymin.")
    width, height = self.width_height()
    xmin = max(int(np.floor(xmin))  , 0)
    xmax = min(int(np.ceil (xmax))+1, width)
    ymin = max(int(np.floor(ymin))  , 0)
    ymax = min(int(np.ceil (ymax))+1, height)
    image = self.view()
    return image[:, ymin:ymax, xmin:xmax].copy()
