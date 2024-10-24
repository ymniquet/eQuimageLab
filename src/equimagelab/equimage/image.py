# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image class."""

import numpy as np

from . import params
from . import image_colorspaces
from . import image_utils
from . import image_geometry
from . import image_colors
from . import image_stretch
from . import image_filters
from . import image_skimage
from . import image_stats

class Image(np.ndarray,
            image_colorspaces.Mixin, image_utils.Mixin, image_geometry.Mixin,
            image_colors.Mixin, image_stretch.Mixin, image_filters.Mixin, image_skimage.Mixin,
            image_stats.Mixin):
  """Image class.
     The images are stored as arrays with shape (3, height, width) and type params.IMGTYPE = np.float32 or np.float64.
     The leading axis contains the color channels, and the next two are the height and width axes.
     The class embeds 'colorspace' and 'colormodel' attributes for the color space and model of the image.
     The 'colorspace' can be
       - "lRGB" for the linear RGB color space.
       - "sRGB" for the sRGB color space.
     The 'colormodel' can be
       - "RGB": the 3 channels of the image are the red, blue, and green values within [0, 1].
       - "HSV": the 3 channels of the image are the hue, value, and saturation within [0, 1].
     The default color space is sRGB and the default colormodel is RGB."""

  #################
  # Constructors. #
  #################

  def __new__(cls, image, colorspace = "sRGB", colormodel = "RGB"):
    """Return a new Image instance with image 'image', color space 'colorspace' and color model 'colormodel',
       if not defined by 'image'."""
    return cls.newImage(image, colorspace, colormodel)

  def __array_finalize__(self, obj):
    """Finalize object creation."""
    if obj is None: return
    self.copy_meta(obj)

  ######################
  # Object management. #
  ######################

  @classmethod
  def newImage(cls, image, colorspace = "sRGB", colormodel = "RGB"):
    """Return a new Image instance with image 'image', color space 'colorspace' and color model 'colormodel',
       if not defined by 'image'."""
    obj = np.asarray(image, dtype = params.IMGTYPE).view(cls)
    # Validate the image.
    if obj.ndim != 3: raise ValueError(f"Error, an image must have 3 dimensions (found {obj.ndim}).")
    if obj.shape[0] != 3: raise ValueError(f"Error, an image must have 3 color channels (found {image.shape[0]}).")
    # Register color space and model.
    if getattr(obj, "colorspace", None) is None: obj.colorspace = colorspace
    if getattr(obj, "colormodel", None) is None: obj.colormodel = colormodel
    return obj

  @classmethod
  def newImage_like(cls, source, image, **kwargs):
    """Return a new Image instance with image 'image' and meta-data (color space and model, ...) copied from 'source'.
       These meta-data may be overridden with the kwargs (e.g., colormodel = "HSV", etc...)."""
    obj = cls.newImage(image)
    obj.copy_meta(source)
    for name, value in kwargs.items():
      if not hasattr(obj, name): raise ValueError(f"Error, the image object has no attribute {name}.")
      setattr(obj, name, value)
    return obj

  def copy_meta(self, source):
    """Copy meta-data (color space and model, ...) from image 'source'."""
    self.colorspace = getattr(source, "colorspace", None)
    self.colormodel = getattr(source, "colormodel", None)

  def print_meta(self):
    """Print image meta-data (color space and model, ...)."""
    print("Image meta-data:")
    print(f"Color space = {self.colorspace}")
    print(f"Color model = {self.colormodel}")

  def image(self, channels = 0, cls = None):
    """Return a view on the image as a 'cls' object (np.ndarray or Image if None),
       with the color channels as axis 'channels'.
       Note: Moving the channel axis will raise an error if cls is Image (or None)."""
    view = self.view(type = cls) if cls is not None else self.view()
    return view if channels == 0 else np.moveaxis(view, 0, channels)

  ######################
  # Image conversions. #
  ######################

  def rgb8(self):
    """Return a RGB image as a (height, width, 3) array of 8 bits integers in the range [0, 255]."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*255, 0, 255)
    return np.rint(data).astype("uint8")

  def rgb16(self):
    """Return a RGB image as a (height, width, 3) array of 16 bits integers in the range [0, 255]."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*65535, 0, 65535)
    return np.rint(data).astype("uint16")

  def rgb32(self):
    """Return a RGB image as a (height, width, 3) array of 32 bits integers in the range [0, 255]."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*4294967295, 0, 4294967295)
    return np.rint(data).astype("uint32")
