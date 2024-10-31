# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01
# DOC+MCI.

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
  
  Color images are stored as arrays with shape (3, height, width) and grayscale images as arrays with
  shape (1, height, width). The leading axis spans the color channels, and the next two are the height 
  and width axes. The images are encoded as floats with type params.IMGTYPE = np.float32 or np.float64).    
  The class embeds colorspace and colormodel attributes for the color space and model of the image.
  
  The colorspace attribute can be:
    - "lRGB" for the linear RGB color space.
    - "sRGB" for the sRGB color space.
    
  The colormodel attribute can be:
    - "RGB": the 3 channels of the image are the red, blue, and green values within [0, 1].
    - "HSV": the 3 channels of the image are the hue, value, and saturation within [0, 1].
    - "gray": grayscale image with one single channel within [0, 1].
    
  The default color space is sRGB and the default color model is RGB.
  """

  #################
  # Constructors. #
  #################

  def __new__(cls, image, colorspace = "sRGB", colormodel = "RGB"):
    """Return a new Image object with the input image.
    
    Args:
      image: The input image (np.array or Image).
      colorspace (str, optional): The image color space (if not defined by the input image, default sRGB).
      colormodel (str, optional): The image color model (if not defined by the input image, default  RGB).
      
    Returns:
      Image: The new image object.
    """
    return cls.newImage(image, colorspace, colormodel)

  def __array_finalize__(self, obj):
    """Finalize object creation.
    
    Args: 
      obj: The parent object.
    """  
    if obj is None: return
    # Is self a valid image ?
    if self.ndim != 3: return
    if self.shape[0] not in [1, 3]: return
    if self.dtype != params.IMGTYPE: return
    # If so, copy meta-data from obj.
    self.__copy_meta__(obj)
    
  def __copy_meta__(self, source):
    """Copy meta-data from the source.

    Note: The colormodel attribute can not be overridden if the image is a grayscale.

    Args:
      source (Image): The source for meta-data.
    """
    self.colorspace = getattr(source, "colorspace", "sRGB")
    self.colormodel = getattr(source, "colormodel",  "RGB") if self.shape[0] > 1 else "gray"

  ######################
  # Object management. #
  ######################

  @classmethod
  def newImage(cls, image, colorspace = "sRGB", colormodel = "RGB"):
    """Return a new Image object with the input image.
    
    Args:
      image: The input image (np.array or Image).
      colorspace (str, optional): The image color space (if not defined by the input image, default sRGB).
      colormodel (str, optional): The image color model (if not defined by the input image, default  RGB).
      
    Returns:
      Image: The new image object.
    """
    colorspace = getattr(image, "colorspace", colorspace) 
    colormodel = getattr(image, "colormodel", colormodel)       
    obj = np.asarray(image, dtype = params.IMGTYPE).view(cls)
    # Validate the image.
    if obj.ndim == 2: 
      colormodel = "gray"
      obj = np.expand_dims(obj, axis = 0)      
    elif obj.ndim == 3:
      nc = obj.shape[0]
      if nc == 1:
        colormodel = "gray"
      elif nc != 3: 
        raise ValueError(f"Error, a color image must have 3 channels (found {nc}).")
    else:
      raise ValueError(f"Error, an image must have 2 (grayscale) or 3 (color) dimensions (found {obj.ndim}).")
    # Register color space and model.
    obj.colorspace = colorspace
    obj.colormodel = colormodel    
    return obj

  @classmethod
  def newImage_like(cls, source, image, **kwargs):
    """Return a new Image object with the input image but the meta-data (color space and model, ...) from an other source.
      
    These meta-data may be overridden with the kwargs (e.g., colorspace = "lRGB", etc...).
    The colormodel attribute can not, however, be overridden if the image is a grayscale.

    Args:
      image: The input image (np.array or Image).
      source (Image): The source for meta-data.      
      kwargs: The meta-data to be overridden (e.g., colorspace = "lRGB", ...).
      
    Returns:
      Image: The new image object.    
    """
    obj = cls.newImage(image)
    obj.__copy_meta__(source)
    for name, value in kwargs.items():
      if not hasattr(obj, name): raise ValueError(f"Error, the image object has no attribute {name}.")
      if name == "colormodel" and obj.shape[0] == 1: 
        continue # Do not override color model of grayscale images.
      setattr(obj, name, value)
    return obj

  def image(self, channels = 0, cls = None):
    """Return a view on the image.
    
    Args:
      cls, optional: The class of the returned view object [np.ndarray or Image if None (default)]
      channels: Position of the channel axis. Moving the channel axis will raise an error if cls is Image (or None).
    
    Returns:
      A view on the image with class cls.
    """
    view = self.view(type = cls) if cls is not None else self.view()
    return view if channels == 0 else np.moveaxis(view, 0, channels)

  ##################
  # Image queries. #
  ##################
  
  def get_nc(self):
    """Return the number of channels of the image.
    
    Returns:
      int: The number of channels of the image.
    """
    return self.shape[0]
  
  def print_meta(self):
    """Print the image meta-data (color space and model, ...)."""
    print("Image meta-data:")
    print(f"Color space = {self.colorspace}")
    print(f"Color model = {self.colormodel}")
  
  ######################
  # Image conversions. #
  ######################

  def int8(self):
    """Return the image as a (height, width, channels) array of 8 bits integers in the range [0, 255].
    
    Returns: 
      np.array: The image as a (height, width, channels) array of 8 bits integers in the range [0, 255].
    """
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*255, 0, 255)
    return np.rint(data).astype("uint8")

  def int16(self):
    """Return the image as a (height, width, channels) array of 16 bits integers in the range [0, 65535].

    Returns: 
      np.array: The image as a (height, width, channels) array of 16 bits integers in the range [0, 65535].
    """
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*65535, 0, 65535)
    return np.rint(data).astype("uint16")

  def int32(self):
    """Return the image as a (height, width, channels) array of 32 bits integers in the range [0, 4294967295].
    
    Returns: 
      np.array: The image as a (height, width, channels) array of 32 bits integers in the range [0, 4294967295].
    """
    image = self.image(cls = np.ndarray, channels = -1)
    data = np.clip(image*4294967295, 0, 4294967295)
    return np.rint(data).astype("uint32")
