# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Image class."""

import copy
import numpy as np

from . import params
from . import image_colorspaces
from . import image_utils
from . import image_geometry
from . import image_colors
from . import image_stretch
from . import image_filters
from . import image_skimage
from . import image_masks
from . import image_stats
from . import image_editors
from . import image_io

class Image(np.lib.mixins.NDArrayOperatorsMixin,
            image_colorspaces.Mixin, image_utils.Mixin, image_geometry.Mixin,
            image_colors.Mixin, image_stretch.Mixin, image_filters.Mixin, image_skimage.Mixin,
            image_masks.Mixin, image_stats.Mixin, image_editors.Mixin, image_io.Mixin):
  """Image class.

  The image is stored as self.image, a numpy.ndarray with dtype params.IMGTYPE = np.float32 or np.float64.
  Color images are stored as arrays with shape (3, height, width) and grayscale images as arrays with
  shape (1, height, width). The leading axis spans the color channels, and the last two the height
  and width of the image.

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

  ################
  # Constructor. #
  ################

  def __init__(self, image, colorspace = "sRGB", colormodel = "RGB", channels = 0):
    """Initialize a new Image object with the input image.

    Args:
      image: The input image (numpy.ndarray or Image).
      colorspace (str, optional): The image color space (default "sRGB").
        Can be "lRGB" (linear RGB color space) or "sRGB" (sRGB color space).
      colormodel (str, optional): The image color model (default "RGB").
        Can be "RGB" (RGB image), "HSV" (HSV image) or "gray (grayscale image).
      channels (int, optional): The position of the channel axis for color images (default 0).
    """
    # Check color space and model.
    if colorspace not in ["lRGB", "sRGB"]:
      raise ValueError(f"Error, the color space must either be 'lRGB' or 'sRGB' (got {colorspace}).")
    if colormodel not in ["RGB", "HSV", "gray"]:
      raise ValueError(f"Error, the color model must either be 'RGB', 'HSV' or 'gray' (got {colormodel}).")
    # Convert the input image into an array.
    image = np.asarray(image, dtype = params.IMGTYPE)
    # Validate the image.
    if image.ndim == 2:
      colormodel = "gray"  # Enforce colormodel = "gray".
      image = np.expand_dims(image, axis = 0)
    elif image.ndim == 3:
      if channels != 0: image = np.moveaxis(image, channels, 0)
      nc = image.shape[0]
      if nc == 1:
        colormodel = "gray" # Enforce colormodel = "gray".
      elif nc == 3:
        if colormodel == "gray":
          raise ValueError(f"Error, a grayscale image must have one single channel (found {nc}).")
      else:
        raise ValueError(f"Error, an image must have 1 or 3 channels (found {nc}).")
    else:
      raise ValueError(f"Error, an image must have 2 or 3 dimensions (found {image.ndim}).")
    # Register image, color space and model.
    self.image = image
    self.colorspace = colorspace
    self.colormodel = colormodel

  def newImage(self, image, **kwargs):
    """Return a new Image object with the input image.

    Args:
      image (numpy.ndarray): The input image.
      colorspace (str, optional): The image color space (default self.colorspace).
        Can be "lRGB" (linear RGB color space) or "sRGB" (sRGB color space).
      colormodel (str, optional): The image color model (default self.colormodel).
        Can be "RGB" (RGB image), "HSV" (HSV image) or "gray (grayscale image).

    Returns:
      Image: The new Image object.
    """
    colorspace = kwargs.pop("colorspace", self.colorspace)
    colormodel = kwargs.pop("colormodel", self.colormodel)
    if kwargs: print("Discarding extra keyword arguments in Image.newImage...")
    return Image(image, colorspace = colorspace, colormodel = colormodel)

  def copy(self):
    """Return a copy of the object.

    Returns:
      Image: A (deep) copy of the object.
    """
    return copy.deepcopy(self)

  ######################
  # Object management. #
  ######################

  def __repr__(self):
    """Return the object representation."""
    return f"{self.__class__.__name__}(colorspace = {self.colorspace}, colormodel = {self.colormodel}, size = {self.image.shape[2]}x{self.image.shape[1]} pixels)"

  def __array__(self, dtype = None, copy = None):
    """Expose the object as an numpy.ndarray."""
    return np.array(self.image, dtype = dtype, copy = copy)

  def __array_ufunc__(self, ufunc, method, *args, **kwargs):
    """Apply numpy ufuncs to the object."""
    if method != "__call__": return
    inputs = []
    mixed = False
    reference = None
    for arg in args:
      if isinstance(arg, Image):
        if reference is None:
          reference = arg
        else:
          if arg.colorspace != reference.colorspace or arg.colormodel != reference.colormodel and not mixed:
            print("Warning ! This operation mixes images with different color spaces or models !..")
            mixed = True
        inputs.append(arg.image)
      else:
        inputs.append(arg)
    output = ufunc(*inputs, **kwargs)
    if isinstance(output, np.ndarray):
      if output.ndim == 0: # Is output actually a scalar ?
        return output[()]
      else:
        if not mixed and output.shape == reference.image.shape:
          return Image(output, colorspace = reference.colorspace, colormodel = reference.colormodel)
        else:
          return output
    else:
      return output

  def __array_function__(self, func, types, args, kwargs):
    """Apply numpy array functions to the object."""
    inputs = []
    mixed = False
    reference = None
    for arg in args:
      if isinstance(arg, Image):
        if reference is None:
          reference = arg
        else:
          if arg.colorspace != reference.colorspace or arg.colormodel != reference.colormodel and not mixed:
            print("Warning ! This operation mixes images with different color spaces or models !..")
            mixed = True
        inputs.append(arg.image)
      else:
        inputs.append(arg)
    output = func(*inputs, **kwargs)
    if isinstance(output, np.ndarray):
      if output.ndim == 0: # Is output actually a scalar ?
        return output[()]
      else:
        if not mixed and output.shape == reference.image.shape:
          return Image(output, colorspace = reference.colorspace, colormodel = reference.colormodel)
        else:
          return output
    else:
      return output

  ##################
  # Image queries. #
  ##################

  def get_image(self, channels = 0, copy = False):
    """Return the image data.

    Args:
      channels (int, optional): The position of the channel axis (default 0).
      copy (bool, optional): If True, return a copy of the image data;
                             If False (default), return a view.

    Returns:
      numpy.ndarray: The image data.
    """
    image = self.image
    if channels != 0: image = np.moveaxis(image, 0, channels)
    return image.copy() if copy else image

  def get_shape(self):
    """Return the shape of the image data.

    Returns:
      tuple: (number of channels, height of the image in pixels, width of the image in pixels).
    """
    return self.image.shape

  def get_size(self):
    """Return the width and height of the image.

    Returns:
      tuple: (width, height) of the image in pixels.
    """
    return self.image.shape[2], self.image.shape[1]

  def get_nc(self):
    """Return the number of channels of the image.

    Returns:
      int: The number of channels of the image.
    """
    return self.image.shape[0]

  ######################
  # Image conversions. #
  ######################

  def int8(self):
    """Return the image as a (height, width, channels) array of 8 bits integers in the range [0, 255].

    Returns:
      numpy.ndarray: The image as a (height, width, channels) array of 8 bits integers in the range [0, 255].
    """
    image = self.get_image(channels = -1)
    data = np.clip(image*255, 0, 255)
    return np.rint(data).astype("uint8")

  def int16(self):
    """Return the image as a (height, width, channels) array of 16 bits integers in the range [0, 65535].

    Returns:
      numpy.ndarray: The image as a (height, width, channels) array of 16 bits integers in the range [0, 65535].
    """
    image = self.get_image(channels = -1)
    data = np.clip(image*65535, 0, 65535)
    return np.rint(data).astype("uint16")

  def int32(self):
    """Return the image as a (height, width, channels) array of 32 bits integers in the range [0, 4294967295].

    Returns:
      numpy.ndarray: The image as a (height, width, channels) array of 32 bits integers in the range [0, 4294967295].
    """
    image = self.get_image(channels = -1)
    data = np.clip(image*4294967295, 0, 4294967295)
    return np.rint(data).astype("uint32")
