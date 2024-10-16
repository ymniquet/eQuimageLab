# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Color spaces and models management."""

import numpy as np
import skimage.color as skcolor

from . import params

#############################
# sRGB <-> lRGB conversion. #
#############################

def sRGB_to_lRGB(image):
  """Convert the input sRGB image into a linear RGB image."""
  srgb = np.clip(image, 0., 1.)
  return np.where(srgb > .04045, ((srgb+0.055)/1.055)**2.4, srgb/12.92)

def lRGB_to_sRGB(image):
  """Convert the input linear RGB image into a sRGB image."""
  lrgb = np.clip(image, 0., 1.)
  return np.where(lrgb > .0031308, 1.055*lrgb**(1./2.4)-0.055, 12.92*lrgb)

#########
# Luma. #
#########

def get_RGB_luma():
  """Return the weights of the RGB components of the luma as a tuple (red, green, blue)."""
  return params.rgbluma

def set_RGB_luma(rgb):
  """Set the weights of the RGB components of the luma.
     The input RGB components can be:
       - a tuple, list or array of the (red, green, blue) weights. They will be normalized so that their sum is 1.
       - the string "uniform": the weights are set to (1/3, 1/3, 1/3).
       - the string "human": the weights are set to (.3, .6, .1), which mimics (approximately) human vision."""
  if isinstance(rgb, str):
    if rgb == "uniform":
      set_RGB_luma((1./3., 1./3., 1./3.))
    elif rgb == "human":
      set_RGB_luma((.3, .6, .1))
    else:
      raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
  else:
    w = np.array(rgb, dtype = params.IMGTYPE)
    if w.shape != (3,): raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
    if any(w < 0.): raise ValueError("Error, the input rgb weights must be >= 0.")
    s = np.sum(w)
    if s == 0.: raise ValueError("Error, the sum of the input rgb weights must be > 0.")
    params.rgbluma = tuple(w/s)
    print(f"Luma RGB weights = ({params.rgbluma[0]:.3f}, {params.rgbluma[1]:.3f}, {params.rgbluma[2]:.3f}).")

def luma(image):
  """Return the luma of the input RGB image,
     defined as the average of the RGB components weighted by rgbluma = get_RGB_luma():
       luma = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2]."""
  return params.rgbluma[0]*image[0]+params.rgbluma[1]*image[1]+params.rgbluma[2]*image[2]

############################
# Luminance and lightness. #
############################

def lRGB_luminance(image):
  """Return the luminance Y of the input linear RGB image."""
  return .2126*image[0]+0.7152*image[1]+0.0722*image[2]

def lRGB_lightness(image):
  """Return the CIE lightness L* of the input linear RGB image.
     Warning: L* is defined within [0, 100] instead of [0, 1]."""
  Y = lRGB_luminance(image)
  return np.where(Y > .008856, 116.*Y**(1./3.)-16., 903.3*Y)

def sRGB_luminance(image):
  """Return the luminance Y of the input sRGB image."""
  return lRGB_luminance(sRGB_to_lRGB(image))

def sRGB_lightness(image):
  """Return the CIE lightness L* of the input sRGB image.
     Warning: L* is defined within [0, 100] instead of [0, 1]."""
  return lRGB_lightness(sRGB_to_lRGB(image))

###########################
# RGB <-> HSV conversion. #
###########################

def value(image):
  """Return the HSV value = max(RGB) of the input RGB image."""
  return image.max(axis = 0)

def saturation(image):
  """Return the HSV saturation = 1-min(RGB)/max(RGB) of the input RGB image."""
  return 1.-image.min(axis = 0)/image.max(axis = 0, initial = params.IMGTOL) # Safe evaluation.

def RGB_to_HSV(image):
  """Convert the input RGB image into a HSV image."""
  return skcolor.rgb2hsv(image, channel_axis = 0)

def HSV_to_RGB(image):
  """Convert the input HSV image into a RGB image."""
  return skcolor.hsv2rgb(image, channel_axis = 0)

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  #####################################
  # Color space and model management. #
  #####################################

  def set_color_space_model(colorspace, colormodel):
    """Set color space 'colorspace' and color model 'color model'."""
    self.colorspace = colorspace
    self.colormodel = colormodel

  def color_space_error(self):
    """Raise a color space error."""
    class ColorSpaceError(Exception):
      pass
    raise ColorSpaceError(f"Error, this operation is not implemented for {self.colorspace} images.")

  def color_model_error(self):
    """Raise a color model error."""
    class ColorModelError(Exception):
      pass
    raise ColorModelError(f"Error, this operation is not implemented for {self.colormodel} images.")

  def check_color_spaces(self, *colorspaces):
    """Raise an error if the color space of the image is not in the arguments."""
    if self.colorspace not in colorspaces: self.color_space_error()

  def check_color_model(self, *colormodels):
    """Raise an error if the color model of the image is not in the arguments."""
    if self.colormodel not in colormodels: self.color_model_error()

  ############################
  # Color space conversions. #
  ############################

  def lRGB(self):
    """Convert the image to the linear RGB color space."""
    self.check_color_model("RGB")
    if self.colorspace == "lRGB":
      return self.copy()
    elif self.colorspace == "sRGB":
      image = self.image()
      return self.newImage_like(self, sRGB_to_lRGB(image), colorspace = "lRGB")
    else:
      self.color_space_error()

  def sRGB(self):
    """Convert the image to the sRGB color space."""
    self.check_color_model("RGB")
    if self.colorspace == "lRGB":
      image = self.image()
      return self.newImage_like(self, lRGB_to_sRGB(image), colorspace = "sRGB")
    elif self.colorspace == "sRGB":
      return self.copy()
    else:
      self.color_space_error()

  ############################
  # Color model conversions. #
  ############################

  def RGB(self):
    """Convert the image to the RGB color model."""
    if self.colormodel == "RGB":
      return self.copy()
    elif self.colormodel == "HSV":
      image = self.image()
      return self.newImage_like(self, HSV_to_RGB(image), colormodel = "RGB")
    else:
      self.color_model_error()

  def HSV(self):
    """Convert the image to the HSV color model."""
    if self.colormodel == "RGB":
      image = self.image()
      return self.newImage_like(self, RGB_to_HSV(image), colormodel = "HSV")
    elif self.colormodel == "HSV":
      return self.copy()
    else:
      self.color_model_error()

  #######################
  # Composite channels. #
  #######################

  def luminance(self):
    """Return the luminance."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray)
    if self.colorspace == "lRGB":
      return lRGB_luminance(image)
    elif self.colorspace == "sRGB":
      return sRGB_luminance(image)
    else:
      self.color_space_error()

  def lightness(self):
    """Return the lightness."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray)
    if self.colorspace == "lRGB":
      return lRGB_lightness(image)
    elif self.colorspace == "sRGB":
      return sRGB_lightness(image)
    else:
      raise self.color_space_error()

  def luma(self):
    """Return the luma."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray)
    return luma(image)

  def value(self):
    """Return the HSV value = max(RGB)."""
    image = self.image(cls = np.ndarray)
    if self.colormodel == "RGB":
      return value(image)
    elif self.colormodel == "HSV":
      return image[2]
    else:
      self.color_model_error()

  def saturation(self):
    """Return the HSV saturation = 1-min(RGB)/max(RGB)."""
    image = self.image(cls = np.ndarray)
    if self.colormodel == "RGB":
      return saturation(image)
    elif self.colormodel == "HSV":
      return image[1]
    else:
      self.color_model_error()
