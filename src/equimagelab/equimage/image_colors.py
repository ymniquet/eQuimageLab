# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Color management."""

import numpy as np

from . import params
from . import image_colorspaces as colorspaces

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  ##################
  # Image queries. #
  ##################

  def is_gray_scale(self):
    """Return True if a RGB image is a gray scale (same RGB channels), False otherwise."""
    self.check_color_model("RGB")
    image = self.image(cls = np.ndarray)
    return np.all(abs(image[1]-image[0]) < params.IMGTOL) and np.all(abs(image[2]-image[0]) < params.IMGTOL)

  ##########################
  # Color transformations. #
  ##########################

  # TESTED.
  def RGB_balance(self, red = 1., green = 1., blue = 1.):
    """Multiply the red channel of a RGB image by 'red', the green channel by 'green', and the blue channel by 'blue'."""
    self.check_color_model("RGB")
    if red < 0.: raise ValueError("Error, red must be >= 0.")
    if green < 0.: raise ValueError("Error, green must be >= 0.")
    if blue < 0.: raise ValueError("Error, blue must be >= 0.")
    image = self.copy()
    if red   != 1.: image[0] *= red
    if green != 1.: image[1] *= green
    if blue  != 1.: image[2] *= blue
    return image

  ############################
  # Gray scales & negatives. #
  ############################

  # TESTED.
  def negative(self):
    """Return the negative of a RGB image."""
    self.check_color_model("RGB")
    return 1.-self

  # TESTED.
  def gray_scale(self, channel = "Y"):
    """Convert the selected channel of a RGB image into a gray scale image.
       'channel' can be "V" (value), "L" (luma) or "Y" (luminance)."""
    self.check_color_model("RGB")
    if channel == "V":
      grayscale = self.value()
    elif channel == "L":
      grayscale = self.luma()
    elif channel == "Y":
      grayscale = self.luminance()
      if self.colorspace == "sRGB":
        grayscale = colorspaces.lRGB_to_sRGB(grayscale) # Preserve luminance in the sRGB color space.
    else:
      raise ValueError(f"Error, invalid channel '{channel}' (must be 'V', 'L' or 'Y').")
    return self.newImage_like(self, np.repeat(grayscale[np.newaxis, :, :,], 3, axis = 0))

  def SCNR(self, hue = "green", protection "avgneutral", mixing = 1., lightness = True):
    """Selective color noise reduction of a given 'hue' of a RGB image. 'hue' can be "red" alias "R", 
       "yellow" alias "Y", "green" alias "G", "cyan" alias "C", "blue" alias "B", or "magenta" alias "M".
       The selected hue is reduced according to the 'protection' mode. For the green hue for example,
         - G <- min(G, c) with c = (R+B)/2 for average neutral protection ('protection' = "avgneutral").
         - G <- min(G, c) with c = max(R, B) for maximum neutral protection ('protection' = "maxneutral").
         - G <- G[(1-m)+c*m] with c = (R+B)/2 and m = 'mixing' for additive mask protection ('protection' = "addmask").
         - G <- G[(1-m)+c*m] with c = max(R, B) and m = 'mixing' for maximum mask protection ('protection' = "maxmask").
       The RGB components of each pixel are then rescaled to preserve the CIE lightness L* if 'lightness' is True."""
    self.check_color_model("RGB")
    if hue == "red" or hue == "R":
      icc, ic1, ic2, negative = 0, 1, 2, False
    elif hue == "yellow" or hue == "Y":
      icc, ic1, ic2, negative = 2, 0, 1, True
    elif hue == "green" or hue == "G":
      icc, ic1, ic2, negative = 1, 0, 2, False
    elif hue == "cyan" or hue == "C":
      icc, ic1, ic2, negative = 0, 1, 2, True
    elif hue == "blue" or hue == "B":
      icc, ic1, ic2, negative = 2, 0, 1, False
    elif hue == "magenta" or hue == "M":
      icc, ic1, ic2, negative = 1, 0, 2, True
    else:
      raise ValueError(f"Error, unknowm hue '{hue}'.")
    image = self.clip() # Clip before reducing color noise.
    if negative: image = image.negative()
    if protection == "avgneutral":
      c = (image[ic1]+image[ic2])/2.
      image[icc] = np.minimum(image[icc], c)
    elif protection == "maxneutral":
      c = np.maximum(image[ic1], image[ic2])
      image[icc] = np.minimum(image[icc], c)
    elif protection == "addmask":
      c = np.minimum(1., image[ic1]+image[ic2])
      image[icc] *= (1.-mixing)+c*mixing
    elif protection == "maxmask":
      c = np.maximum(image[ic1], image[ic2])
      image[icc] *= (1.-mixing)+c*mixing
    else:
      raise ValueError(f"Error, unknown protection mode '{protection}'.")
    if negative: image = image.negative()
    if lightness:
      self.check_color_space("lRGB", "sRGB")
      image = image.lRGB()
      image = image.scale_pixels(image.luminance(), self.luminance())
      if self.colormodel == "sRGB": image = image.sRGB()
      difflight = image.lightness()-self.lightness()
      print(f"Maximum lightness difference = {abs(difflight).max()}.")
    return image
    