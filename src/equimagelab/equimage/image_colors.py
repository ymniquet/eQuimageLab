# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Color management."""

import numpy as np

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
  def RGB_balance(self, red = 1., green = 1., blue = 1.,):
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
    return 1.-self.image()

  # TESTED.
  def gray_scale(self, channel = "Y"):
    """Convert the selected channel of a RGB image into a gray scale image.
       channel can be "V" (value), "L" (luma) or "Y" (luminance)."""
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

  ##########################
  # Highlights protection. #
  ##########################

  def protect_highlights(self, luma = None):
    """Normalize out-of-range pixels with HSV value > 1 by adjusting the saturation at constant luma.
       'luma' is the luma of the image, if available (if None, the luma is recomputed on the fly).
       Warning: This method aims at protecting the highlights from overflowing when stretching the luma.
       It assumes that the luma remains <= 1 even though some pixels have HSV value > 1."""
    self.check_color_model("RGB")
    if luma is None: luma = self.luma() # Original luma.
    newimage = self.copy()
    newimage /= np.maximum(image.max(axis = 0), 1.) # Rescale maximum HSV value to 1.
    newluma = newimage.luma() # Updated luma.
    # Scale the saturation.
    # Note: The following implementation is failsafe when newluma -> 1 (in which case luma is also 1 in principle),
    # at the cost of a small error.
    fs = ((1.-luma)+params.IMGTOL)/((1.-newluma)+params.IMGTOL)
    return 1.-fs*(1.-newimage)
