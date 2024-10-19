# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Histogram stretch."""

import numpy as np

from . import stretchfunctions as stf

######################
# Stretch functions. #
######################

def mts(image, midtone):
  """Apply a midtone stretch to the input image, with midtone 'midtone'."""
  return (midtone-1.)*image/((2.*midtone-1.)*image-midtone)

def ghs(image, lnD1, b, SYP, SPP = 0., HPP = 1.):
  """Apply a generalized hyperbolic stretch to the input image, with global
     strength 'lnD1' = ln(D+1), local strength 'b', symmetry point 'SYP',
     shadow protection point 'SPP', and highlight protection point 'HPP'."""
  return stf.ghyperbolic_stretch_function(image, (lnD1, b, SYP, SPP, HPP, False))

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  def black_point(self, shadow, channels = ""):
    """Set black level 'shadow' in selected 'channels' of the image.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if shadow < 0.:
      print("Warning, changed shadow = 0 !")
      shadow = 0.
    if shadow > .9:
      print("Warning, changed shadow = 0.9 !")
      shadow = .9
    return self.apply_channels(lambda channel: stf.black_point_stretch_function(channel, (shadow, )), channels)

  def asinh_stretch(self, stretch, shadow = 0., channels = ""):
    """Apply an arcsinh stretch to selected 'channels' of the image,
       with strength 'stretch' and black level 'shadow'.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if stretch < 0.: raise ValueError("Error, stretch must be >= 0.")
    if shadow < 0.:
      print("Warning, changed shadow = 0 !")
      shadow = 0.
    if shadow > .9:
      print("Warning, changed shadow = 0.9 !")
      shadow = .9
    return self.apply_channels(lambda channel: stf.asinh_stretch_function(channel, (shadow, stretch)), channels)

  def ghyperbolic_stretch(self, lnD1, b, SYP, SPP = 0., HPP = 1., channels = "", inverse = False):
    """Apply a generalized hyperbolic stretch to selected 'channels' of the image,
       with global strength 'lnD1' = ln(D+1), local strength 'b', symmetry point 'SYP',
       shadow protection point 'SPP', and highlight protection point 'HPP'.
       Inverse the transformation if 'inverse' is True.
       See: https://ghsastro.co.uk/.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if lnD1 < 0.: raise ValueError("Error, lnD1 must be >= 0.")
    if SYP < 0.:
      print("Warning, changed SYP = 0 !")
      SYP = 0.
    if SYP > 1.:
      print("Warning, changed SYP = 1 !")
      SYP = 1.
    if SPP > SYP:
      print("Warning, changed SPP = SYP !")
      SPP = SYP
    if HPP < SYP:
      print("Warning, changed HPP = SYP !")
      HPP = SYP
    return self.apply_channels(lambda channel: stf.ghyperbolic_stretch_function(channel, (lnD1, b, SYP, SPP, HPP, inverse)), channels)

  def midtone_stretch(self, midtone, shadow = 0., highlight = 1., low = 0., high = 1., channels = ""):
    """Apply a midtone stretch to selected 'channels' of the image,
       with midtone 'midtone', black point 'shadow', highlight point 'highlight',
       low range 'low' and high range 'high'.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if midtone <= 0.: raise ValueError("Error, midtone must be >= 0.")
    if shadow < 0.:
      print("Warning, changed shadow = 0 !")
      shadow = 0.
    if shadow > .9:
      print("Warning, changed shadow = 0.9 !")
      shadow = .9
    if highlight < shadow+.01:
      print(f"Warning, changed highlight = {highlight:.3f} !")
      highlight = shadow+.01
    if low > 0.:
      print("Warning, changed low = 0.")
      low = 0.
    if high < 1.:
      print("Warning, changed high = 1.")
      high = 1.
    return self.apply_channels(lambda channel: stf.midtone_stretch_function(channel, (shadow, midtone, highlight, low, high)), channels)
