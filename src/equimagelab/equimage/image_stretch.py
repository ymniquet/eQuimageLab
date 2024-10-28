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
  return stf.midtone_stretch_function(image, (midtone,))

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

  def set_black_point(self, shadow, channels = ""):
    """Set black (shadow) level 'shadow' in selected 'channels' of the image.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if shadow < 0.:
      shadow = 0.
      print("Warning, changed shadow = 0 !")
    if shadow > .99:
      shadow = .99
      print("Warning, changed shadow = 0.99 !")
    return self.apply_channels(lambda channel: stf.shadow_stretch_function(channel, (shadow,)), channels)

  def clip_shadow_highlight(self, shadow, highlight, channels = ""):
    """Set black (shadow) level 'shadow' and highlight level 'highlight' in selected 'channels' of the image.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if shadow < 0.:
      shadow = 0.
      print("Warning, changed shadow = 0 !")
    if shadow > .99:
      shadow = .99
      print("Warning, changed shadow = 0.99 !")
    if highlight > 1.:
      highlight = 1.
      print("Warning, changed highlight = 1 !")
    if highlight < shadow+.01:
      highlight = shadow+.01
      print(f"Warning, changed highlight = {highlight:.3f} !")
    return self.apply_channels(lambda channel: stf.shadow_highlight_stretch_function(channel, (shadow, highlight)), channels)

  def set_dynamic_range(self, fr, to, channels = ""):
    """Map (fr[0], fr[1]) to (to[0], to[1]) in selected 'channels' of the image,
       and clip the output in the [to[0], to[1]] range.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if fr[0] >= fr[1]:
      raise ValueError("Error, fr[0] >= fr[1] !")
    if to[0] >= to[1]:
      raise ValueError("Error, to[0] >= to[1] !")
    return self.apply_channels(lambda channel: stf.dynamic_range_stretch_function(channel, (fr, to)), channels)

  def asinh_stretch(self, stretch, channels = ""):
    """Apply an arcsinh stretch with strength 'stretch' to selected 'channels' of the image.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if stretch < 0.: raise ValueError("Error, stretch must be >= 0.")
    return self.apply_channels(lambda channel: stf.asinh_stretch_function(channel, (stretch,)), channels)

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
      SYP = 0.
      print("Warning, changed SYP = 0 !")
    if SYP > 1.:
      SYP = 1.
      print("Warning, changed SYP = 1 !")
    if SPP > SYP:
      SPP = SYP
      print("Warning, changed SPP = SYP !")
    if HPP < SYP:
      HPP = SYP
      print("Warning, changed HPP = SYP !")
    return self.apply_channels(lambda channel: stf.ghyperbolic_stretch_function(channel, (lnD1, b, SYP, SPP, HPP, inverse)), channels)

  def midtone_stretch(self, midtone, channels = ""):
    """Apply a midtone stretch with midtone 'midtone' to selected 'channels' of the image,
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if midtone <= 0. or midtone >= 1.: raise ValueError("Error, midtone must be > 0 and < 1.")
    return self.apply_channels(lambda channel: stf.midtone_stretch_function(channel, (midtone,)), channels)

  def gamma_stretch(self, gamma):
    """Apply power law stretch with exponent 'gamma' to selected 'channels' of the image.
       The 'channels' can be:
         - An empty string: Apply the operation to all channels (RGB and HSV images).
         - "L": Apply the operation to the luma (RGB images).
         - "Lp": Apply the operation to the luma, with highlights protection.
                (after the operation, the out-of-range pixels are desaturated at constant luma).
         - "V": Apply the operation to the HSV value (RGB and HSV images).
         - "S": Apply the operation to the HSV saturation (RGB and HSV images).
         - A combination of "R", "G", "B": Apply the operation to the R/G/B channels (RGB images)."""
    if gamma < 0.:
      raise ValueError("Error, gamma must be >= 0.")
    return self.apply_channels(lambda channel: stf.gamma_stretch_function(channel, (gamma,)), channels)

  def adjust_midtone(self, midtone, shadow = 0., highlight = 1., low = 0., high = 1., channels = ""):
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
    if midtone <= 0. or midtone >= 1.: raise ValueError("Error, midtone must be > 0 and < 1.")
    if shadow < 0.:
      shadow = 0.
      print("Warning, changed shadow = 0 !")
    if shadow > .99:
      shadow = .99
      print("Warning, changed shadow = 0.99 !")
    if highlight > 1.:
      highlight = 1.
      print("Warning, changed highlight = 1 !")
    if highlight < shadow+.01:
      highlight = shadow+.01
      print(f"Warning, changed highlight = {highlight:.3f} !")
    if low > 0.:
      low = 0.
      print("Warning, changed low = 0.")
    if high < 1.:
      high = 1.
      print("Warning, changed high = 1.")
    return self.apply_channels(lambda channel: stf.adjust_midtone_function(channel, (shadow, midtone, highlight, low, high)), channels)
