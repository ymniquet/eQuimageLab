# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.2 / 2024.12.28
# Sphinx OK.

"""Histogram stretch."""

import numpy as np

from . import stretchfunctions as stf

######################
# Stretch functions. #
######################

def ghs(image, lnD1, b, SYP, SPP = 0., HPP = 1.):
  """Apply a generalized hyperbolic stretch function to the input image.

  For details about generalized hyperbolic stretches, see: https://ghsastro.co.uk/.

  Args:
    image (numpy.ndarray): The input image.
    logD1 (float): The global stretch factor ln(D+1) (must be >= 0).
    b (float): The local stretch factor.
    SYP (float): The symmetry point (expected in [0, 1]).
    SPP (float, optional): The shadow protection point (default 0; expected in [0, SYP]).
    HPP (float, optional): The highlight protection point (default 1; expected in [SYP, 1]).
    inverse (bool): Return the inverse stretch function if True.

  Returns:
    numpy.ndarray: The stretched image.
  """
  return stf.ghyperbolic_stretch_function(image, lnD1, b, SYP, SPP, HPP, False)

def mts(image, midtone):
  """Apply a midtone stretch function to the input image.

  The midtone stretch function is defined as:

    f(x) = (midtone-1)*x/((2*midtone-1)*x-midtone)

  In particular, f(0) = 0, f(midtone) = 0.5 and f(1) = 1.

  Args:
    image (numpy.ndarray): The input image.
    midtone (float): The midtone level (expected in ]0, 1[).

  Returns:
    numpy.ndarray: The stretched image.
  """
  return stf.midtone_stretch_function(image, midtone, False)

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def set_black_point(self, shadow, channels = "", trans = True):
    """Set the black (shadow) level in selected channels of the image.

    The selected channels are clipped below shadow and linearly stretched to map [shadow, 1] onto [0, 1].
    The output, stretched image channels therefore fit in the [0, infty[ range.

    Args:
      shadow (float): The black (shadow) level (expected < 1).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    if shadow > .9999: raise ValuerError("Error, shadow must be <= 0.9999.")
    output = self.apply_channels(lambda channel: stf.shadow_stretch_function(channel, shadow), channels, trans = trans)
    if trans: output.trans.xticks = [shadow]
    return output

  def set_shadow_highlight(self, shadow, highlight, channels = "", trans = True):
    """Set shadow and highlight levels in selected channels of the image.

    The selected channels are clipped below shadow and above highlight and linearly stretched
    to map [shadow, highlight] onto [0, 1].
    The output, stretched channels levels therefore fit in the [0, 1] range.

    Args:
      shadow (float): The shadow level (expected < 1).
      highlight (float): The highlight level (expected > shadow).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    if shadow > .9999: raise ValuerError("Error, shadow must be <= 0.9999.")
    if highlight-shadow < .0001: raise ValuerError("Error, highlight-shadow must be >= 0.0001.")
    output = self.apply_channels(lambda channel: stf.shadow_highlight_stretch_function(channel, shadow, highlight), channels, trans = trans)
    if trans: output.trans.xticks = [shadow, highlight]
    return output

  def set_dynamic_range(self, fr, to, channels = "", trans = True):
    """Set the dynamic range of selected channels of the image.

    The selected channels are linearly stretched to map [fr[0], fr[1]] onto [to[0], to[1]],
    and clipped in the [to[0], to[1]] range.

    Args:
      fr (a tuple or list of two floats such that fr[1] > fr[0]): The input range.
      to (a tuple or list of two floats such that to[1] > to[0]): The output range.
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    if fr[1]-fr[0] < 0.0001: raise ValueError("Error, fr[1]-fr[0] must be >= 0.0001 !")
    if to[1]-to[0] < 0.0001: raise ValueError("Error, to[1]-to[0] must be >= 0.0001 !")
    return self.apply_channels(lambda channel: stf.dynamic_range_stretch_function(channel, fr, to), channels, trans = trans)

  def asinh_stretch(self, D, SYP = 0., SPP = 0., HPP = 1., inverse = False, channels = "", trans = True):
    """Apply a (generalized) arcsinh stretch to selected channels of the image.

    The generalized arcsinh stretch function f is applied to the selected channels:

      - f(x) = b1*x when x <= SPP,
      - f(x) = a2+b2*arcsinh(-D*(x-SYP)) when SPP <= x <= SYP,
      - f(x) = a3+b3*arcsinh( D*(x-SYP)) when SYP <= x <= HPP,
      - f(x) = a4+b4*x when x >= HPP.

    The coefficients a and b are computed so that f is continuous and derivable.
    SYP is the "symmetry point"; SPP is the "shadow protection point" and HPP is
    the "highlight protection point". They can tuned to protect contrast in the
    low and high brightness areas, respectively.

    f(x) falls back to the "standard" arcsinh stretch function:

      f(x) = arcsinh(D*x)/arcsinh(D)

    when SPP = SYP = 0 and HPP = 1.

    For details about generalized hyperbolic stretches, see: https://ghsastro.co.uk/.

    Args:
      D (float): The stretch factor (must be >= 0).
      SYP (float): The symmetry point (expected in [0, 1]).
      SPP (float, optional): The shadow protection point (default 0, must be <= SYP).
      HPP (float, optional): The highlight protection point (default 1, must be >= SYP).
      inverse (bool, optional): Return the inverse stretch if True (default False).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The stretched image.
    """
    if D < 0.: raise ValueError("Error, D must be >= 0.")
    if SPP > SYP:
      SPP = SYP
      print("Warning, changed SPP = SYP !")
    if HPP < SYP:
      HPP = SYP
      print("Warning, changed HPP = SYP !")
    output =  self.apply_channels(lambda channel: stf.gasinh_stretch_function(channel, D, SYP, SPP, HPP, inverse), channels, trans = trans)
    if trans: output.trans.xticks = [SPP, SYP, HPP]
    return output

  def ghyperbolic_stretch(self, lnD1, b, SYP, SPP = 0., HPP = 1., inverse = False, channels = "", trans = True):
    """Apply a generalized hyperbolic stretch to selected channels of the image.

    For details about generalized hyperbolic stretches, see: https://ghsastro.co.uk/.

    Args:
      logD1 (float): The global stretch factor ln(D+1) (must be >= 0).
      b (float): The local stretch factor.
      SYP (float): The symmetry point (expected in [0, 1]).
      SPP (float, optional): The shadow protection point (default 0, must be <= SYP).
      HPP (float, optional): The highlight protection point (default 1, must be >= SYP).
      inverse (bool, optional): Return the inverse stretch if True (default False).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      numpy.ndarray: The stretched image.
    """
    if lnD1 < 0.: raise ValueError("Error, lnD1 must be >= 0.")
    if SPP > SYP:
      SPP = SYP
      print("Warning, changed SPP = SYP !")
    if HPP < SYP:
      HPP = SYP
      print("Warning, changed HPP = SYP !")
    output = self.apply_channels(lambda channel: stf.ghyperbolic_stretch_function(channel, lnD1, b, SYP, SPP, HPP, inverse), channels, trans = trans)
    if trans: output.trans.xticks = [SPP, SYP, HPP]
    return output

  def gamma_stretch(self, gamma, channels = "", trans = True):
    """Apply a power law stretch (gamma correction) to selected channels of the image.

    The gamma stretch function f is applied to the selected channels:

      f(x) = x**gamma

    This function clips the selected channels below 0 before stretching.

    Args:
      gamma (float): The stretch exponent (must be > 0).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The stretched image.
    """
    if gamma <= 0.: raise ValueError("Error, gamma must be > 0.")
    return self.apply_channels(lambda channel: stf.gamma_stretch_function(channel, gamma), channels, trans = trans)

  def midtone_stretch(self, midtone, inverse = False, channels = "", trans = True):
    """Apply a midtone stretch to selected channels of the image.

    The midtone stretch function f is applied to the selected channels:

      f(x) = (midtone-1)*x/((2*midtone-1)*x-midtone)

    In particular, f(0) = 0, f(midtone) = 0.5 and f(1) = 1.

    Args:
      midtone (float): The midtone level (expected in ]0, 1[).
      inverse (bool, optional): Return the inverse stretch if True (default False).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The stretched image.
    """
    if midtone < .0001 or midtone >= .9999: raise ValueError("Error, midtone must be >= 0.0001 and <= 0.9999.")
    output = self.apply_channels(lambda channel: stf.midtone_stretch_function(channel, midtone, inverse), channels, trans = trans)
    if trans: output.trans.xticks = [midtone]
    return output

  def set_midtone_levels(self, midtone, shadow = 0., highlight = 1., low = 0., high = 1., channels = "", trans = True):
    """Set shadow, midtone, highlight, low and high levels in selected channels of the image.

    This method:
      1) Clips the selected channels in the [shadow, highlight] range and maps [shadow, highlight] to [0, 1].
      2) Applies the midtone stretch function f(x) = (m-1)*x/((2*m-1)*x-m),
         with m = (midtone-shadow)/(highlight-shadow) the remapped midtone.
      3) Maps [low, high] to [0, 1] and clips the output data in the [0, 1] range.

    Args:
      midtone (float): The input midtone level (expected in ]0, 1[).
      shadow (float, optional): The input shadow level (default 0; expected < midtone).
      highlight (float, optional): The input highlight level (default 1; expected > midtone).
      low (float, optional): The "low" output level (default 0; expected <= 0).
      high (float, optional): The "high" output level (default 1; expected >= 1).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The stretched image.
    """
    if midtone < .0001 or midtone > .9999: raise ValueError("Error, midtone must be >= 0.0001 and <= 0.9999.")
    if midtone-shadow < .0001: raise ValueError("Error, midtone-shadow must be >= 0.0001.")
    if highlight-midtone < .0001: raise ValueError("Error, highlight-midtone must be >= 0.0001.")
    if low > 0.:
      low = 0.
      print("Warning, changed low = 0.")
    if high < 1.:
      high = 1.
      print("Warning, changed high = 1.")
    output = self.apply_channels(lambda channel: stf.midtone_levels_stretch_function(channel, shadow, midtone, highlight, low, high), channels, trans = trans)
    if trans: output.trans.xticks = [shadow, midtone, highlight]
    return output

  def rational_stretch(self, D, SYP = 0., SPP = 0., HPP = 1., inverse = False, channels = "", trans = True):
    """Apply a rational stretch to selected channels of the image.

    The rational stretch function f is applied to the selected channels:

      - f(x) = b1*x when x <= SPP,
      - f(x) = a2+(b2*x+c2)/(d2*x+e2) when SPP <= x <= SYP,
      - f(x) = a3+(b3*x+c3)/(d3*x+e3) when SYP <= x <= HPP,
      - f(x) = a4+b4*x when x >= HPP.

    The coefficients a, b, c, d and e are computed so that f is continuous and derivable.
    SYP is the "symmetry point"; SPP is the "shadow protection point" and HPP is
    the "highlight protection point". They can tuned to protect contrast in the
    low and high brightness areas, respectively. The strength of the stretch is
    controlled by the input parameter D that sets the slope at x = SYP.

    f(x) falls back to the midtone stretch function:

      f(x) = (midtone-1)*x/((2*midtone-1)*x-midtone) with midtone = 1/(2*(D+1))

    when SPP = SYP = 0 and HPP = 1.

    Args:
      D (float): The stretch factor (must be >= 0).
      SYP (float): The symmetry point (expected in [0, 1]).
      SPP (float, optional): The shadow protection point (default 0, must be <= SYP).
      HPP (float, optional): The highlight protection point (default 1, must be >= SYP).
      inverse (bool, optional): Return the inverse stretch if True (default False).
      channels (str, optional): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the
          operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, with highlights protection by blending
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The stretched image.
    """
    if D < 0.: raise ValueError("Error, D must be >= 0.")
    if SPP > SYP:
      SPP = SYP
      print("Warning, changed SPP = SYP !")
    if HPP < SYP:
      HPP = SYP
      print("Warning, changed HPP = SYP !")
    output = self.apply_channels(lambda channel: stf.rational_stretch_function(channel, D, SYP, SPP, HPP, inverse), channels, trans = trans)
    if trans: output.trans.xticks = [SPP, SYP, HPP]
    return output
