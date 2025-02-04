# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.2.0 / 2025.02.02
# Sphinx OK.

"""Color management."""

import numpy as np
import scipy.interpolate as spint
from scipy.interpolate import Akima1DInterpolator

from . import params
from . import helpers
from . import image_colorspaces as colorspaces

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  ##################
  # Image queries. #
  ##################

  def is_grayscale_RGB(self):
    """Return True if a RGB image is actually a grayscale (same RGB channels), False otherwise.

    Returns:
      bool: True if the RGB image is a grayscale (same RGB channels), False otherwise.
    """
    self.check_color_model("RGB")
    epsilon = helpers.fpepsilon(self.dtype)
    return np.all(abs(self.image[1]-self.image[0]) < epsilon) and np.all(abs(self.image[2]-self.image[0]) < epsilon)

  ############################
  # Gray scales & negatives. #
  ############################

  def negative(self):
    """Return the negative of a RGB or grayscale image.

    Returns:
      Image: The negative of the image.
    """
    self.check_color_model("RGB", "gray")
    return 1.-self

  def grayscale(self, channel = "Y", RGB = False):
    """Convert the selected channel of a RGB image into a grayscale image.

    Args:
      channel: The converted channel ("V" for the HSV value, "L" for the luma, "Y" or "L*" for the
        luminance/lightness). Namely, the output grayscale image has the same value/luma/luminance
        and lightness as the original RGB image.
      RGB (bool, optional): If True, return the grayscale as a RGB image (with identical R/G/B
        channels). If False (default), return the grayscale as a single channel image.

    Returns:
      Image: The grayscale image.
    """
    self.check_color_model("RGB")
    if channel == "V":
      grayscale = self.HSV_value()
    elif channel == "L":
      grayscale = self.luma()
    elif channel == "Y" or channel == "L*":
      grayscale = self.luminance()
      if self.colorspace == "sRGB":
        grayscale = colorspaces.lRGB_to_sRGB(grayscale) # Preserve luminance & lightness in the sRGB color space.
    else:
      raise ValueError(f"Error, invalid channel '{channel}' (must be 'V', 'L' or 'Y'/'L*').")
    if RGB:
      return self.newImage(np.repeat(grayscale[np.newaxis, :, :], 3, axis = 0))
    else:
      return self.newImage(grayscale, colormodel = "gray")

  ##########################
  # Color transformations. #
  ##########################

  def color_balance(self, red = 1., green = 1., blue = 1.):
    """Adjust the color balance of a RGB image.

    Scales the red/green/blue channels by the input multipliers.

    Args:
      red (float, optional): The multiplier for the red channel (default 1).
      green (float, optional): The multiplier for the green channel (default 1).
      blue (float, optional): The multiplier for the blue channel (default 1).

    Returns:
      Image: The processed image.
    """
    self.check_color_model("RGB")
    if red < 0.: raise ValueError("Error, red must be >= 0.")
    if green < 0.: raise ValueError("Error, green must be >= 0.")
    if blue < 0.: raise ValueError("Error, blue must be >= 0.")
    output = self.copy()
    if red   != 1.: output.image[0] *= red
    if green != 1.: output.image[1] *= green
    if blue  != 1.: output.image[2] *= blue
    return output

  def color_saturation(self, A = 0., mode = "midsat", interpolation = "akima", model = "HSV", lightness = False, trans = True, **kwargs):
    """Adjust color saturation.

    The image is converted (if needed) to the HSV or HSL color model and the color saturation S is transformed according
    to the 'mode' kwarg:

      - "addsat": Shift the saturation S ← S+delta.
      - "mulsat": Scale the saturation S ← S*(1+delta).
      - "midsat": Apply a midtone stretch function S ← f(S) = (m-1)S/((2m-1)S-m) with midtone m = (1-delta)/2.
        This function increases monotonously from f(0) = 0 to f(m) = 1/2 and f(1) = 1.

    The image is then converted back to the original color model after this operation.
    delta is expected in the [-1, 1] range (except in the "mulsat" mode, where it can be > 1). Whatever the saturation mode,
    delta = 0 leaves the image unchanged, delta > 0 saturates the colors, and delta < 0 turns the image into a gray scale.
    delta is first set for all hues (with the 'A' kwarg), then can be updated for the red ('R'), yellow ('Y'), green ('G'),
    cyan ('C'), blue ('B') and magenta ('M') hues by providing the corresponding kwarg. delta is interpolated for arbitrary
    hues using nearest neighbor, linear, cubic or akima spline interpolation, according to the 'interpolation' kwarg.

    Args:
      A (float, optional): The delta for all hues (default 0).
      R (float, optional): The red delta (default A).
      Y (float, optional): The yellow delta (default A).
      G (float, optional): The green delta (default A).
      C (float, optional): The cyan delta (default A).
      B (float, optional): The blue delta (default A).
      M (float, optional): The magenta delta (default A).
      mode (str, optional): The saturation mode ["addsat", "mulsat" or "midsat" (default)].
      model (str, optional): The color model for saturation ["HSV" (default) or "HSL"].
      interpolation (str, optional): The interpolation method for delta(hue):

        - "nearest": Nearest neighbor interpolation.
        - "linear": Linear spline interpolation.
        - "cubic": Cubic spline interpolation.
        - "akima": Akima spline interpolation (default).

      lightness (bool, optional): If True (default), preserve the CIE lightness L* of the original image.
        This will however decrease color saturation in the bright areas of the image. Available only for RGB images.
      trans(bool, optional): If True (default), embed the transormation in the output image as
        output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """

    def interpolate(hue, psat, interpolation):
      """Interpolate the saturation parameter psat[RYGCBM] for arbitrary hues."""
      if np.all(psat == psat[0]):
        return np.full_like(hue, psat[0]) # Short-cut if the saturation parameter is the same for RYGCBM.
      if interpolation == "nearest":
        hsat = np.linspace(0., 1., 7)
        psat = np.append(psat, psat[0]) # Enforce periodic boundary conditions.
        fsat = spint.interp1d(hsat, psat, kind = "nearest")
      elif interpolation == "linear" or interpolation == "cubic":
        hsat = np.linspace(0., 1., 7)
        psat = np.append(psat, psat[0]) # Enforce periodic boundary conditions.
        k = 3 if interpolation == "cubic" else 1
        tck = spint.splrep(hsat, psat, k = k, per = True)
        def fsat(x): spint.splev(x, tck)
      elif interpolation == "akima":
        hsat = np.linspace(-1./3., 4./3, 11)
        psat = np.concatenate(([psat[-2], psat[-1]], psat, [psat[0], psat[1], psat[2]])) # Enforce periodic boundary conditions.
        fsat = Akima1DInterpolator(hsat, psat)
      else:
        raise ValueError(f"Error, unknown interpolation method '{interpolation}'.")
      return fsat(hue)

    self.check_color_model("RGB", "HSV", "HSL")
    psat = np.empty(6)
    psat[0] = kwargs.pop("R", A)
    psat[1] = kwargs.pop("Y", A)
    psat[2] = kwargs.pop("G", A)
    psat[3] = kwargs.pop("C", A)
    psat[4] = kwargs.pop("B", A)
    psat[5] = kwargs.pop("M", A)
    if kwargs: print("Discarding extra keyword arguments in Image.color_saturation...")
    if model == "HSV":
      hsx = self.HSV()
    elif model == "HSL":
      hsx = self.HSL()
    else:
      raise ValueError("Error, model must be 'HSV' or 'HSL'.")
    hue = hsx.image[0]
    sat = hsx.image[1]
    delta = interpolate(hue, psat, interpolation)
    if mode == "addsat":
      sat += delta
    elif mode == "mulsat":
      sat *= 1.+delta
    elif mode == "midsat":
      midsat = np.clip(.5*(1.-delta), .005, .995)
      sat = (midsat-1.)*sat/((2.*midsat-1.)*sat-midsat)
    else:
      raise ValueError(f"Error, unknown saturation mode '{mode}.")
    hsx.image[1] = np.clip(sat, 0., 1.)
    output = hsx.convert(colormodel = self.colormodel)
    if lightness: output.set_channel("L*", self.lightness(), inplace = True)
    if trans:
      t = helpers.Container()
      t.type = "hue"
      t.input = self
      t.xm = np.linspace(0., 1., 7)
      t.ym = np.append(psat, psat[0])
      t.cm = ["red", "yellow", "green", "cyan", "blue", "magenta", "red"]
      t.x = np.linspace(0., 1., params.ntranslo)
      t.y = interpolate(t.x, psat, interpolation)
      t.ylabel = "\u0394"
      output.trans = t
    return output

  ##########################
  # Color noise reduction. #
  ##########################

  def SCNR(self, hue = "green", protection = "avgneutral", amount = 1., lightness = True):
    """Subtractive Chromatic Noise Reduction of a given hue of a RGB image.

    The input hue is reduced according to the 'protection' kwarg. For the green hue for example,

      - G ← min(G, C) with C = (R+B)/2 for average neutral protection (protection = "avgneutral").
      - G ← min(G, C) with C = max(R, B) for maximum neutral protection (protection = "maxneutral").
      - G ← G*[(1-A)+C*A] with C = (R+B)/2 for additive mask protection (protection = "addmask").
      - G ← G*[(1-A)+C*A] with C = max(R, B) for maximum mask protection (protection = "maxmask").

    The parameter A in [0, 1] controls the strength of the mask protection.

    Args:
      hue (str, optional): The hue to be reduced ["red" alias "R", "yellow" alias "Y", "green" alias "G" (default),
        "cyan" alias "C", "blue" alias "B", or "magenta" alias "M"].
      protection (str, optional): The protection mode ["avgneutral" (default), "maxneutral", "addmask" or "maxmask"].
      amount (float, optional): The parameter A for mask protection (protection = "addmask" or "maxmask", default 1).
      lightness (bool, optional): If True (default), preserve the CIE lightness L* of the original image.

    Returns:
      Image: The processed image.
    """
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
      raise ValueError(f"Error, unknown hue '{hue}'.")
    image = np.clip(self.image, 0., 1.) # Clip before reducing color noise.
    if negative: image = 1.-image
    if protection == "avgneutral":
      c = (image[ic1]+image[ic2])/2.
      image[icc] = np.minimum(image[icc], c)
    elif protection == "maxneutral":
      c = np.maximum(image[ic1], image[ic2])
      image[icc] = np.minimum(image[icc], c)
    elif protection == "addmask":
      c = np.minimum(1., image[ic1]+image[ic2])
      image[icc] *= (1.-amount)+c*amount
    elif protection == "maxmask":
      c = np.maximum(image[ic1], image[ic2])
      image[icc] *= (1.-amount)+c*amount
    else:
      raise ValueError(f"Error, unknown protection mode '{protection}'.")
    if negative: image = 1.-image
    output = self.newImage(image)
    if lightness: output.set_channel("L*", self.lightness(), inplace = True)
    return output
