# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.0 / 2025.03.08
# Sphinx OK.

"""Color management."""

import numpy as np
import scipy.interpolate as spint
from scipy.interpolate import Akima1DInterpolator

from . import params
from . import helpers
from . import image_colorspaces as colorspaces

def parse_hue_kwargs(D, kwargs):
  """Parse hue keywords in the kwargs.

  This function looks for the keywords 'R' (red, hue H = 0), 'Y' (yellow, H = 1/6), 'G' (green,
  H = 1/3), 'C' (cyan, H = 1/2), 'B' (blue, H = 2/3) and 'M' (magenta, H = 5/6) in the kwargs and
  returns the grid of H's and the corresponding values of the kwargs as numpy arrays. Whenever a
  keyword is missing, its value is replaced by the default, D.
  Additional points may be inserted in the grid by providing the keywords 'RY' (red-yellow, H = 1/12),
  'YG' (yellow-green, H = 1/4), 'GC' (green-cyan, H = 5/12), 'CB' (cyan-blue, H = 7/12), 'BM' (blue-
  magenta, H = 3/4) and 'MR' (magenta-red, H = 11/12).

  Note:
    Used by Image.HSX_color_saturation, Image.CIE_chroma_saturation, Image.rotate_HSX_hue
    and Image.rotate_CIE_hue.

  Args:
    D (float): The default value for the R/Y/G/C/B/M hues.
    kwargs (dict): The dictionary of kwargs.

  Returns:
    The grid of hues (numpy.ndarray), the corresponding keyword values (numpy.ndarray), and the
    curated kwargs (with the used keys deleted).
  """
  R = kwargs.pop("R", D)
  Y = kwargs.pop("Y", D)
  G = kwargs.pop("G", D)
  C = kwargs.pop("C", D)
  B = kwargs.pop("B", D)
  M = kwargs.pop("M", D)
  hgrid = [0., 1./6., 2./6., 3./6., 4./6., 5./6., 1.] # Enforce periodic boundary conditions.
  value = [R, Y, G, C, B, M, R]
  if (RY := kwargs.pop("RY", None)) is not None:
    hgrid.append(1./12.)
    value.append(RY)
  if (YG := kwargs.pop("YG", None)) is not None:
    hgrid.append(3./12.)
    value.append(YG)
  if (GC := kwargs.pop("GC", None)) is not None:
    hgrid.append(5./12.)
    value.append(GC)
  if (CB := kwargs.pop("CB", None)) is not None:
    hgrid.append(7./12.)
    value.append(CB)
  if (BM := kwargs.pop("BM", None)) is not None:
    print(BM)
    hgrid.append(9./12.)
    value.append(BM)
  if (MR := kwargs.pop("MR", None)) is not None:
    hgrid.append(11./12.)
    value.append(MR)
  hgrid = np.array(hgrid)
  value = np.array(value)
  idx = np.argsort(hgrid)
  return hgrid[idx], value[idx], kwargs

def interpolate_hue(hue, hgrid, param, interpolation):
  """Interpolate a parameter param defined on a grid of hues to arbitrary hues.

  Note:
    Used by Image.HSX_color_saturation, Image.CIE_chroma_saturation, Image.rotate_HSX_hue
    and Image.rotate_CIE_hue.

  Args:
    hue (numpy.ndarray): The hues at which the parameter must be interpolated.
    hgrid (numpy.ndarray): The grid of hues on which the parameter is defined.
    param (numpy.ndarray): The parameter on the grid.
    interpolation (str, optional): The interpolation method:

      - "nearest": Nearest neighbor interpolation.
      - "linear": Linear spline interpolation.
      - "cubic": Cubic spline interpolation.
      - "akima": Akima spline interpolation (default).

  Returns:
    numpy.ndarray: The parameter interpolated for all input hues.
  """
  if np.all(param == param[0]):
    return np.full_like(hue, param[0]) # Short-cut if the parameter is the same for all points of the grid.
  if interpolation == "nearest":
    finter = spint.interp1d(hgrid, param, kind = "nearest")
  elif interpolation == "linear" or interpolation == "cubic":
    k = 3 if interpolation == "cubic" else 1
    tck = spint.splrep(hgrid, param, k = k, per = True)
    def finter(x): spint.splev(x, tck)
  elif interpolation == "akima":
    hgrid = np.concatenate(([hgrid[-3]-1., hgrid[-2]-1.], hgrid, [hgrid[1]+1., hgrid[2]+1.])) # Enforce periodic boundary conditions.
    param = np.concatenate(([param[-3], param[-2]], param, [param[1], param[2]]))
    finter = Akima1DInterpolator(hgrid, param)
  else:
    raise ValueError(f"Error, unknown interpolation method '{interpolation}'.")
  return finter(hue)

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

  def grayscale(self, channel = "L*", RGB = False):
    """Convert the selected channel of a RGB image into a grayscale image.

    Args:
      channel: The converted channel ("V" for the HSV value, "L'" for HSL lightness, "L" for the
        luma, "Y" or "L*" for the luminance/lightness). Namely, the output grayscale image has the
        same value/luma/luminance and lightness as the original RGB image.
      RGB (bool, optional): If True, return the grayscale as a RGB image (with identical R/G/B
        channels). If False (default), return the grayscale as a single channel image.

    Returns:
      Image: The grayscale image.
    """
    self.check_color_model("RGB")
    if channel == "V":
      grayscale = self.HSV_value()
    elif channel == "L'":
      grayscale = self.HSL_lightness()
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

  def RGB_color_balance(self, red = 1., green = 1., blue = 1):
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

  def mix_RGB(self, M):
    """Mix RGB channels.

    Transforms each pixel P = (R, G, B) of the image into M@P, with M a 3x3 mixing matrix.

    Args:
      M (numpy.ndarray): The mixing matrix.

    Returns:
      Image: The processed image.
    """
    self.check_color_model("RGB")
    return self.newImage(np.tensordot(np.asarray(M, dtype = self.dtype), self.image, axes = 1))

  def set_color_temperature(self, T, T0 = 6650., lightness = False):
    """Adjust the color temperature of a RGB image.

    Adjusts the color balance assuming that the scene is (or is lit by) a black body source whose
    temperature is changed from T0 (default 6650K) to T.
    Setting T < T0 casts a red tint on the image, while setting T > T0 casts a blue tint.
    This is not a rigorous transformation and is intended for "cosmetic" purposes.
    The colors are balanced in the linear RGB color space.

    Args:
      T (float): The target temperature between 1000K and 40000K.
      T0 (float, optional): The initial temperature between 1000K and 40000K (default 6650K).
      lightness (bool, optional): If True, preserve the lightness L* of the original image.
        Note that this may result in some out-of-range pixels. Default is False.

    Returns:
      Image: The processed image.
    """

    def RGB_multipliers(T):
      """Return RGB multipliers for a given temperature T.

      These are fits to the data of:
      http://www.vendian.org/mncharity/dir3/blackbody/
      http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html

      See equimagelab/misc/CTdata.py for the fits.

      Args:
        T (float): The temperature (K).

      Returns:
        float: The red, green, blue multipliers for temperature T.
      """

      def fitfunction(T, a, b, c, n):
        """Fit function for the RGB multipliers."""
        Tr = (T-6650.)/10000.
        return max(0., 1.+a*Tr+b*Tr**2+c*Tr**3)**n

      if T < 1000. or T > 40000.:
        raise ValueError("Error, the temperature must range between 1000K and 40000K.")
      if T < 6650.:
        red = 1.
        green = .955*fitfunction(T, 0.6328063016856568, -0.7554188937494166, 1.7905208976761535, 1.2444903530611828)
        blue = fitfunction(T, 1.2773641159953395, -1.0621459476222992, 1.1594787131985838, 1.7962669751286322)
      else:
        red = fitfunction(T, 3.7874055198445142, -1.2956261274808025, 0.16818980687443094, -0.6136402157590004)
        green = .955*fitfunction(T, 3.490058031650525, -1.1425871791667106, 0.14401832008759521, -0.42381907464549073)
        blue = 1.
      return red, green, blue

    self.check_color_space("lRGB", "sRGB")
    red0, green0, blue0 = RGB_multipliers(T0)
    red , green , blue  = RGB_multipliers(T)
    red /= red0 ; green /= green0 ; blue /= blue0
    maxi = np.max([red, green, blue])
    red /= maxi ; green /= maxi ; blue /= maxi
    print(f"Red multiplier = {red:.3f}.")
    print(f"Green multiplier = {green:.3f}.")
    print(f"Blue multiplier = {blue:.3f}.")
    image = self.convert(colorspace = "lRGB", colormodel = "RGB", copy = False)
    balanced = image.RGB_color_balance(red, green, blue)
    if lightness: balanced.set_channel("L*sh", self.lightness(), inplace = True)
    return balanced.convert(colorspace = self.colorspace, colormodel = self.colormodel, copy = False)

  def HSX_color_saturation(self, D = 0., mode = "midsat", colormodel = "HSV", colorspace = None, interpolation = "akima", lightness = False, trans = True, **kwargs):
    """Adjust color saturation in the HSV or HSL color models.

    The image is converted (if needed) to the HSV or HSL color model, then the color saturation S is
    transformed according to the 'mode' kwarg:

      - "addsat": Shift the saturation S ← S+delta.
      - "mulsat": Scale the saturation S ← S*(1+delta).
      - "midsat": Apply a midtone stretch function S ← f(S) = (m-1)S/((2m-1)S-m) with midtone m = (1-delta)/2.
        This function increases monotonously from f(0) = 0 to f(m) = 1/2 and f(1) = 1, and thus leaves
        the saturation of the least/most saturated pixels unchanged.

    The image is then converted back to the original color model after this operation.
    delta is expected to be > -1, and to be < 1 in the "midsat" mode. Whatever the mode, delta = 0
    leaves the image unchanged, delta > 0 saturates the colors, and delta < 0 turns the image into
    a gray scale. delta is set for the red ('R'), yellow ('Y'), green ('G'), cyan ('C'), blue ('B')
    and magenta ('M') hues by the corresponding kwarg (delta = D if missing). It is interpolated
    for arbitrary hues using nearest neighbor, linear, cubic or akima spline interpolation according
    to the 'interpolation' kwarg. Midpoint deltas may also be specified for finer interpolation by
    providing the kwargs 'RY' (red-yellow), 'YG' (yellow-green), 'GC' (green-cyan), 'CB' (cyan-blue),
    'BM' (blue-magenta) and 'MR' (magenta-red).

    See also:
      CIE_chroma_saturation

    Args:
      D (float, optional): The delta for all hues (default 0).
      R (float, optional): The red delta (default D).
      Y (float, optional): The yellow delta (default D).
      G (float, optional): The green delta (default D).
      C (float, optional): The cyan delta (default D).
      B (float, optional): The blue delta (default D).
      M (float, optional): The magenta delta (default D).
      mode (str, optional): The saturation mode ["addsat", "mulsat" or "midsat" (default)].
      colormodel (str, optional): The color model for saturation ["HSV" (default) or "HSL"].
      colorspace (str, optional): The color space for saturation ["lRGB", "sRGB", or None (default)
        to use the color space of the image].
      interpolation (str, optional): The interpolation method for delta(hue):

        - "nearest": Nearest neighbor interpolation.
        - "linear": Linear spline interpolation.
        - "cubic": Cubic spline interpolation.
        - "akima": Akima spline interpolation (default).

      lightness (bool, optional): If True, preserve the lightness L* of the original image.
        Note that this may result in some out-of-range pixels. Default is False.
      trans (bool, optional): If True (default), embed the transormation in the output image
        as output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    self.check_color_space("lRGB", "sRGB")
    hgrid, dgrid, kwargs = parse_hue_kwargs(D, kwargs)
    if kwargs: raise ValueError(f"Error, unknown keyword argument(s): {', '.join(kwargs.keys())}.")
    if colorspace is None:
      colorspace = self.colorspace
    elif colorspace not in ["lRGB", "sRGB"]:
      raise ValueError("Error, colorspace must be 'lRGB' or 'sRGB'.")
    if colormodel == "HSV":
      channel = "S"
    elif colormodel == "HSL":
      channel = "S'"
    else:
      raise ValueError("Error, colormodel must be 'HSV' or 'HSL'.")
    hsx = self.convert(colorspace = colorspace, colormodel = colormodel, copy = True)
    hue = hsx.image[0]
    sat = hsx.image[1]
    print(f"Before operation: min(saturation) = {np.min(sat):.5f}; median(saturation) = {np.median(sat):.5f}; max(saturation) = {np.max(sat):.5f}.")
    delta = interpolate_hue(hue, hgrid, dgrid, interpolation)
    if mode == "addsat":
      sat += delta
    elif mode == "mulsat":
      sat *= 1.+delta
    elif mode == "midsat":
      midsat = np.clip(.5*(1.-delta), .001, .999)
      sat = (midsat-1.)*sat/((2.*midsat-1.)*sat-midsat)
    else:
      raise ValueError(f"Error, unknown saturation mode '{mode}.")
    print(f"After  operation: min(saturation) = {np.min(sat):.5f}; median(saturation) = {np.median(sat):.5f}; max(saturation) = {np.max(sat):.5f}.")
    hsx.image[1] = np.clip(sat, 0., 1.)
    output = hsx.convert(colorspace = self.colorspace, colormodel = self.colormodel, copy = False)
    if lightness: output.set_channel("L*sh", self.lightness(), inplace = True)
    if trans:
      t = helpers.Container()
      t.type = "hue"
      t.input = self
      t.channels = channel if self.colorspace == colorspace else ""
      t.xm = np.linspace(0., 1., 7)
      t.ym = interpolate_hue(t.xm, hgrid, dgrid, interpolation)
      t.cm = ["red", "yellow", "green", "cyan", "blue", "magenta", "red"]
      t.x = np.linspace(0., 1., params.ntranslo)
      t.y = interpolate_hue(t.x, hgrid, dgrid, interpolation)
      t.ylabel = "\u0394"
      output.trans = t
    return output

  def CIE_chroma_saturation(self, D = 0., mode = "midsat", colormodel = "Lsh", interpolation = "akima", ref = None, trans = True, **kwargs):
    """Adjust color chroma or saturation in the CIELab or CIELuv color spaces.

    The image is converted (if needed) to the CIELab or CIELuv colorspace, then the CIELab chroma
    CS = c* = sqrt(a*^2+b*^2) (colormodel = "Lab"), or the CIELuv chroma CS = c* = sqrt(u*^2+v*^2)
    (colormodel = "Luv"), or the CIELuv saturation CS = s* = c*/L* (colormodel = "Lsh") is transformed
    according to the 'mode' kwarg:

      - "addsat": Shift the chroma/saturation CS ← CS+delta.
      - "mulsat": Scale the chroma/saturation CS ← CS*(1+delta).
      - "midsat": Apply a midtone stretch function CS ← f(CS) = (m-1)CS/((2m-1)CS/ref-m) with midtone m = (1-delta)/2.
        This function increases monotonously from f(0) = 0 to f(m*ref) = ref/2 and f(ref) = ref,
        where ref is a reference chroma/saturation (ref = max(CS) by default).

    The image is then converted back to the original color space and model after this operation.
    delta is expected to be > -1, and to be < 1 in the "midsat" mode. Whatever the mode, delta = 0
    leaves the image unchanged, delta > 0 saturates the colors, and delta < 0 turns the image into
    a gray scale. However, please keep in mind that the chroma/saturation in the CIELab/CIELuv color
    spaces is not bounded by 1 as it is in the lRGB and sRGB color spaces (HSV and HSL color models).
    The choice of the reference can, therefore, be critical in the "midsat" mode. In particular,
    pixels with chroma/saturation > ref get desaturated if delta > 0, and oversaturated if delta < 0
    (with a possible singularity at CS = -ref*(1-delta)/(2*delta)).
    delta is set for the red ('R'), yellow ('Y'), green ('G'), cyan ('C'), blue ('B') and magenta
    ('M') hues by the corresponding kwarg (delta = D if missing). It is interpolated for arbitrary
    hues using nearest neighbor, linear, cubic or akima spline interpolation according to the
    'interpolation' kwarg. Midpoint deltas may also be specified for finer interpolation by providing
    the kwargs 'RY' (red-yellow), 'YG' (yellow-green), 'GC' (green-cyan), 'CB' (cyan-blue), 'BM'
    (blue-magenta) and 'MR' (magenta-red).
    Contrary to the saturation of HSV or HSL images, chroma/saturation transformations in the CIELab
    and CIELuv color spaces preserve the lightness by design. They may, however, result in out-of-
    range RGB pixels (as not all points of of these color spaces correspond to physical RGB colors).

    Note:
      Chroma and saturation are related, but different quantities (s* = c*/L* in the CIELuv color space). There is no rigorous
      definition of saturation in the CIELab color space.

    See also:
      HSX_color_saturation

    Args:
      D (float, optional): The delta for all hues (default 0).
      R (float, optional): The red delta (default D).
      Y (float, optional): The yellow delta (default D).
      G (float, optional): The green delta (default D).
      C (float, optional): The cyan delta (default D).
      B (float, optional): The blue delta (default D).
      M (float, optional): The magenta delta (default D).
      mode (str, optional): The saturation mode ["addsat", "mulsat" or "midsat" (default)].
      colormodel (str, optional): The color model for saturation ["Lab", "Luv" or "Lsh" (default)].
      interpolation (str, optional): The interpolation method for delta(hue angle):

        - "nearest": Nearest neighbor interpolation.
        - "linear": Linear spline interpolation.
        - "cubic": Cubic spline interpolation.
        - "akima": Akima spline interpolation (default).

      ref (float, optional): The reference chroma/saturation for the "midsat" mode. If None,
        defaults to the maximum chroma/saturation of the input image.
      trans (bool, optional): If True (default), embed the transormation in the output image
        as output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    hgrid, dgrid, kwargs = parse_hue_kwargs(D, kwargs)
    if kwargs: raise ValueError(f"Error, unknown keyword argument(s): {', '.join(kwargs.keys())}.")
    if colormodel == "Lab":
      name = "chroma"
      channel = "c*" if self.colorspace == "CIELab" else ""
      CIE = self.convert(colorspace = "CIELab", colormodel = "Lch", copy = True)
    elif colormodel == "Luv":
      name = "chroma"
      channel = "c*" if self.colorspace == "CIELuv" else ""
      CIE = self.convert(colorspace = "CIELuv", colormodel = "Lch", copy = True)
    elif colormodel == "Lsh":
      name = "saturation"
      channel = "s*" if self.colorspace == "CIELuv" else ""
      CIE = self.convert(colorspace = "CIELuv", colormodel = "Lsh", copy = True)
    else:
      raise ValueError("Error, colormodel must be 'Lab' or 'Luv' or 'Lsh'.")
    hue = CIE.image[2]
    sat = CIE.image[1]
    maxsat = np.max(sat)
    print(f"Before operation: min({name}) = {np.min(sat):.5f}; median({name}) = {np.median(sat):.5f}; max({name}) = {maxsat:.5f}.")
    delta = interpolate_hue(hue, hgrid, dgrid, interpolation)
    if mode == "addsat":
      sat += delta
    elif mode == "mulsat":
      sat *= 1.+delta
    elif mode == "midsat":
      if ref is None: ref = maxsat
      midsat = np.clip(.5*(1.-delta), .001, .999)
      sat = (midsat-1.)*sat/((2.*midsat-1.)*sat/ref-midsat)
    else:
      raise ValueError(f"Error, unknown saturation mode '{mode}.")
    print(f"After  operation: min({name}) = {np.min(sat):.5f}; median({name}) = {np.median(sat):.5f}; max({name}) = {np.max(sat):.5f}.")
    CIE.image[1] = sat
    output = CIE.convert(colorspace = self.colorspace, colormodel = self.colormodel, copy = False)
    if trans:
      t = helpers.Container()
      t.type = "hue"
      t.input = self
      t.channels = channel
      t.xm = np.linspace(0., 1., 7)
      t.ym = interpolate_hue(t.xm, hgrid, dgrid, interpolation)
      t.cm = ["red", "yellow", "green", "cyan", "blue", "magenta", "red"]
      t.x = np.linspace(0., 1., params.ntranslo)
      t.y = interpolate_hue(t.x, hgrid, dgrid, interpolation)
      t.ylabel = "\u0394"
      output.trans = t
    return output

  def rotate_HSX_hue(self, D = 0., colorspace = None, interpolation = "akima", lightness = False, trans = True, **kwargs):
    """Rotate color hues in the HSV/HSL color models.

    The image is converted (if RGB) to the HSV color model, and the hue H is rotated:

      H ← (H+delta)%1.

    The image is then converted back to the original color model after this operation.
    delta is set for the original red ('R'), yellow ('Y'), green ('G'), cyan ('C'), blue ('B') and
    magenta ('M') hues by the corresponding kwarg (delta = D if missing). It is interpolated for
    arbitrary hues using nearest neighbor, linear, cubic or akima spline interpolation according to
    the 'interpolation' kwarg. Midpoint deltas may also be specified for finer interpolation by
    providing the kwargs 'RY' (red-yellow), 'YG' (yellow-green), 'GC' (green-cyan), 'CB' (cyan-blue),
    'BM' (blue-magenta) and 'MR' (magenta-red).

    Note:
      H(red) = 0, H(yellow) = 1/6, H(green) = 1/3, H(cyan) = 1/2, H(blue) = 2/3, and H(magenta) = 5/6.
      A uniform rotation D = 1/6 converts red → yellow, yellow → green, green → cyan, cyan → blue,
      blue → magenta, and magenta → red.
      A uniform rotation D = -1/6 converts red → magenta, yellow → red, green → yellow, cyan → green,
      blue → cyan, and magenta → blue.

    See also:
      rotate_CIE_hue

    Args:
      D (float, optional): The delta for all hues (default 0).
      R (float, optional): The red delta (default D).
      Y (float, optional): The yellow delta (default D).
      G (float, optional): The green delta (default D).
      C (float, optional): The cyan delta (default D).
      B (float, optional): The blue delta (default D).
      M (float, optional): The magenta delta (default D).
      colorspace (str, optional): The color space for saturation ["lRGB", "sRGB", or None (default)
        to use the color space of the image].
      interpolation (str, optional): The interpolation method for delta(hue):

        - "nearest": Nearest neighbor interpolation.
        - "linear": Linear spline interpolation.
        - "cubic": Cubic spline interpolation.
        - "akima": Akima spline interpolation (default).

      lightness (bool, optional): If True, preserve the lightness L* of the original image.
        Note that this may result in some out-of-range pixels. Default is False.
      trans (bool, optional): If True (default), embed the transormation in the output image
        as output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    self.check_color_space("lRGB", "sRGB")
    hgrid, dgrid, kwargs = parse_hue_kwargs(D, kwargs)
    if kwargs: raise ValueError(f"Error, unknown keyword argument(s): {', '.join(kwargs.keys())}.")
    if colorspace is None:
      colorspace = self.colorspace
    elif colorspace not in ["lRGB", "sRGB"]:
      raise ValueError("Error, colorspace must be 'lRGB' or 'sRGB'.")
    colormodel = self.colormodel if self.colormodel in ["HSV", "HSL"] else "HSV"
    hsx = self.convert(colorspace = colorspace, colormodel = colormodel, copy = True)
    hue = hsx.image[0]
    delta = interpolate_hue(hue, hgrid, dgrid, interpolation)
    hsx.image[0] = (hue+delta)%1.
    output = hsx.convert(colorspace = self.colorspace, colormodel = self.colormodel, copy = False)
    if lightness: output.set_channel("L*sh", self.lightness(), inplace = True)
    if trans:
      t = helpers.Container()
      t.type = "hue"
      t.input = self
      t.channels = "H" if self.colorspace == colorspace else ""
      t.xm = np.linspace(0., 1., 7)
      t.ym = interpolate_hue(t.xm, hgrid, dgrid, interpolation)
      t.cm = ["red", "yellow", "green", "cyan", "blue", "magenta", "red"]
      t.x = np.linspace(0., 1., params.ntranslo)
      t.y = interpolate_hue(t.x, hgrid, dgrid, interpolation)
      t.ylabel = "\u0394"
      output.trans = t
    return output

  def rotate_CIE_hue(self, D = 0., colorspace = "CIELab", interpolation = "akima", trans = True, **kwargs):
    """Rotate color hues in the CIELab or CIELuv color space.

    The image is converted (if needed) to the CIELab or CIELuv color space, and the reduced hue
    angle h* (within [0, 1]) is rotated:

      h* ← (h*+delta)%1.

    The image is then converted back to the original color model after this operation.
    delta is set for the original red ('R'), yellow ('Y'), green ('G'), cyan ('C'), blue ('B') and
    magenta ('M') hues by the corresponding kwarg (delta = D if missing). It is interpolated for
    arbitrary hues using nearest neighbor, linear, cubic or akima spline interpolation according to
    the 'interpolation' kwarg. Midpoint deltas may also be specified for finer interpolation by
    providing the kwargs 'RY' (red-yellow), 'YG' (yellow-green), 'GC' (green-cyan), 'CB' (cyan-blue),
    'BM' (blue-magenta) and 'MR' (magenta-red).
    Contrary to the rotation of HSV or HSL images, rotations in the CIELab and CIELuv color spaces
    preserve the lightness by design. They may, however, result in out-of-range RGB pixels (as not
    all points of of these color spaces correspond to physical RGB colors).

    Note:
      h*(red) ~ 0, h*(yellow) ~ 1/6, h*(green) ~ 1/3, h*(cyan) ~ 1/2, h*(blue) ~ 2/3, and h*(magenta) ~ 5/6.
      A uniform rotation D = 1/6 converts red → yellow, yellow → green, green → cyan, cyan → blue,
      blue → magenta, and magenta → red.
      A uniform rotation D = -1/6 converts red → magenta, yellow → red, green → yellow, cyan → green,
      blue → cyan, and magenta → blue.

    See also:
      rotate_HSX_hue

    Args:
      D (float, optional): The delta for all hues (default 0).
      R (float, optional): The red delta (default D).
      Y (float, optional): The yellow delta (default D).
      G (float, optional): The green delta (default D).
      C (float, optional): The cyan delta (default D).
      B (float, optional): The blue delta (default D).
      M (float, optional): The magenta delta (default D).
      colorspace (str, optional): The color space for rotation ["CIELab" (default) or "CIELuv"].
      interpolation (str, optional): The interpolation method for delta(hue angle):

        - "nearest": Nearest neighbor interpolation.
        - "linear": Linear spline interpolation.
        - "cubic": Cubic spline interpolation.
        - "akima": Akima spline interpolation (default).

      trans (bool, optional): If True (default), embed the transormation in the output image
        as output.trans (see Image.apply_channels).

    Returns:
      Image: The processed image.
    """
    hgrid, dgrid, kwargs = parse_hue_kwargs(D, kwargs)
    if kwargs: raise ValueError(f"Error, unknown keyword argument(s): {', '.join(kwargs.keys())}.")
    if colorspace in ["CIELab", "CIELuv"]:
      CIE = self.convert(colorspace = colorspace, colormodel = "Lch", copy = True)
    else:
      raise ValueError("Error, colorspace must be 'CIELab' or 'CIELuv'.")
    hue = CIE.image[2]
    delta = interpolate_hue(hue, hgrid, dgrid, interpolation)
    CIE.image[2] = (hue+delta)%1.
    output = CIE.convert(colorspace = self.colorspace, colormodel = self.colormodel, copy = False)
    if trans:
      t = helpers.Container()
      t.type = "hue"
      t.input = self
      t.channels = "h*" if self.colorspace in ["CIELab", "CIELuv"] else ""
      t.xm = np.linspace(0., 1., 7)
      t.ym = interpolate_hue(t.xm, hgrid, dgrid, interpolation)
      t.cm = ["red", "yellow", "green", "cyan", "blue", "magenta", "red"]
      t.x = np.linspace(0., 1., params.ntranslo)
      t.y = interpolate_hue(t.x, hgrid, dgrid, interpolation)
      t.ylabel = "\u0394"
      output.trans = t
    return output

  ##########################
  # Color noise reduction. #
  ##########################

  def SCNR(self, hue = "green", protection = "avgneutral", amount = 1., colorspace = None, lightness = True):
    """Subtractive Chromatic Noise Reduction of a given hue of a RGB image.

    The input hue is reduced according to the 'protection' kwarg. For the green hue for example,

      - G ← min(G, C) with C = (R+B)/2 for average neutral protection (protection = "avgneutral").
      - G ← min(G, C) with C = max(R, B) for maximum neutral protection (protection = "maxneutral").
      - G ← G*[(1-A)+C*A] with C = (R+B)/2 for additive mask protection (protection = "addmask").
      - G ← G*[(1-A)+C*A] with C = max(R, B) for maximum mask protection (protection = "maxmask").

    The parameter A in [0, 1] controls the strength of the mask protection.

    Args:
      hue (str, optional): The hue to be reduced ["red" alias "R", "yellow" alias "Y", "green"
        alias "G" (default), "cyan" alias "C", "blue" alias "B", or "magenta" alias "M"].
      protection (str, optional): The protection mode ["avgneutral" (default), "maxneutral",
        "addmask" or "maxmask"].
      amount (float, optional): The parameter A for mask protection (protection = "addmask"
        or "maxmask", default 1).
      colorspace (str, optional): The color space for SCNR ["lRGB", "sRGB", or None (default)
        to use the color space of the image].
      lightness (bool, optional): If True (default), preserve the lightness L* of the original image.
        Note that this may result in some out-of-range pixels.

    Returns:
      Image: The processed image.
    """
    self.check_color_space("lRGB", "sRGB")
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
    if colorspace is None:
      colorspace = self.colorspace
    elif colorspace not in ["lRGB", "sRGB"]:
      raise ValueError("Error, colorspace must be 'lRGB' or 'sRGB'.")
    converted = self.convert(colorspace = colorspace, colormodel = "RGB", copy = False)
    image = np.clip(converted.image, 0., 1.) # Clip before reducing color noise.
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
    output = converted.newImage(image)
    if lightness: output.set_channel("L*sh", self.lightness(), inplace = True)
    return output.convert(colorspace = self.colorspace, copy = False)
