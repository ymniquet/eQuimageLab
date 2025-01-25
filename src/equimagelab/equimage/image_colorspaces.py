# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Color spaces and models management."""

import numpy as np
import skimage.color as skcolor

from . import params
from . import helpers

#############################
# sRGB <-> lRGB conversion. #
#############################

def sRGB_to_lRGB(image):
  """Convert the input sRGB image into a linear RGB image.

  See also:
    The reciprocal lRGB_to_sRGB function.

  Args:
    image (numpy.ndarray): The input sRGB image.

  Returns:
    numpy.ndarray: The converted lRGB image.
  """
  output = .07739943839475116*image
  mask = (image > .04045)
  output[mask] = ((image[mask]+.055)/1.055)**2.4
  return output

def lRGB_to_sRGB(image):
  """Convert the input linear RGB image into a sRGB image.

  See also:
    The reciprocal sRGB_to_lRGB function.

  Args:
    image (numpy.ndarray): The input lRGB image.

  Returns:
    numpy.ndarray: The converted sRGB image.
  """
  output = 12.919990386749564*image
  mask = (image > .0031308072830676845)
  output[mask] = 1.055*image[mask]**(1./2.4)-.055
  return output

###########################
# RGB <-> HSV conversion. #
###########################

def RGB_to_HSV(image):
  """Convert the input RGB image into a HSV image.

  See also:
    The reciprocal HSV_to_RGB function.

  Args:
    image (numpy.ndarray): The input RGB image.

  Returns:
    numpy.ndarray: The converted HSV image.
  """
  return skcolor.rgb2hsv(image, channel_axis = 0)

def HSV_to_RGB(image):
  """Convert the input HSV image into a RGB image.

  See also:
    The reciprocal RGB_to_HSV function.

  Args:
    image (numpy.ndarray): The input HSV image.

  Returns:
    numpy.ndarray: The converted RGB image.
  """
  return skcolor.hsv2rgb(image, channel_axis = 0)

def value(image):
  """Return the HSV value V = max(RGB) of the input RGB image.

  Note: Compatible with single channel grayscale images.

  Args:
    image (numpy.ndarray): The input RGB image.

  Returns:
    numpy.ndarray: The HSV value V.
  """
  return image.max(axis = 0)

def saturation(image):
  """Return the HSV saturation S = 1-min(RGB)/max(RGB) of the input RGB image.

  Note: Compatible with single channel grayscale images.

  Args:
    image (numpy.ndarray): The input RGB image.

  Returns:
    numpy.ndarray: The HSV saturation S.
  """
  return 1.-image.min(axis = 0)/image.max(axis = 0, initial = helpers.fpaccuracy(image.dtype)) # Safe evaluation.

#########
# Luma. #
#########

def luma(image):
  """Return the luma L of the input RGB image.

  The luma L is the average of the RGB components weighted by rgbluma = get_RGB_luma():

    L = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2].

  Note: Compatible with single channel grayscale images.

  Args:
    image (numpy.ndarray): The input RGB image.

  Returns:
    numpy.ndarray: The luma L.
  """
  rgbluma = params.get_RGB_luma()
  return rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2] if image.shape[0] > 1 else image[0]

############################
# Luminance and lightness. #
############################

def luminance_to_lightness(Y):
  """Compute the CIE lightness L* from the lRGB luminance Y.

  The CIE lightness L* is defined from the lRGB luminance Y as:

    L* = 116*Y**(1/3)-16 if Y > 0.008856 and L* = 903.3*Y if Y < 0.008856.

  Note that L* is conventionally defined within [0, 100]. However, this
  function returns the scaled lightness L*/100 within [0, 1].

  See also:
    The reciprocal lightness_to_luminance function.

  Args:
    Y (numpy.ndarray): The luminance Y.

  Returns:
    numpy.ndarray: The CIE lightness L*/100.
  """
  Lstar = 9.032962955130763*Y
  mask = (Y > .008856)
  Lstar[mask] = 1.16*Y[mask]**(1./3.)-.16
  return Lstar

def lightness_to_luminance(Lstar):
  """Compute the lRGB luminance Y from the CIE lightness L*.

  See also:
    The reciprocal luminance_to_lightness function.

  Args:
    Lstar (numpy.ndarray): The CIE lightness L*/100.

  Returns:
    numpy.ndarray: The luminance Y.
  """
  Y = 0.11070564608393481*Lstar
  mask = (Lstar > .07999591993063804)
  Y[mask] = ((Lstar[mask]+.16)/1.16)**3

def lRGB_luminance(image):
  """Return the luminance Y of the input linear RGB image.

  The luminance Y of a lRGB image is defined as:

    Y = 0.2126*R+0.7152*G+0.0722*B

  It is equivalently the luma of the lRGB image for RGB weights (0.2126, 0.7152, 0.0722),
  and is the basic ingredient of the perceptual lightness L*.

  Note: Compatible with single channel grayscale images.

  See also:
    lRGB_lightness,
    sRGB_luminance,
    sRGB_lightness,
    luma

  Args:
    image (numpy.ndarray): The input lRGB image.

  Returns:
    numpy.ndarray: The luminance Y.
  """
  return .2126*image[0]+.7152*image[1]+.0722*image[2] if image.shape[0] > 1 else image[0]

def lRGB_lightness(image):
  """Return the CIE lightness L* of the input linear RGB image.

  The CIE lightness L* is defined from the lRGB luminance Y as:

    L* = 116*Y**(1/3)-16 if Y > 0.008856 and L* = 903.3*Y if Y < 0.008856.

  It is a measure of the perceptual lightness of the image. Note that L* is
  conventionally defined within [0, 100]. However, this function returns
  the scaled lightness L*/100 within [0, 1].

  Note: Compatible with single channel grayscale images.

  See also:
    lRGB_luminance,
    sRGB_luminance,
    sRGB_lightness,
    luma

  Args:
    image (numpy.ndarray): The input lRGB image.

  Returns:
    numpy.ndarray: The CIE lightness L*/100.
  """
  return luminance_to_lightness(lRGB_luminance(image))

def sRGB_luminance(image):
  """Return the luminance Y of the input sRGB image.

  The image is converted to the lRGB color space to compute the luminance Y.

  Note: Although they have the same functional forms, the luma and luminance are
  different concepts for sRGB images: the luma is computed in the sRGB color space
  as a *substitute* for the perceptual lightness, whereas the luminance is
  computed after conversion in the lRGB color space and is the basic ingredient of
  the *genuine* perceptual lightness (see lRGB_lightness).

  Note: Compatible with single channel grayscale images.

  See also:
    sRGB_lightness,
    lRGB_luminance,
    lRGB_lightness,
    luma

  Args:
    image (numpy.ndarray): The input sRGB image.

  Returns:
    numpy.ndarray: The luminance Y.
  """
  return lRGB_luminance(sRGB_to_lRGB(image))

def sRGB_lightness(image):
  """Return the CIE lightness L* of the input sRGB image.

  The image is converted to the lRGB color space to compute the CIE lightness L*.
  L* is a measure of the perceptual lightness of the image. Note that L* is
  conventionally defined within [0, 100]. However, this function returns
  the scaled lightness L*/100 within [0, 1].

  Note: Compatible with single channel grayscale images.

  See also:
    sRGB_luminance,
    lRGB_luminance,
    lRGB_lightness,
    luma

  Args:
    image (numpy.ndarray): The input sRGB image.

  Returns:
    numpy.ndarray: The CIE lightness L*/100.
  """
  return lRGB_lightness(sRGB_to_lRGB(image))

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  #####################################
  # Color space and model management. #
  #####################################

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

  def check_color_space(self, *colorspaces):
    """Raise a color space error if the color space of the image is not in the arguments.

    See also:
      color_space_error
    """
    if self.colorspace not in colorspaces: self.color_space_error()

  def check_color_model(self, *colormodels):
    """Raise a color model error if the color model of the image is not in the arguments.

    See also:
      color_model_error
    """
    if self.colormodel not in colormodels: self.color_model_error()

  def check_is_not_gray(self):
    """Raise a color model error if the image is a grayscale.

    See also:
      color_model_error
    """
    if self.colormodel == "gray": self.color_model_error()

  ############################
  # Color space conversions. #
  ############################

  def lRGB(self):
    """Convert the image to the linear RGB color space.

    Warning:
      The color model must be "RGB" or "gray".

    Returns:
      Image: The converted lRGB image (a copy of the original image if already lRGB).
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return self.copy()
    elif self.colorspace == "sRGB":
      return self.newImage(sRGB_to_lRGB(self.image), colorspace = "lRGB")
    else:
      self.color_space_error()

  def sRGB(self):
    """Convert the image to the sRGB color space.

    Warning:
      The color model must be "RGB" or "gray".

    Returns:
      Image: The converted sRGB image (a copy of the original image if already sRGB).
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return self.newImage(lRGB_to_sRGB(self.image), colorspace = "sRGB")
    elif self.colorspace == "sRGB":
      return self.copy()
    else:
      self.color_space_error()

  ############################
  # Color model conversions. #
  ############################

  def RGB(self):
    """Convert the image to the RGB color model.

    Returns:
      Image: The converted RGB image (a copy of the original image if already RGB).
    """
    if self.colormodel == "RGB":
      return self.copy()
    elif self.colormodel == "HSV":
      return self.newImage(HSV_to_RGB(self.image), colormodel = "RGB")
    elif self.colormodel == "gray":
      return self.newImage(np.repeat(self.image[0, :, :], 3, axis = 0), colormodel = "RGB")
    else:
      self.color_model_error()

  def HSV(self):
    """Convert the image to the HSV color model.

    Warning:
      The conversion from a gray scale to a HSV image is ill-defined (no hue).

    Returns:
      Image: The converted HSV image (a copy of the original image if already HSV).
    """
    if self.colormodel == "RGB":
      return self.newImage(RGB_to_HSV(self.image), colormodel = "HSV")
    elif self.colormodel == "HSV":
      return self.copy()
    else:
      self.color_model_error()

  #######################
  # Composite channels. #
  #######################

  def value(self):
    """Return the HSV value V = max(RGB) of the image.

    Returns:
      numpy.ndarray: The HSV value V.
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return value(self.image)
    elif self.colormodel == "HSV":
      return self.image[2]
    else:
      self.color_model_error()

  def saturation(self):
    """Return the HSV saturation S = 1-min(RGB)/max(RGB) of the image.

    Returns:
      numpy.ndarray: The HSV saturation S.
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return saturation(self.image)
    elif self.colormodel == "HSV":
      return self.image[1]
    else:
      self.color_model_error()

  def luma(self):
    """Return the luma L of the image.

    The luma L is the average of the RGB components weighted by rgbluma = get_RGB_luma():

      L = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2].

    Warning:
      The luma is available only for RGB and grayscale images.

    Returns:
      numpy.ndarray: The luma L.
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return luma(self.image)
    else:
      self.color_model_error()

  def luminance(self):
    """Return the luminance Y of the image.

    Warning:
      The luminance is available only for RGB and grayscale images.

    Returns:
      numpy.ndarray: The luminance Y.
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return lRGB_luminance(self.image)
    elif self.colorspace == "sRGB":
      return sRGB_luminance(self.image)
    else:
      self.color_space_error()

  def lightness(self):
    """Return the CIE lightness L* of the image.

    L* is a measure of the perceptual lightness of the image. Note that L*
    is conventionally defined within [0, 100]. However, this method returns
    the scaled lightness L*/100 within [0, 1].

    Warning:
      The lightness is available only for RGB and grayscale images.

    Returns:
      numpy.ndarray: The CIE lightness L*/100.
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return lRGB_lightness(self.image)
    elif self.colorspace == "sRGB":
      return sRGB_lightness(self.image)
    else:
      raise self.color_space_error()

  #################################
  # Channel-selective operations. #
  #################################

  def get_channel(self, channel):
    """Return the selected channel of the image.

    Args:
      channel (str): The selected channel:

        - "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
          The first/second/third channel (RGB, HSV and grayscale images).
        - "V": The HSV value (RGB, HSV and and grayscale images).
        - "S": The HSV saturation (RGB and HSV images).
        - "L": The luma (RGB and grayscale images).
        - "L*": The CIE lightness L* (RGB and grayscale images).

    Returns:
      numpy.ndarray: The selected channel.
    """
    channel = channel.strip()
    if channel.isdigit():
      ic = int(channel)-1
      if ic < 0 or ic >= self.get_nc(): raise ValueError(f"Error, invalid channel '{channel}'.")
      return self.image[ic]
    elif channel in ["R", "G", "B"]:
      self.check_color_model("RGB")
      ic = "RGB".index(channel)
      return self.image[ic]
    elif channel == "V":
      return self.value()
    elif channel == "S":
      return self.saturation()
    elif channel == "L":
      return self.luma()
    elif channel == "L*":
      return self.lightness()
    else:
      raise ValueError(f"Error, unknown channel '{channel}'.")

  def set_channel(self, channel, data, inplace = False):
    """Update the selected channel of the image.

    Args:
      channel (str): The updated channel:

        - "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
          Update the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Update the HSV value (RGB, HSV and and grayscale images).
        - "S": Update the HSV saturation (RGB and HSV images).
        - "L": Update the luma (RGB and grayscale images).
        - "L*": Update the lightness L* in the CIE L*a*b* color space (RGB and grayscale images).

      data (numpy.ndarray): The updated channel data, as a 2D array with the same width and height as the image.
      inplace (bool, optional): If True, update the image "in place"; if False (default), return a new image.

    Returns:
      Image: The updated image.
    """
    dtype = self.dtype if inplace else params.imagetype
    data = np.asarray(data, dtype)
    width, height = self.get_size()
    if data.shape != (height, width):
      raise ValueError("Error, the channel data must be a 2D array with the same with and height as the image.")
    is_RGB  = (self.colormodel == "RGB")
    is_HSV  = (self.colormodel == "HSV")
    is_gray = (self.colormodel == "gray")
    output = self if inplace else self.copy()
    channel = channel.strip()
    if channel.isdigit():
      ic = int(channel)-1
      if ic < 0 or ic >= self.get_nc(): raise ValueError(f"Error, invalid channel '{channel}'.")
      output.image[ic] = data
      return output
    elif channel in ["R", "G", "B"]:
      if not is_RGB: self.color_model_error()
      ic = "RGB".index(channel)
      output.image[ic] = data
      return output
    elif channel == "V":
      if is_gray:
        output.image[0] = data
      elif is_RGB:
        output.image[:] = helpers.scale_pixels(output.image, output.value(), data)
      elif is_HSV:
        output.image[2] = data
      else:
        self.color_model_error()
      return output
    elif channel == "S":
      if is_RGB:
        hsv = RGB_to_HSV(output.image)
        hsv[1] = data
        output.image[:] = HSV_to_RGB(hsv)
      elif is_HSV:
        output.image[1] = data
      else:
        self.color_model_error()
      return output
    elif channel == "L":
      if is_gray:
        output.image[0] = data
      elif is_RGB:
        output.image[:] = helpers.scale_pixels(output.image, output.luma(), data)
      else:
        self.color_model_error()
      return output
    elif channel == "L*":
      if is_gray:
        luminance = lightness_to_luminance(data)
        if output.colorspace == "lRGB":
          output.image[0] = luminance
        elif self.colorspace == "sRGB":
          output.image[0] = lRGB_to_sRGB(luminance)
        else:
          self.color_space_error()
      elif is_RGB:
        if self.colorspace == "lRGB": # Convert to L*a*b* color space.
          lRGB = output.image
        elif self.colorspace == "sRGB":
          lRGB = sRGB_to_lRGB(output.image)
        else:
          self.color_space_error()
        xyz = np.tensordot(np.asarray(params.RGB2XYZ, dtype = dtype), lRGB, axes = 1)
        lab = skcolor.xyz2lab(xyz, channel_axis = 0)
        lab[0] = 100.*data
        xyz = skcolor.lab2xyz(lab, channel_axis = 0) # Convert from L*a*b* color space.
        lRGB = np.tensordot(np.asarray(params.XYZ2RGB, dtype = dtype), xyz, axes = 1)
        if self.colorspace == "lRGB":
          output.image[:] = lRGB
        elif self.colorspace == "sRGB":
          output.image[:] = lRGB_to_sRGB(lRGB)
      else:
        self.color_model_error()
      return output
    else:
      raise ValueError(f"Error, unknown channel '{channel}'.")

  def apply_channels(self, f, channels, multi = True, trans = False):
    """Apply the operation f(channel) to selected channels of the image.

    Note: When applying an operation to the luma, the RGB components of the image are rescaled
    by the ratio f(luma)/luma. This preserves the hue, but may bring some RGB components
    out-of-range even though f(luma) is within [0, 1]. These out-of-range components can be
    regularized with two highlights protection methods:

      - "saturation": The out-of-range pixels are desaturated at constant luma (namely, the
        out-of-range components are decreased while the in-range components are increased so
        that the luma is conserved). This tends to whiten the out-of-range pixels.
        f(luma) must be within [0, 1] to make use of this highlights protection method.
      - "mixing": The out-of-range pixels are blended with f(RGB) (the operation applied to the
        RGB channels). This usually tends to whiten the out-of-range pixels too.
        f(RGB) must be within [0, 1] to make use of this highlights protection method.

    Alternatively, applying the operation to the HSV value V also preserves the hue and can not
    produce out-of-range pixels if f([0, 1]) is within [0, 1]. However, this may strongly affect
    the balance of the image, the HSV value being a very poor approximation to the perceptual
    lightness.

    Args:
      f (function): The function f(numpy.ndarray) → numpy.ndarray applied to the selected channels.
      channels (str): The selected channels:

        - An empty string: Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
          Apply the operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, and protect highlights by desaturation.
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, and protect highlights by blending.
          (after the operation, the out-of-range pixels are blended with f(RGB)).
        - "Ln": Apply the operation to the luma, and protect highlights by normalization.
          (after the operation, the image is normalized so that all pixels fall in the [0, 1] range).
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space
          (RGB and grayscale images).

      multi (bool, optional): if True (default), the operation can be applied to the whole image at once;
        if False, the operation must be applied one channel at a time.
      trans (bool, optional): If True (default False), embeds the transformation y = f(x in [0, 1]) in the
        output image as output.trans, where:

        - output.trans.type = "hist".
        - output.trans.input is a reference to the input image (self).
        - output.trans.channels are the channels selected for the transformation.
        - output.trans.x is a mesh of the [0, 1] interval.
        - output.trans.y = f(output.trans.x)
        - output.trans.ylabel is a label for output.trans.y.
        - output.trans.xticks is a list of remarkable x values for this transformation (if any).

        trans shall be set True only for *local* histogram transformations f.

    Returns:
      Image: The processed image.
    """

    def transformation(f, x, channels):
      """Return the transformation container."""
      xmin = min(0., x.min())
      xmax = max(1., x.max())
      t = helpers.Container()
      t.type = "hist"
      t.input = self
      t.channels = channels
      t.x = np.linspace(xmin, xmax, max(int(round(params.ntranshi*(xmax-xmin))), 2*params.ntranshi))
      t.y = f(t.x)
      t.ylabel = "f"
      return t

    nc = self.get_nc()
    is_RGB  = (self.colormodel == "RGB")
    is_HSV  = (self.colormodel == "HSV")
    is_gray = (self.colormodel == "gray")
    channels = channels.strip()
    if channels == "":
      if is_gray:
        channels = "L"
      elif is_RGB:
        channels = "RGB"
      else:
        for ic in range(nc): channels += str(ic+1)
    if channels == "V":
      if is_gray:
        output = self.newImage(f(self.image))
        if trans: output.trans = transformation(f, self.image, "V")
      elif is_RGB:
        value = self.value()
        output = self.scale_pixels(value, f(value))
        if trans: output.trans = transformation(f, value, "V")
      elif is_HSV:
        output = self.copy()
        output.image[2] = f(self.image[2])
        if trans: output.trans = transformation(f, self.image[2], "V")
      else:
        self.color_model_error()
      return output
    elif channels == "S":
      if is_RGB:
        hsv = self.HSV()
        if trans: t = transformation(f, hsv.image[1], "S")
        hsv.image[1] = f(hsv.image[1])
        output = hsv.RGB()
        if trans: output.trans = t
      elif is_HSV:
        output = self.copy()
        output.image[1] = f(self.image[1])
        if trans: output.trans = transformation(f, self.image[1], "S")
      else:
        self.color_model_error()
      return output
    elif channels in ["L", "Ls", "Lb", "Ln"]:
      if is_gray:
        output = self.newImage(f(self.image))
        if trans: output.trans = transformation(f, self.image, "L")
      elif is_RGB:
        luma = self.luma()
        output = self.scale_pixels(luma, f(luma))
        if trans: t = transformation(f, luma, "L")
        if channels == "Ls":
          output = output.protect_highlights_saturation()
        elif channels == "Lb":
          output = output.protect_highlights_blend(self.apply_channels(f, "RGB", multi))
        elif channels == "Ln":
          maximum = np.max(output.image)
          if maximum > 1:
            output.image /= maximum
            if trans: t.y /= maximum
        if trans: output.trans = t
      else:
        self.color_model_error()
      return output
    elif channels == "L*":
      if is_gray:
        lightness = self.lightness()
        flightness = f(lightness)
        fluminance = lightness_to_luminance(flightness)
        if self.colorspace == "lRGB":
          output = self.newImage(fluminance)
        elif self.colorspace == "sRGB":
          output = self.newImage(lRGB_to_sRGB(fluminance))
        else:
          self.color_space_error()
      elif is_RGB:
        if self.colorspace == "lRGB": # Convert to L*a*b* color space.
          lRGB = self.image
        elif self.colorspace == "sRGB":
          lRGB = sRGB_to_lRGB(self.image)
        else:
          self.color_space_error()
        xyz = np.tensordot(np.asarray(params.RGB2XYZ, dtype = params.imagetype), lRGB, axes = 1)
        lab = skcolor.xyz2lab(xyz, channel_axis = 0)
        lightness = lab[0]/100. # Apply transformation.
        lab[0] = 100.*f(lightness)
        xyz = skcolor.lab2xyz(lab, channel_axis = 0) # Convert from L*a*b* color space.
        lRGB = np.tensordot(np.asarray(params.XYZ2RGB, dtype = params.imagetype), xyz, axes = 1)
        if self.colorspace == "lRGB":
          output = self.newImage(lRGB)
        elif self.colorspace == "sRGB":
          output = self.newImage(lRGB_to_sRGB(lRGB))
      else:
        self.color_model_error()
      if trans: output.trans = transformation(f, lightness, "L*")
      return output
    else:
      selected = nc*[False]
      for c in channels:
        if c.isdigit():
          ic = int(c)-1
          if ic < 0 or ic >= nc: raise ValueError(f"Error, invalid channel '{c}'.")
        elif c in "RGB":
          if not is_RGB: self.color_model_error()
          ic = "RGB".index(c)
        elif c == " ": # Skip spaces.
          continue
        else:
          raise ValueError(f"Syntax errror in the channels string '{channels}'.")
        if selected[ic]: print(f"Warning, channel '{c}' selected twice or more...")
        selected[ic] = True
      if all(selected) and multi:
        output = self.newImage(f(self.image))
      else:
        output = self.newImage(np.empty_like(self.image))
        for ic in range(nc):
          if selected[ic]:
            output.image[ic] = f(self.image[ic])
          else:
            output.image[ic] =   self.image[ic]
      if trans: output.trans = transformation(f, self.image[selected], channels)
      return output

  def clip_channels(self, channels):
    """Clip selected channels of the image in the [0, 1] range.

    Args:
      channels (str): The selected channels:

        - An empty string (default): Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
          Apply the operation to the first/second/third channel (RGB, HSV and grayscale images).
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, and protect highlights by desaturation.
          (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lb": Apply the operation to the luma, and protect highlights by blending.
          (after the operation, the out-of-range pixels are blended with channels = "RGB").
        - "L*": Apply the operation to the lightness L* in the CIE L*a*b* color space.
          (RGB and grayscale images).

    Returns:
      Image: The clipped image.
    """
    return self.apply_channels(lambda channel: np.clip(channel, 0., 1.), channels)

  def protect_highlights_saturation(self):
    """Normalize out-of-range pixels with HSV value > 1 by adjusting the saturation at constant luma.

    The out-of-range RGB components of the pixels are decreased while the in-range RGB components are
    increased so that the luma is conserved. This desaturates (whitens) the pixels with out-of-range
    components.
    This aims at protecting the highlights from overflowing when stretching the luma or lightness.

    Warning:
      The luma must be <= 1 even though some pixels have HSV value > 1.

    Returns:
      Image: The processed image.
    """
    self.check_color_model("RGB")
    imgluma = self.luma() # Original luma.
    epsilon = helpers.fpaccuracy(imgluma.dtype)
    if np.any(imgluma > 1.+epsilon/2):
      print("Warning, can not protect highlights if the luma is out-of-range. Returning original image...")
      return self.copy()
    newimage = self.image/np.maximum(self.image.max(axis = 0), 1.) # Rescale maximum HSV value to 1.
    newluma = luma(newimage) # Updated luma.
    # Scale the saturation.
    # Note: The following implementation is failsafe when newluma -> 1 (in which case luma is also 1 in principle),
    # at the cost of a small error.
    fs = ((1.-imgluma)+epsilon)/((1.-newluma)+epsilon)
    output = 1.-(1.-newimage)*fs
    diffluma = imgluma-luma(output)
    print(f"Maximum luma difference = {abs(diffluma).max()}.")
    return self.newImage(output)

  def protect_highlights_blend(self, inrange):
    """Normalize out-of-range pixels with HSV value > 1 by blending with an "in-range" image with HSV values <= 1.

    Each pixel of the image with out-of-range RGB components is brought back in the [0, 1] range by blending with the
    corresponding pixel of the input "in-range" image.
    This aims at protecting the highlights from overflowing when stretching the luma.

    Args:
      inrange (Image): The "in-range" image to blend with. All pixels must have HSV values <= 1.

    Returns:
      Image: The processed image.
    """
    self.check_color_model("RGB") ; inrange.check_color_model("RGB")
    epsilon = helpers.fpaccuracy(self.dtype)
    if np.any(inrange.value() > 1.+epsilon/2.):
      print("Warning, can not protect highlights if the input inrange image is out-of-range. Returning original image...")
      return self.copy()
    mixing = np.where(self.image > 1.+epsilon, helpers.failsafe_divide(self.image-1., self.image-inrange.image), 0.)
    return self.blend(inrange, mixing.max(axis = 0))
