# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01
# DOC+MCI.

"""Color spaces and models management."""

import numpy as np
import skimage.color as skcolor

from . import params

#############################
# sRGB <-> lRGB conversion. #
#############################

def sRGB_to_lRGB(image):
  """Convert the input sRGB image into a linear RGB image.

  Args:
    image (np.array): The input sRGB image.
    
  Returns:
    np.array: The converted lRGB image.  
  """
  srgb = np.clip(image, 0., None)
  return np.where(srgb > .04045, ((srgb+0.055)/1.055)**2.4, srgb/12.92)

def lRGB_to_sRGB(image):
  """Convert the input linear RGB image into a sRGB image.

  Args:
    image (np.array): The input lRGB image.
    
  Returns:
    np.array: The converted sRGB image.  
  """
  lrgb = np.clip(image, 0., None)
  return np.where(lrgb > .0031308, 1.055*lrgb**(1./2.4)-0.055, 12.92*lrgb)

###########################
# RGB <-> HSV conversion. #
###########################

def RGB_to_HSV(image):
  """Convert the input RGB image into a HSV image.
  
  Args:
    image (np.array): The input RGB image.
    
  Returns:
    np.array: The converted HSV image.  
  """
  return skcolor.rgb2hsv(image, channel_axis = 0)

def HSV_to_RGB(image):
  """Convert the input HSV image into a RGB image.
    
  Args:
    image (np.array): The input HSV image.
    
  Returns:
    np.array: The converted RGB image.  
  """  
  return skcolor.hsv2rgb(image, channel_axis = 0)

def value(image):
  """Return the HSV value V = max(RGB) of the input RGB image.
  
  Note: Compatible with single channel grayscale images.
  
  Args:
    image (np.array): The input RGB image.
    
  Returns:
    np.array: The HSV value V.  
  """
  return image.max(axis = 0)

def saturation(image):
  """Return the HSV saturation S = 1-min(RGB)/max(RGB) of the input RGB image.
  
  Note: Compatible with single channel grayscale images.
  
  Args:
    image (np.array): The input RGB image.
    
  Returns:
    np.array: The HSV saturation S.  
  """
  return 1.-image.min(axis = 0)/image.max(axis = 0, initial = params.IMGTOL) # Safe evaluation.

#########
# Luma. #
#########

def get_RGB_luma():
  """Return the RGB weights of the luma.
    
  Returns:
    tuple: The (red, blue, green) weights of the luma.
    
  Also see:
    luma,
    set_RGB_luma
  """   
  return params.rgbluma

def set_RGB_luma(rgb):
  """Set the RGB weights of the luma.
  
  Args:
    rgb: The RGB weights of the luma as:
      - a tuple, list or array of the (red, green, blue) weights. They will be normalized so that their sum is 1.
      - the string "uniform": the RGB weights are set to (1/3, 1/3, 1/3).
      - the string "human": the RGB weights are set to (.299, .587, .114). The luma is then the luminance for lRGB images,
        and an approximate substitute for the lightness for sRGB images.
      
  Also see:
    luma,
    get_RGB_luma      
  """
  if isinstance(rgb, str):
    if rgb == "uniform":
      set_RGB_luma((1./3., 1./3., 1./3.))
    elif rgb == "human":
      set_RGB_luma((.299, .587, .114))
    else:
      raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
  else:
    w = np.array(rgb, dtype = params.IMGTYPE)
    if w.shape != (3,): raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
    if any(w < 0.): raise ValueError("Error, the input rgb weights must be >= 0.")
    s = np.sum(w)
    if s == 0.: raise ValueError("Error, the sum of the input rgb weights must be > 0.")
    params.rgbluma = w/s
    print(f"Luma = {params.rgbluma[0]:.3f}R+{params.rgbluma[1]:.3f}G+{params.rgbluma[2]:.3f}B.")

def luma(image):
  """Return the luma L of the input RGB image.
  
  The luma L is the average of the RGB components weighted by rgbluma = get_RGB_luma():
    L = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2].
       
  Note: Compatible with single channel grayscale images.

  Args:
    image (np.array): The input RGB image.
    
  Returns:
    np.array: The luma L. 
  """
  return params.rgbluma[0]*image[0]+params.rgbluma[1]*image[1]+params.rgbluma[2]*image[2] if image.shape[0] > 1 else image[0]

############################
# Luminance and lightness. #
############################

def lRGB_luminance(image):
  """Return the luminance Y of the input linear RGB image.
  
  The luminance Y of a lRGB image is defined as:
    Y = 0.2126*R+0.7152*G+0.0722*B
  It is equivalently the luma of the lRGB image for RGB weights (0.2126, 0.7152, 0.0722),
  and is the basic ingredient of the perceptual lightness L*.
  
  Note: Compatible with single channel grayscale images.
  
  Args:
    image (np.array): The input lRGB image.
    
  Returns:
    np.array: The luminance Y.   
    
  See also:
    luma, 
    lRGB_lightness
  """
  return .2126*image[0]+.7152*image[1]+.0722*image[2] if image.shape[0] > 1 else image[0]

def lRGB_lightness(image):
  """Return the CIE lightness L* of the input linear RGB image.
  
  The CIE lightness L* is defined from the lRGB luminance Y as:
    L* = 116*Y**(1/3)-16 if Y > 0.008856 and L* = 903.3*Y if Y < 0.008856.
  It is a measure of the perceptual lightness of the image.
  
  Warning: L* is defined within [0, 100] instead of [0, 1].
  
  Note: Compatible with single channel grayscale images.
  
  Args:
    image (np.array): The input lRGB image.
    
  Returns:
    np.array: The CIE lightness L*.    
  """
  Y = lRGB_luminance(image)
  return np.where(Y > .008856, 116.*Y**(1./3.)-16., 903.3*Y)

def sRGB_luminance(image):
  """Return the luminance Y of the input sRGB image.
  
  The image is converted to the lRGB color space to compute the luminance Y.
  
  Note: Although they have the same functional forms, the luma and luminance are 
  different concepts for sRGB images: the luma is computed in the sRGB color space
  as a *substitute* for the perceptual lightness, whereas the luminance is
  computed after conversion in the lRGB color space and is the basic ingredient of 
  the *genuine* perceptual lightness (see lRGB_lightness).

  Note: Compatible with single channel grayscale images.

  Args:
    image (np.array): The input sRGB image.
    
  Returns:
    np.array: The luminance Y.    
    
  See also:
    luma, 
    sRGB_lightness    
  """  
  return lRGB_luminance(sRGB_to_lRGB(image))

def sRGB_lightness(image):
  """Return the CIE lightness L* of the input sRGB image.
  
  The image is converted to the lRGB color space to compute the CIE lightness L*.
  L* is a measure of the perceptual lightness of the image.
  
  Warning: L* is defined within [0, 100] instead of [0, 1].

  Note: Compatible with single channel grayscale images.
  
  Args:
    image (np.array): The input sRGB image.
    
  Returns:
    np.array: The CIE lightness L*.    
  """
  return lRGB_lightness(sRGB_to_lRGB(image))

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
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

  def check_color_spaces(self, *colorspaces):
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
      return self.newImage_like(self, sRGB_to_lRGB(self), colorspace = "lRGB")
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
      return self.newImage_like(self, lRGB_to_sRGB(self), colorspace = "sRGB")
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
      return self.newImage_like(self, HSV_to_RGB(self), colormodel = "RGB")
    elif self.colormodel == "gray":
      return self.newImage_like(self, np.repeat(self[0, :, :], 3, axis = 0), colormodel = "RGB")      
    else:
      self.color_model_error()

  def HSV(self):
    """Convert the image to the HSV color model.
    
    Note: The conversion from a gray scale to a HSV image is ill-defined (no hue).
    
    Returns:
      Image: The converted HSV image (a copy of the original image if already HSV).     
    """
    if self.colormodel == "RGB":
      return self.newImage_like(self, RGB_to_HSV(self), colormodel = "HSV")
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
      np.array: The HSV value V.
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return value(self)
    elif self.colormodel == "HSV":
      return self[2]
    else:
      self.color_model_error()

  def saturation(self):
    """Return the HSV saturation S = 1-min(RGB)/max(RGB) of the image.
    
    Returns:
      np.array: The HSV saturation S.    
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return saturation(self)
    elif self.colormodel == "HSV":
      return self[1]  
    else:
      self.color_model_error()

  def luma(self):
    """Return the luma L of the image.

    The luma L is the average of the RGB components weighted by rgbluma = get_RGB_luma():
      L = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2].
         
    Warning: The luma is available only for RGB and grayscale images.
    
    Returns:
      np.array: The luma L.     
    """
    if self.colormodel == "RGB" or self.colormodel == "gray":
      return luma(self)
    else:
      self.color_model_error()      

  def luminance(self):
    """Return the luminance Y of the image.
    
    Warning: The luminance is available only for RGB and grayscale images.
    
    Returns:
      np.array: The luminance Y.     
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return lRGB_luminance(self)
    elif self.colorspace == "sRGB":
      return sRGB_luminance(self)
    else:
      self.color_space_error()

  def lightness(self):
    """Return the CIE lightness L* of the image.
    
    Warning: The lightness is available only for RGB and grayscale images.
    
    Returns:
      np.array: The lightness L*.    
    """
    self.check_color_model("RGB", "gray")
    if self.colorspace == "lRGB":
      return lRGB_lightness(self)
    elif self.colorspace == "sRGB":
      return sRGB_lightness(self)
    else:
      raise self.color_space_error()

  #################################
  # Channel-selective operations. #
  #################################

  def apply_channels(self, f, channels, multi = True):
    """Apply the operation f(channel) to selected channels of the image.
    
    Note: When applying an operation to the luma, the RGB components of the image are rescaled 
    by the ratio f(luma)/luma. This preserves the hue, but may bring some RGB components 
    out-of-range even though f(luma) is within [0, 1]. These out-of-range components can be  
    regularized with two highlight protection methods:
      - "saturation": The out-of-range pixels are desaturated at constant luma (namely, the
        out-of-range components are decreased while the in-range components are increased so 
        that the luma is conserved). This tends to whiten the out-of-range pixels. 
        f(luma) must be within [0, 1] to make use of this highlight protection method.
      - "mixing": The out-of-range pixels are blended with f(RGB) (the operation applied to the
        RGB channels). This usually tends to whiten the out-of-range pixels too. 
        f(RGB) must be within [0, 1] to make use of this highlight protection method.
    Alternatively, applying the operation to the HSV value V also preserves the hue and can not
    produce out-of-range pixels if f([0, 1]) is within [0, 1]. However, this may strongly affect
    the balance of the image, the HSV value being a poor approximation to the perceptual lightness.
    
    Args:
      f: The function f(np.array) -> np.array applied to the selected channels.
      channels (str): The selected channels:
        - An empty string: Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
            Apply the operation to the first/second/third channel (RGB, HSV and grayscale images).         
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation.
               (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lm": Apply the operation to the luma, with highlights protection by mixing.
               (after the operation, the out-of-range pixels are mixed with f(RGB)).
      multi (bool, optional): if True (default), the operation can be applied to the whole image at once;
                              if False, the operation must be applied one channel at a time.
                                 
    Returns:
      Image: The processed image.
    """ 
    is_RGB  = (self.colormodel == "RGB")
    is_HSV  = (self.colormodel == "HSV")
    is_gray = (self.colormodel == "gray")    
    if channels == "V":
      if is_RGB or is_gray:
        value = self.value()
        return self.scale_pixels(value, f(value))
      elif is_HSV:
        hsv = self.copy()
        hsv[2] = f(self[2])
        return hsv
      else:
        self.color_model_error()
    elif channels == "S":
      if is_RGB:
        hsv = self.HSV()
        hsv[1] = f(hsv[1])
        return hsv.RGB()
      elif is_HSV:
        hsv = self.copy()
        hsv[1] = f(self[1])
        return hsv
      else:
        self.color_model_error()
    elif channels == "L" or channels == "Ls" or channels == "Lm":
      luma = self.luma()
      output = self.scale_pixels(luma, f(luma))
      if channels == "Ls":
        return output.protect_highlights_sat()
      elif channels == "Lm":
        raise RuntimeError("Not yet implemented !..")
      else:
        return output        
    else:
      selected = [False, False, False]
      for c in channels:
        ok = True
        if c == "1":
          ic = 0
        elif c == "2":
          ic = 1
        elif c == "3":
          ic = 2
        elif c == "R":
          ic = 0
          ok = ok and (is_RGB or is_gray)
        elif c == "G":
          ic = 1
          ok = ok and is_RGB
        elif c == "B":
          ic = 2
          ok = ok and is_RGB
        else:
          ok = False
        if not ok:
          raise ValueError(f"Error, unknown or incompatible channel '{c}'.")
        if selected[ic]:
          print(f"Warning, channel '{c}' selected twice or more...")
        selected[ic] = True
      if not any(selected): 
        selected = [True, True, True]
      if all(selected) and multi:
        return self.newImage_like(self, f(self))
      else:
        output = self.empty()
        for ic in range(self.get_nc()):
          if selected[ic]:
            output[ic] = f(self[ic])
          else:
            output[ic] =   self[ic]
        return output

  def clip_channels(self, channels):
    """Clip selected channels of the image in the [0, 1] range.
    
    Args:
      channels (str): The selected channels:    
        - An empty string: Apply the operation to all channels (RGB, HSV and grayscale images).
        - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images):
            Apply the operation to the first/second/third channel (RGB, HSV and grayscale images).         
        - "V": Apply the operation to the HSV value (RGB, HSV and and grayscale images).
        - "S": Apply the operation to the HSV saturation (RGB and HSV images).
        - "L": Apply the operation to the luma (RGB and grayscale images).
        - "Ls": Apply the operation to the luma, with highlights protection by desaturation.
               (after the operation, the out-of-range pixels are desaturated at constant luma).
        - "Lm": Apply the operation to the luma, with highlights protection by mixing.
               (after the operation, the out-of-range pixels are mixed with f(RGB)).    
    
    Returns:
      Image: The clipped image.
    """
    return self.apply_channels(lambda channel: np.clip(channel, 0., 1.), channels)

  def protect_highlights_sat(self):
    """Normalize out-of-range pixels with HSV value > 1 by adjusting the saturation at constant luma.
    
    This method aims at protecting the highlights from overflowing when stretching the luma.
    
    Warning: The luma must be <= 1 even though some pixels have HSV value > 1.

    Returns:
      Image: The processed image.       
    """
    self.check_color_model("RGB")
    luma = self.luma() # Original luma.
    newimage = self.copy()
    if np.any(luma > 1.+params.IMGTOL/2):
      print("Warning, can not protect highlights if the luma itself is out-of-range. Returning original image...")
      return newimage
    newimage /= np.maximum(self.max(axis = 0), 1.) # Rescale maximum HSV value to 1.
    newluma = newimage.luma() # Updated luma.
    # Scale the saturation.
    # Note: The following implementation is failsafe when newluma -> 1 (in which case luma is also 1 in principle),
    # at the cost of a small error.
    fs = ((1.-luma)+params.IMGTOL)/((1.-newluma)+params.IMGTOL)
    output = 1.-(1.-newimage)*fs
    diffluma = luma-output.luma()
    print(f"Maximum luma difference = {abs(diffluma).max()}.")  
    return output
