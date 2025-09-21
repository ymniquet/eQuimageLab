# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 2.1.0 / 2025.09.21
# Doc OK.

"""Utils for JupyterLab interface.

The following symbols are imported in the equimagelab namespace for convenience:
  "filter", "shadowed", "highlighted", "differences", "stretch".
"""

__all__ = ["filter", "shadowed", "highlighted", "differences", "stretch"]

import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

from equimage import Image
from equimage.stretchfunctions import shadow_stretch_function, midtone_stretch_function

from . import params

def get_image_size(image):
  """Return the width and height of the input image.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).

  Returns:
    int: The (width, height) of the image in pixels.
  """
  if issubclass(type(image), Image):
    return image.get_size()
  elif issubclass(type(image), np.ndarray):
    if image.ndim in [2, 3]: return image.shape[1], image.shape[0]
  raise ValueError(f"Error, {image} is not a valid image.")

def format_images(images, sampling = 1, copy = False):
  """Format images for plotly and Dash.

  Returns all images as numpy.ndarrays with shape (height, width, 3) (for color images),
  or (height, width) (for grayscale images).

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarray(s) with shape (height, width, 3)
      (for color images), (height, width, 1) or (height, width) (for grayscale images).
    sampling (int, optional): The downsampling rate (default 1; set to `jupyter.params.sampling`
      if negative). Only the pixels image[::sampling, ::sampling] of a given image are processed, to
      speed up operations.
    copy (bool, optional): If False (default), the formatted images are (when possible) views of
      the original images; If True, they are always copies.

  Returns:
    A single/tuple/list of numpy.ndarray(s) with shape (height/sampling, width/sampling, 3) (for
    color images) or (height/sampling, width/sampling) (for grayscale images).
  """

  def format_image(image, copy):
    """Format input image."""
    if issubclass(type(image), Image):
      image = image.get_image(channels = -1)
    elif issubclass(type(image), np.ndarray):
      valid = image.ndim in [2, 3]
      if image.ndim == 3: valid = image.shape[2] in [1, 3]
      valid = valid and image.dtype in [np.float32, np.float64]
      if not valid: raise ValueError(f"Error, image {image} is not a valid image.")
    if image.ndim == 3 and image.shape[2] == 1: image = np.squeeze(image, axis = 2)
    if sampling > 1: image = image[::sampling, ::sampling]
    image = np.asarray(image, dtype = np.float32) # Downgrade floats for display.
    return image.copy() if copy else image

  if sampling <= 0: sampling = params.sampling
  if not isinstance(images, (tuple, list)): return format_image(images, copy)
  return type(images)(format_image(image, copy) for image in images)

def format_images_as_b64strings(images, sampling = 1, compression = 4):
  """Format images for plotly and Dash as PNGs encoded in base64 strings.

  Returns all images as PNGs encoded in base64 strings.

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarray(s) with shape (height, width, 3)
      (for color images), (height, width, 1) or (height, width) (for grayscale images).
    sampling (int, optional): The downsampling rate (default 1; set to `jupyter.params.sampling`
      if negative). Only the pixels image[::sampling, ::sampling] of a given image are processed, to
      speed up operations.
    compression (int, optional): The PNG compression level (default 4).

  Returns:
    A single/tuple/list of PNG image(s) encoded in base64 string(s).
  """

  def format_image(image):
    """Format input image."""
    if issubclass(type(image), Image):
      image = image.get_image(channels = -1)
    elif issubclass(type(image), np.ndarray):
      valid = image.ndim in [2, 3]
      if image.ndim == 3: valid = image.shape[2] in [1, 3]
      valid = valid and image.dtype in [np.float32, np.float64]
      if not valid: raise ValueError(f"Error, image {image} is not a valid image.")
    if image.ndim == 3 and image.shape[2] == 1: image = np.squeeze(image, axis = 2)
    PILimg = PILImage.fromarray(np.clip(image[::sampling, ::sampling]*255, 0, 255).astype("uint8"))
    buffer = BytesIO()
    PILimg.save(buffer, format = "PNG", compress_level = compression)
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("utf-8")

  if sampling <= 0: sampling = params.sampling
  if not isinstance(images, (tuple, list)): return format_image(images)
  return type(images)(format_image(image) for image in images)

def filter(image, channels):
  """Filter the channels of a RGB image.

  Returns a copy of the image with the selected red/green/blue channels set to zero.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    channels (str): The *displayed* channels. A combination of the letters "R" (red), "G" (green),
      and "B" (blue).

  Returns:
    numpy.ndarray: A copy of the image as an array with shape (height, width, 3) and the non-
    selected channels set to zero.
  """
  selected = np.array([False, False, False])
  for c in channels:
    if c == "R":
      selected[0] = True
    elif c == "G":
      selected[1] = True
    elif c == "B":
      selected[2] = True
    else:
      raise ValueError(f"Error, unknown channel '{c}'.")
  image = format_images(image, copy = True)
  if image.ndim != 3: raise ValueError("Error, the input must be a RGB (not a grayscale) image.""")
  image[:, :, ~selected] = 0.
  return image

def shadowed(image, reference = None, dest = None):
  """Highlight black pixels of an image.

  Highlights black pixels on the input image with color `jupyter.params.shadowcolor`.
  If a reference image is provided, highlights pixels black on both input and reference images with
  color 0.5 * `jupyter.params.shadowcolor`.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    reference (optional): An Image object or numpy.ndarray with shape (height, width, 3)
      (for a color image), (height, width, 1) or (height, width) (for a grayscale image).
      Default is None.
    dest (optional): If None (default), the pixels are highlighted on a copy of the input image.
      Otherwise, the pixels are highlighted on a copy of dest, an Image object or numpy.ndarray
      with shape (height, width, 3) (for a color image), (height, width, 1) or (height, width)
      (for a grayscale image).

  Returns:
    numpy.ndarray: A copy of the image or of dest as an array with shape (height, width, 3) and the
    black pixels highlighted with color `jupyter.params.shadowcolor`.
  """
  image = format_images(image)
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  dest = image.copy() if dest is None else format_images(dest, copy = True)
  if dest.ndim == 2: dest = np.expand_dims(dest, axis = -1)
  if dest.shape[2] == 1: dest = np.repeat(dest, 3, axis = 2)
  imgmask = np.all(image < params.IMGTOL, axis = 2)
  dest[imgmask, :] = params.shadowcolor
  if reference is not None:
    reference = format_images(reference)
    if reference.shape[0:2] != image.shape[0:2]:
      print("Warning, image and reference have different sizes !")
      return dest
    if reference.ndim == 2: reference = np.expand_dims(reference, -1)
    refmask = np.all(reference < params.IMGTOL, axis = 2)
    dest[imgmask & refmask, :] = .5*params.shadowcolor
  return dest

def highlighted(image, reference = None, dest = None):
  """Highlight saturated pixels of an image.

  A pixel is saturated if at least one channel is >= 1.
  Shows pixels saturated on the input image with color `jupyter.params.highlightcolor`.
  If a reference image is provided, shows pixels saturated on both input and reference images with
  color 0.5 * `jupyter.params.highlightcolor`.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    reference (optional): An Image object or numpy.ndarray with shape (height, width, 3)
      (for a color image), (height, width, 1) or (height, width) (for a grayscale image).
      Default is None.
    dest (optional): If None (default), the pixels are highlighted on a copy of the input image.
      Otherwise, the pixels are highlighted on a copy of dest, an Image object or numpy.ndarray
      with shape (height, width, 3) (for a color image), (height, width, 1) or (height, width)
      (for a grayscale image).

  Returns:
    numpy.ndarray: A copy of the image or of dest as an array with shape (height, width, 3) and the
    saturated pixels highlighted with color `jupyter.params.highlightcolor`.
  """
  image = format_images(image)
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  dest = image.copy() if dest is None else format_images(dest, copy = True)
  if dest.ndim == 2: dest = np.expand_dims(dest, axis = -1)
  if dest.shape[2] == 1: dest = np.repeat(dest, 3, axis = 2)
  imgmask = np.any(image > 1., axis = 2)
  dest[imgmask, :] = params.highlightcolor
  if reference is not None:
    reference = format_images(reference)
    if reference.shape[0:2] != image.shape[0:2]:
      print("Warning, image and reference have different sizes !")
      return dest
    if reference.ndim == 2: reference = np.expand_dims(reference, -1)
    refmask = np.any(reference > 1., axis = 2)
    dest[imgmask & refmask, :] = .5*params.highlightcolor
  return dest

def differences(image, reference, dest = None):
  """Highlight differences between an image and a reference.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    reference (optional): An Image object or numpy.ndarray with shape (height, width, 3)
      (for a color image), (height, width, 1) or (height, width) (for a grayscale image).
    dest (optional): If None (default), the pixels are highlighted on a copy of the input image.
      Otherwise, the pixels are highlighted on a copy of dest, an Image object or numpy.ndarray
      with shape (height, width, 3) (for a color image), (height, width, 1) or (height, width)
      (for a grayscale image).

  Returns:
    numpy.ndarray: A copy of the image or of dest as an array with shape (height, width, 3) and the
    differences highlighted with color `jupyter.params.diffcolor`.
  """
  image, reference = format_images((image, reference))
  if image.shape != reference.shape:
    raise ValueError("Error, image and reference have different sizes/number of channels !")
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  if reference.ndim == 2: reference = np.expand_dims(reference, -1)
  dest = image.copy() if dest is None else format_images(dest, copy = True)
  if dest.ndim == 2: dest = np.expand_dims(dest, axis = -1)
  if dest.shape[2] == 1: dest = np.repeat(dest, 3, axis = 2)
  mask = np.any(image != reference, axis = 2)
  dest[mask, :] = params.diffcolor
  return dest

def stretch(image, median, clip = 3.):
  """Stretch the input image.

  The image is first clipped below max(min(image), avgmed-clip*maxstd), where avgmed is the average
  median of all channels, maxstd the maximum standard deviation of all channels, and clip an input
  factor. It is then stretched with a midtone (aka harmonic) transformation so that the average
  median of all channels matches the target median.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    median (float): The target median of the image.
    clip (float, optional): The clip factor (default 3).

  Returns:
    numpy.ndarray: The stretched image as an array with shape (height, width, 3) (for color images)
    or (height, width) (for grayscale images).
  """
  image = format_images(image)
  avgmed = np.mean(np.median(image, axis = (0, 1)))
  maxstd = np.max(np.std(image, axis = (0, 1)))
  shadow = max(np.min(image), avgmed-clip*maxstd)
  avgmed = shadow_stretch_function(avgmed, shadow)
  midtone = midtone_stretch_function(avgmed, median, False)
  return midtone_stretch_function(shadow_stretch_function(image, shadow), midtone, False)
