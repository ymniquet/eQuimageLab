# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Utils for Jupyter Lab interface."""

import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

from . import params

from ..equimage import Image

def get_image_size(image):
  """Return the width and height of the input image.

  Args:
    image: An Image object or a numpy.ndarray with dimensions (height, width, 3) (for
      color images), (height, width, 1) or (height, width) (for grayscale images).

  Returns:
    A tuple (width, height) of the image in pixels.
  """
  if issubclass(type(image), Image):
    return image.get_size()
  elif issubclass(type(image), np.ndarray):
    if image.ndim in [2, 3]: return image.shape[1], image.shape[0]
  raise ValueError(f"Error, {image} is not a valid image.")

def prepare_images(images, sampling = -1, copy = False):
  """Prepare images for plotly and Dash.

  Returns all images as numpy.ndarrays with dimensions (height, width, 3) (for color images),
  or (height, width) (for grayscale images).

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarrays with dimensions (height, width, 3)
      (for color images), (height, width, 1) or (height, width) (for grayscale images).
    sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
      Only images[::sampling, ::sampling] are processed, to speed up operations.
    copy (bool, optional): If False (default), the output images are (when possible) views of the
      original images; If True, they are always copies.

  Returns:
    A single/tuple/list of float numpy.ndarrays with dimensions (height/sampling, width/sampling, 3) (for
    color images) or (height/sampling, width/sampling) (for grayscale images).
  """

  def prepare(image, copy):
    """Prepare input image."""
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
  if not isinstance(images, (tuple, list)): return prepare(images, copy)
  return type(images)(prepare(image, copy) for image in images)

def prepare_images_as_b64strings(images, sampling = -1, compression = 4):
  """Prepare images for plotly and Dash as PNGs encoded in base64 strings.

  Returns all images as PNGs encoded in base64 strings.

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarrays with dimensions (height, width, 3)
      (for color images), (height, width, 1) or (height, width) (for grayscale images).
    sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
      Only images[::sampling, ::sampling] are processed, to speed up operations.
    compression (int, optional): PNG compression level (default 4).

  Returns:
    A single/tuple/list of images as PNGs encoded in base64 strings.
  """

  def prepare(image):
    """Prepare input image."""
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
  if not isinstance(images, (tuple, list)): return prepare(images)
  return type(images)(prepare(image) for image in images)

def filter(image, channels):
  """Filter the channels of a RGB image.

  Returns a copy of the image with selected red/green/blue channels set to zero.

  Args:
    image: The image (Image object or numpy.ndarray with dimension (height, width, 3)).
    channels (str): The *displayed* channels. A combination of the letters "R" (red),
      "G" (green), and "B" (blue).

  Returns:
    numpy.ndarray: A copy of the image as an array with dimensions (height, width, 3) and the
    non-displayed channels set to zero.
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
      raise ValueError(f"Error, unknown channel {c}.")
  image = prepare_images(image, sampling = 1, copy = True)
  if image.ndim != 3: raise ValueError("Error, the input must be a RGB (not a grayscale) image.""")
  image[:, :, ~selected] = 0.
  return image

def shadowed(image, reference = None):
  """Highlight black pixels in an image.

  Highlight black pixels on the input image with color `jupyter.params.shadowcolor`.
  If a reference image is provided, highlight pixels black on both input and reference images
  with color 0.5 * `jupyter.params.shadowcolor`.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference (optional): The reference image (Image object or numpy.ndarray, default None).

  Returns:
    numpy.ndarray: A copy of the image as an array with dimensions (height, width, 3) and the
    black pixels highlighted with color `jupyter.params.shadowcolor`.
  """
  image = prepare_images(image, sampling = 1, copy = True)
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  imgmask = np.all(image < params.IMGTOL, axis = 2)
  if image.shape[2] == 1: image = np.repeat(image, 3, axis = 2)
  image[imgmask, :] = params.shadowcolor
  if reference is not None:
    reference = prepare_images(reference, sampling = 1)
    if reference.shape[0:2] != image.shape[0:2]:
      print("Warning, image and reference have different sizes !")
      return image
    if reference.ndim == 2: reference = np.expand_dims(reference, -1)
    refmask = np.all(reference < params.IMGTOL, axis = 2)
    image[imgmask & refmask, :] = .5*params.shadowcolor
  return image

def highlighted(image, reference = None):
  """Highlight saturated pixels in an image.

  A pixel is saturated if at least one channel is >= 1.
  Show pixels saturated on the input image with color `jupyter.params.highlightcolor`.
  If a reference image is provided, show pixels saturated on both input and reference images
  with color 0.5 * `jupyter.params.highlightcolor`.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference (optional): The reference image (Image object or numpy.ndarray, default None).

  Returns:
    numpy.ndarray: A copy of the image as an array with dimensions (height, width, 3) and the saturated
    pixels highlighted with color `jupyter.params.highlightcolor`.
  """
  image = prepare_images(image, sampling = 1, copy = True)
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  imgmask = np.any(image > 1.-params.IMGTOL, axis = 2)
  if image.shape[2] == 1: image = np.repeat(image, 3, axis = 2)
  image[imgmask, :] = params.highlightcolor
  if reference is not None:
    reference = prepare_images(reference, sampling = 1)
    if reference.shape[0:2] != image.shape[0:2]:
      print("Warning, image and reference have different sizes !")
      return image
    if reference.ndim == 2: reference = np.expand_dims(reference, -1)
    refmask = np.any(reference > 1.-params.IMGTOL, axis = 2)
    image[imgmask & refmask, :] = .5*params.highlightcolor
  return image

def differences(image, reference):
  """Highlight differences between an image and a reference.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference: The reference image (Image object or numpy.ndarray).

  Returns:
    numpy.ndarray: A copy of the image as an array with dimensions (height, width, 3)
    and the differences with the reference highlighted with color `jupyter.params.diffcolor`.
  """
  image = prepare_images(image, sampling = 1, copy = True)
  reference = prepare_images(reference, sampling = 1)
  if image.shape != reference.shape:
    raise ValueError("Error, image and reference have different sizes/number of channels !")
  if image.ndim == 2: image = np.expand_dims(image, axis = -1)
  if reference.ndim == 2: reference = np.expand_dims(reference, -1)
  mask = np.any(image != reference, axis = 2)
  if image.shape[2] == 1: image = np.repeat(image, 3, axis = 2)
  image[mask, :] = params.diffcolor
  return image
