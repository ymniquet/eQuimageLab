# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Utils for Jupyter-lab interface."""

import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

from . import params

from ..equimage import Image
from ..equimage import params as eqparams

def get_image_size(image):
  """Return the width and height of the input image.

  Args:
    image: An Image object or a numpy.ndarray with dimensions (3, height, width) (for
      color images), (1, height, width) or (height, width) (for grayscale images).

  Returns:
    A tuple (width, height) of the image in pixels.
  """
  if issubclass(type(image), Image):
    return image.get_size()
  elif issubclass(type(image), np.ndarray):
    if image.ndim in [2, 3]: return image.shape[-1], image.shape[-2]
  raise ValueError(f"Error, {image} is not a valid image.")

def prepare_images(images, sampling = -1):
  """Prepare images for plotly and Dash.

  Returns all images as numpy.ndarrays with dimensions (3, height, width) (for color images)
  or (1, height, width) (for grayscale images).

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarrays with dimensions (3, height, width)
      (for color images), (1, height, width) or (height, width) (for grayscale images).
    sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
      Only images[:, ::sampling, ::sampling] are processed, to speed up operations.

  Returns:
    All images as float numpy.ndarrays with dimensions (3, height/sampling, width/sampling) (for color images)
      or (1, height/sampling, width/sampling) (for grayscale images). These arrays are references (not
      copies) of the original images when possible.
  """

  def prepare(image):
    """Prepare input image."""
    if issubclass(type(image), Image):
      image = image.get_image()
    elif issubclass(type(image), np.ndarray):
      valid = image.ndim in [2, 3]
      if image.ndim == 3: valid = image.shape[0] in [1, 3]
      valid = valid and image.dtype in [np.float32, np.float64]
      if not valid: raise ValueError(f"Error, image {image} is not a valid image.")
    if image.ndim == 2: image = np.expand_dims(image, axis = 0)
    if sampling > 1: image = image[:, ::sampling, ::sampling]
    return image

  if sampling <= 0: sampling = params.sampling
  if not isinstance(images, (tuple, list)): return prepare(images)
  return type(images)(prepare(image) for image in images)

def prepare_images_as_b64strings(images, sampling = -1):
  """Prepare images as PNGs encoded in base64 strings.

  Returns all images as PNGs encoded in base64 strings.

  Args:
    images: A single/tuple/list of Image object(s) or numpy.ndarrays with dimensions (3, height, width)
      (for color images), (1, height, width) or (height, width) (for grayscale images).
    sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
      Only images[:, ::sampling, ::sampling] are processed, to speed up operations.

  Returns:
    All args as PNGs encoded in base64 strings.
  """

  def prepare(image):
    """Prepare input image."""
    if issubclass(type(image), Image):
      image = image.get_image()
    elif issubclass(type(image), np.ndarray):
      valid = image.ndim in [2, 3]
      if image.ndim == 3: valid = image.shape[0] in [1, 3]
      valid = valid and image.dtype in [np.float32, np.float64]
      if not valid: raise ValueError(f"Error, image {image} is not a valid image.")
    if image.ndim == 2: image = np.expand_dims(image, axis = 0)
    data = np.rint(np.clip(image[:, ::sampling, ::sampling]*255, 0, 255)).astype("uint8")
    if data.shape[0] == 1:
      PILimg = PILImage.fromarray(data[0, :, :])
    else:
      PILimg = PILImage.fromarray(np.moveaxis(data, 0, -1))
    buffer = BytesIO()
    PILimg.save(buffer, format = "PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("utf-8")

  if sampling <= 0: sampling = params.sampling
  if not isinstance(images, (tuple, list)): return prepare(images)
  return type(images)(prepare(image) for image in images)

def filter(image, channels):
  """Filter the channels of an image.

  Returns a copy of the image with unselected red/green/blue channels set to zero.

  Args:
    image: The image (Image object or numpy.ndarray).
    channels (str): The selected channels. A combination of the letters "R" (red),
      "G" (green), and "B" (blue).

  Returns:
    np.ndarray: A copy of the image as an array with dimensions (3, height, width)
      and the unselected channels set to zero.
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
  image = prepare_images(image)
  if image.shape[0] != 3: raise ValueError("Error, the input must be a RGB (not grayscale) image.""")
  output = image.copy()
  output[~selected] = 0.
  return output

def shadowed(image, reference = None):
  """Highlight black pixels in an image.

  Highlight pixels black on the input image with color params.shadowcolor.
  If a reference image is provided, highlight pixels black on both input and reference images
  with color 0.5*params.shadowcolor.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference (optional): The reference image (Image object or numpy.ndarray, default None).

  Returns:
    A copy of the image as an array with dimensions (3, height, width)
      and the black pixels highlighted with color params.shadowcolor.
  """
  image = prepare_images(image)
  imgmask = np.all(image < eqparams.IMGTOL, axis = 0)
  output = image.copy()
  output[:, imgmask] = params.shadowcolor
  if reference is not None:
    reference = prepare_images(reference)
    refmask = np.all(reference < eqparams.IMGTOL, axis = 0)
    output[:, imgmask & refmask] = .5*params.shadowcolor
  return output

def highlighted(image, reference = None):
  """Highlight saturated pixels in an image.

  A pixel is saturated if at least one channel is >= 1.
  Show pixels saturated on the input image with color params.highlightcolor.
  If a reference image is provided, show pixels saturated on both input and reference images
  with color 0.5*params.highlightcolor.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference (optional): The reference image (Image object or numpy.ndarray, default None).

  Returns:
    A copy of the image as an array with dimensions (3, height, width)
      and the saturated pixels highlighted with color params.highlightcolor.
  """
  image = prepare_images(image)
  imgmask = np.any(image > 1.-eqparams.IMGTOL, axis = 0)
  output = image.copy()
  output[:, imgmask] = params.highlightcolor
  if reference is not None:
    reference = prepare_images(reference)
    refmask = np.any(reference > 1.-eqparams.IMGTOL, axis = 0)
    output[:, imgmask & refmask] = .5*params.highlightcolor
  return output

def differences(image, reference):
  """Highlight differences between an image and a reference.

  Args:
    image: The image (Image object or numpy.ndarray).
    reference: The reference image (Image object or numpy.ndarray).

  Returns:
    A copy of the image as an array with dimensions (3, height, width)
      and the differences with the reference highlighted with color params.diffcolor.
  """
  image, reference = prepare_images((image, reference))
  mask = np.any(image != reference, axis = 0)
  output = img.copy()
  output[:, mask] = params.diffcolor
  return output
