# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Utils for Jupyter-lab interface."""

import numpy as np

from . import params

from ..equimage.image import Image

def prepare_images(*args, sample = 1):
  output = ()
  for arg in args:
    valid = False
    if issubclass(type(arg), Image):
      img = arg.get_image()
      valid = True
    elif issubclass(type(arg), np.ndarray):
      img = arg
      valid = img.ndim in [2, 3]
      if img.ndim == 3: valid = img.shape[0] in [1, 3]
      valid = valid and img.dtype in [np.float32, np.float64]
    if not valid:
      raise ValueError(f"Error, arg {arg} is not a valid image.")
    if img.ndim == 2:
      img = np.expand_dims(img, axis = 0)
    if sample > 1:
      img = img[:, ::sample, ::sample]
    output += (img, )
  return output[0] if len(output) == 1 else output

def filter(image, channels):
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
  img = prepare_images(image)
  if img.shape[0] != 3:
    raise ValueError("Error, the input must be a RGB (not grayscale) image.""")
  output = img.copy()
  output[~selected] = 0.
  return output

def shadowed(image, reference = None):
  if reference is None:
    img = prepare_images(image)
  else:
    img, ref = prepare_images(image, reference)
  output = img.copy()
  imgmask = np.all(img < params.IMGTOL, axis = 0)
  output[:, imgmask] = params.shadowcolor
  if reference is not None:
    refmask = np.all(ref < params.IMGTOL, axis = 0)
    output[:, imgmask & refmask] = .5*params.shadowcolor
  return output

def highlighted(image, reference = None):
  if reference is None:
    img = prepare_images(image)
  else:
    img, ref = prepare_images(image, reference)
  output = img.copy()
  imgmask = np.any(img > 1.-params.IMGTOL, axis = 0)
  output[:, imgmask] = params.highlightcolor
  if reference is not None:
    refmask = np.any(ref > 1.-params.IMGTOL, axis = 0)
    output[:, imgmask & refmask] = .5*params.highlightcolor
  return output

def differences(image, reference):
  """Highlight differences between image and reference with color params.DIFFCOLOR."""
  img, ref = prepare_images(image, reference)
  output = img.copy()
  mask = np.any(img != ref, axis = 0)
  output[:, mask] = params.diffcolor
  return output

