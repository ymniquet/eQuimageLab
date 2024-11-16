# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image I/O management."""

import os
import numpy as np

from . import params
from .image import Image

from PIL import Image as PILImage
if params.IMAGEIO:
  import imageio.v3 as iio
else:
  import skimage.io as skio
import astropy.io.fits as pyfits

def load_image(filename):
  """Load an image from a file.

  Note: The color space is assumed to be sRGB and the color model "RGB" or "gray".

  Args:
    filename (str): The file name.

  Returns:
    The image as an Image object and the file meta-data (including exif if available) as a dictionary.
  """
  print(f"Loading file {filename}...")
  try:
    header = PILImage.open(filename)
    fmt = header.format
    print(f"Format = {fmt}.")
  except:
    header = None
    fmt = None
    print("Failed to identify image file format; Attempting to load anyway...")
  if fmt == "PNG": # Load with the FreeImage plugin to enable 16 bits color depth.
    image = iio.imread(filename, plugin = "PNG-FI") if params.IMAGEIO else skio.imread(filename)
  elif fmt == "TIFF":
    image = iio.imread(filename, plugin = "TIFF") if params.IMAGEIO else skio.imread(filename, plugin = "tifffile")
  elif fmt == "FITS":
    hdus = pyfits.open(filename)
    image = hdus[0].data
    if image.ndim == 3:                 # Pyfits returns (channels, height, width)
      image = np.moveaxis(image, 0, -1) # instead of (height, width, channels),
    image = np.flip(image, axis = 0)    # and an upside down image.
  else:
    image = iio.imread(filename) if params.IMAGEIO else skio.imread(filename)
  if image.ndim == 2: # Assume single channel images are monochrome.
    nc = 1
    image = np.expand_dims(image, axis = -1)
  elif image.ndim == 3:
    nc = image.shape[2]
  else:
    raise ValueError(f"Error, invalid image shape = {image.shape}.")
  print(f"Image size = {image.shape[1]}x{image.shape[0]} pixels.")
  print(f"Number of channels = {nc}.")
  if nc not in [1, 3, 4]: raise ValueError(f"Error, images with {nc} channels are not supported.")
  dtype = str(image.dtype)
  print(f"Data type = {dtype}.")
  if dtype == "uint8":
    bpc = 8
    image = params.IMGTYPE(image/255)
  elif dtype == "uint16":
    bpc = 16
    image = params.IMGTYPE(image/65535)
  elif dtype == "uint32":
    bpc = 32
    image = params.IMGTYPE(image/4294967295)
  elif dtype in ["float32", ">f4", "<f4"]: # Assumed normalized in [0, 1] !
    bpc = 32
    image = params.IMGTYPE(image)
  elif dtype in ["float64", ">f8", "<f8"]: # Assumed normalized in [0, 1] !
    bpc = 64
    image = params.IMGTYPE(image)
  else:
    raise TypeError(f"Error, image data type {dtype} is not supported.")
  print(f"Bit depth per channel = {bpc}.")
  print(f"Bit depth per pixel = {nc*bpc}.")
  image = np.moveaxis(image, -1, 0) # Move last (channel) axis to leading position.
  for ic in range(nc):
    print(f"Channel #{ic}: minimum = {image[ic].min():.5f}, maximum = {image[ic].max():.5f}.")
  if nc == 4: image = image[0:3]*image[3] # Assume fourth channel is transparency.
  try:
    exif = header.getexif()
    print("Succesfully read EXIF data...")
  except:
    exif = None
  meta = {"exif": exif, "colordepth": bpc}
  return Image(np.ascontiguousarray(image)), meta

def save_image(image, filename, depth = 8):
  """Save image as a file.

  Note: The color model must be "RGB" or "gray", but the color space is *not* embedded
    in the file at present.

  Args:
    image (Image): The image.
    depth (int, optional): The color depth of the file in bits/channel (default 8).
    filename (str): The file name. The file format is chosen according to the extension:
      - .png: PNG file with depth = 8 or 16 bits/channel.
      - .tif, .tiff: TIFF file with depth = 8, 16 (integers), or 32 (floats) bits/channel.
      - .fit, .fits, .fts: FITS file with 32 bits (floats)/channel (irrespective of depth).
  """
  image.check_color_model("RGB", "gray")
  is_gray = (image.colormodel == "gray")
  if is_gray:
    print(f"Saving grayscale image as file {filename}...")
  else:
    print(f"Saving RGB image as file {filename}...")
  root, ext = os.path.splitext(filename)
  if ext in [".png", ".tif", ".tiff"]:
    if depth == 8:
      image = image.int8()
    elif depth == 16:
      image = image.int16()
    elif depth == 32:
      image = image.int32()
    else:
      raise ValueError("Error, color depth must be 8 or 16, or 32 bits per channel.")
    print(f"Color depth = {depth} bits per channel (integers).")
    if is_gray: image = image[:, :, 0]
    if ext == ".png":
      if params.IMAGEIO:
        if depth > 16: raise ValueError("Error, color depth of png files must be 8 or 16 bits per channel.")
        iio.imwrite(filename, image, plugin = "PNG-FI")
      else:
        if depth > 8: raise ValueError("Error, color depth of png files must be 8 bits per channel.")
        skio.imsave(filename, image, check_contrast = False)
    elif ext == ".tif" or ext == ".tiff":
      if params.IMAGEIO:
        iio.imwrite(filename, image, plugin = "TIFF", metadata = {"compress": 5})
      else:
        skio.imsave(filename, image, plugin = "tifffile", check_contrast = False, compression = "zlib")
  elif ext in [".fit", ".fits", ".fts"]:
    print(f"Color depth = {np.finfo(image.dtype).bits} bits per channel (floats).")
    image = image.flip_height().get_image() # Flip image upside down.
    if is_gray: image = image[0]
    hdu = pyfits.PrimaryHDU(image)
    hdu.writeto(filename, overwrite = True)
  else:
    raise ValueError("Error, file extension must be .png or .tif/.tiff., or .fit/.fits/.fts.")
