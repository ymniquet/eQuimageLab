# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Image processing parameters."""

import numpy as np

# Data type used for images (either np.float32 or np.float64).

IMGTYPE = np.float32

# Expected accuracy in np.float32/np.float64 calculations.

IMGTOL = 1.e-6 if IMGTYPE is np.float32 else 1.e-9

# Use imageio or imread module ?

IMAGEIO = False

# Exclude pixels <= 0 or >= 1 from the percentiles in image statistics ?

exclude01 = False

# Number of x mesh points for transformation function plots y = f(x in [0, 1]).

ntranslo = 128 # Low  resolution.
ntranshi = 256 # High resolution.

# Number of bins in the histograms.

maxhistbins = 8192 # Maximum number of bins within [0, 1].
defhistbins = 0    # Default number of bins within [0, 1] (see set_default_hist_bins).

def set_max_hist_bins(n):
  """Set the maximum number of bins in the histograms.

  Args:
    n (int): The maximum number of bins within [0, 1].
  """
  global maxhistbins
  maxhistbins = n

def set_default_hist_bins(n):
  """Set the default number of bins in the histograms.

  Args:
    n (int): If strictly positive, the default number of bins within [0, 1].
      (practically limited to `equimage.params.maxhistbins`). If zero, the number
      of bins is computed according to the statistics of each image. If strictly
      negative, the number of bins is set to `equimage.params.maxhistbins`.
  """
  global defhistbins
  defhistbins = n if n >= 0 else maxhistbins

# Weights of the RGB components in the luma.

rgbluma = IMGTYPE((.299, .587, .114))

def get_RGB_luma():
  """Return the RGB weights rgbluma of the luma.

  The luma L of an image is the average of the RGB components weighted by rgbluma:

    L = rgbluma[0]*image[0]+rgbluma[1]*image[1]+rgbluma[2]*image[2]

  Returns:
    tuple: The (red, blue, green) weights rgbluma of the luma.
  """
  return rgbluma

def set_RGB_luma(rgb):
  """Set the RGB weights of the luma.

  Args:
    rgb: The RGB weights of the luma as:

      - a tuple, list or array of the (red, green, blue) weights. They will be normalized so that their sum is 1.
      - the string "uniform": the RGB weights are set to (1/3, 1/3, 1/3).
      - the string "human": the RGB weights are set to (.299, .587, .114). The luma is then the luminance for lRGB images,
        and an approximate substitute for the lightness for sRGB images.
  """
  if isinstance(rgb, str):
    if rgb == "uniform":
      set_RGB_luma((1./3., 1./3., 1./3.))
    elif rgb == "human":
      set_RGB_luma((.299, .587, .114))
    else:
      raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
  else:
    w = np.array(rgb, dtype = IMGTYPE)
    if w.shape != (3,): raise ValueError("Error, the input rgb weights must be an array with three scalar elements, the string 'uniform' or the string 'human'.")
    if any(w < 0.): raise ValueError("Error, the input rgb weights must be >= 0.")
    s = np.sum(w)
    if s == 0.: raise ValueError("Error, the sum of the input rgb weights must be > 0.")
    global rgbluma ; rgbluma = w/s
    print(f"Luma = {rgbluma[0]:.3f}R+{rgbluma[1]:.3f}G+{rgbluma[2]:.3f}B.")
