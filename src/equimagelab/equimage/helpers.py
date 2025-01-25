# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Image processing helpers."""

import numpy as np

from . import params

class Container:
  """An empty container class."""
  pass # An empty container class.

def fpaccuracy(dtype):
  """Return the expected floating point accuracy for the input float class.

  Args:
    dtype (class): A float class (numpy.float32 or numpy.float64).

  Returns:
    float: The expected floating point accuracy for this class
      (1.e-6 for numpy.float32 and 1.e-9 for numpy.float64).
  """
  if dtype == np.float32:
    return 1.e-6
  elif dtype == np.float64:
    return 1.e-9
  else:
    raise ValueError("Error, the input class must be numpy.float32 or numpy.float64.")

def failsafe_divide(A, B):
  """Return A/B, ignoring errors (division by zero, ...).

  Args:
    A (numpy.ndarray): The numerator array.
    B (numpy.ndarray): The denominator array.

  Returns:
    numpy.ndarray: The element-wise division A/B.
  """
  status = np.seterr(all = "ignore")
  C = np.divide(A, B)
  np.seterr(**status)
  return C

def scale_pixels(image, source, target, cutoff = None):
  """Scale all pixels of the image by the ratio target/source. Wherever abs(source) < cutoff, set all channels to target.

  Args:
    image (numpy.ndarray): The input image.
    source (numpy.ndarray): The source values for scaling (must be the same size as the input image).
    target (numpy.ndarray): The target values for scaling (must be the same size as the input image).
    cutoff (float, optional): Threshold for scaling. If None, defaults to `equimage.helpers.fpaccuracy(source.dtype)`.

  Returns:
    numpy.ndarray: The scaled image.
  """
  if cutoff is None: cutoff = fpaccuracy(source.dtype)
  return np.where(abs(source) > cutoff, failsafe_divide(image*target, source), target)

def lookup(x, xlut, ylut, slut, nlut):
  """Linearly interpolate y = f(x) between the values ylut = f(xlut) of an evenly spaced look-up table.

  Args:
    x (float): The input abscissa for interpolation.
    xlut (numpy.ndarray): The x values of the look-up table (must be evenly spaced).
    ylut (numpy.ndarray): The y values of the look-up table ylut = f(xlut).
    slut (numpy.ndarray): The slopes (ylut[1:]-ylut[:-1])/(xlut[1:]-xlut[:-1]) used for linear interpolation between the elements of ylut.
    nlut (int): The number of elements in the look-up table.

  Returns:
    float: The interpolated value y = f(x).
  """
  i = np.clip(np.int32(np.floor((nlut-1)*(x-xlut[0])/(xlut[-1]-xlut[0]))), 0, nlut-2)
  return slut[i]*(x-xlut[i])+ylut[i]
