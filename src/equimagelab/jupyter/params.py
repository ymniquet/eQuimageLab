# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Jupyter-lab interface parameters."""

import numpy as np

from ..equimage.params import IMGTYPE, IMGTOL

# Figure size and margins.

maxwidth = 1024 # Maximum width of the displayed image.

lmargin = 96  # Left margin.
rmargin = 160 # Right margin.
bmargin = 48  # Bottom margin.
tmargin = 48  # Top margin.

rowheight = 25 # Height of the table rows.

# Colors for shadows, highlights and differences.

shadowcolor = np.array([[1.], [.5], [0.]], dtype = IMGTYPE)
highlightcolor = np.array([[1.], [1.], [0.]], dtype = IMGTYPE)
diffcolor = np.array([[1.], [1.], [0.]], dtype = IMGTYPE)

# Setters.

def set_max_width(width):
  """Set maximum figure width.

  Args:
    width (int): The maximum figure width in pixels.
  """
  global maxwidth
  maxwidth = width

def set_margins(left = None, right = None, bottom = None, top = None):
  """Set or update figure margins.

  Args:
    left (int, optional): The left margin in pixels [if not None (default)].
    right (int, optional): The right margin in pixels [if not None (default)].
    bottom (int, optional): The bottom margin in pixels [if not None (default)].
    top (int, optional): The top margin in pixels [if not None (default)].
  """
  global lmargin, rmargin, tmargin, bmargin
  if left is not None: lmargin = left
  if right is not None: rmargin = right
  if bottom is not None: bmargin = bottom
  if top is not None: tmargin = top

def set_row_height(height):
  """Set table row height.

  Args:
    height (int): The table row height in pixels.
  """
  global rowheight
  rowheight = height
