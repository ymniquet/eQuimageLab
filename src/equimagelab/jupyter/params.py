# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""Jupyter-lab interface parameters."""

import numpy as np

# Images downsampling rate for display.
# Only images[::sampling, ::sampling] are shown to speed up operations.

sampling = 1

# Figure size and margins (pixels).

maxwidth = 1024 # Maximum width of the displayed image.

lmargin = 96  # Left margin.
rmargin = 160 # Right margin.
bmargin = 48  # Bottom margin.
tmargin = 48  # Top margin.

rowheight = 25 # Height of table rows.

# Color of marker lines.

mlinecolor = "mediumslateblue"

# Colors for shadows, highlights and differences.

shadowcolor = np.array([1., .5, 0.])
highlightcolor = np.array([1., 1., 0.])
diffcolor = np.array([1., 1., 0.])

# Setters.

def set_image_sampling(s):
  """Set image downsampling rate for display.

  Args:
    s (int): The image downsampling rate in pixels.
  """
  global sampling
  sampling = s

def set_figure_max_width(w):
  """Set maximum figure width.

  Args:
    w (int): The maximum figure width in pixels.
  """
  global maxwidth
  maxwidth = w

def set_figure_margins(left = None, right = None, bottom = None, top = None):
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

def set_table_row_height(h):
  """Set table row height.

  Args:
    h (int): The table row height in pixels.
  """
  global rowheight
  rowheight = h
