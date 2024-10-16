# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image processing parameters."""

import numpy as np

# Data type used for images (either np.float32 or np.float64).

IMGTYPE = np.float32

# Expected accuracy in np.float32/np.float64 calculations.

IMGTOL = 1.e-6 if IMGTYPE is np.float32 else 1.e-9

# Use imageio or imread module ?

IMAGEIO = False

# Weights of the RGB components in the luma.

global rgbluma
rgbluma = IMGTYPE((.3, .6, .1))
