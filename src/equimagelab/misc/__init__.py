# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.3.1 / 2025.03.26
# Doc OK.

"""Miscellaneous functions (test images, ...)."""

import os
from .. import equimage
from .. import __packagepath__

def HSV_wheel():
  """Return a HSV wheel as an Image object, to test color transformations.

  Returns:
    Image: An image object with a HSV wheel.
  """
  image, meta = equimage.load_image(os.path.join(__packagepath__, "images", "HSVwheel.png"), verbose = False)
  return image
