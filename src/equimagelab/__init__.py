# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15
# Sphinx OK.

"""eQuimage lab."""

import os
os.environ["LANGUAGE"] = "en"
import inspect

__version__ = "1.0.0"
__packagepath__ = os.path.dirname(inspect.getabsfile(inspect.currentframe()))

# Import everything.

from .equimage import *
from .jupyter import *

ruler = "######################################"+len(__version__)*"#"
print(ruler)
print(f"# Welcome to eQuimageLab version {__version__}... #")
print(ruler)
# print(f"Documentation available at: file://{__packagepath__}/docs/build/html/")
print("Documentation available at: https://astro.ymniquet.fr/codes/equimagelab/docs/")
print("The module numpy is loaded as np.")

set_RGB_luma("human")
