# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""eQuimageLab."""

__version__ = "1.1.1"
__packagepath__ = __path__[0]

# Import top-level symbols.

from .equimage.toplevel import *
from .jupyter.toplevel import *
from . import params

print("######################################"+len(__version__)*"#")
print(f"# Welcome to eQuimageLab version {__version__}... #")
print("######################################"+len(__version__)*"#")
print("Documentation available at: https://astro.ymniquet.fr/codes/equimagelab/docs/")

set_RGB_luma("human")
