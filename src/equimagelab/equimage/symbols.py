# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image processing symbols."""

from PIL import Image as PILImage

# Image resampling methods, imported from PIL.

NEAREST  = PILImage.Resampling.NEAREST
BILINEAR = PILImage.Resampling.BILINEAR
BICUBIC  = PILImage.Resampling.BICUBIC
LANCZOS  = PILImage.Resampling.LANCZOS
BOX      = PILImage.Resampling.BOX
HAMMING  = PILImage.Resampling.HAMMING

