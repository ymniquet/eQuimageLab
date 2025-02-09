# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.2.0 / 2025.02.02
# Sphinx OK.

"""Image processing tools."""

# Import top-level symbols.

from .imports import *

# NB : Here is the list of functions which explicitely deal with channels.
# Please crosscheck/update these functions when adding new channels.
#  - Image.get_channel
#  - Image.set_channel
#  - Image.apply_channel
#  - Image.histograms
#  - Image.statistics
#  - Image.statistical_stretch
#  - Image.LDBS
#  - Image.filter

# TODO:
#
# get_channel, set_channel:
#  - Add a*, b*, u*, v*, c*, s*, h* for CIELab and CIELuv images.
#
# apply_channels:
#  - L* : Apply to the lightness @ constant hue & chroma in the CIELab or CIELuv color space.
#         Equivalent to L*/a*b* for RGB images.
#  - L*/a*b* : Apply to the lightness @ constant a*/b* (constant CIELab hue & chroma) in the CIELab color space.
#  - L*/u*v* : Apply to the lightness @ constant u*/v* (constant CIELuv hue & chroma) in the CIELuv color space.
#  - L*/s*h* : Apply to the lightness @ constant hue & saturation in the CIELuv color space.
#
# CIE_color_saturation.
