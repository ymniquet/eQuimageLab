# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.4.1 / 2025.05.30
# Doc OK.

"""eQuimageLab parameters.

.. code-block:: python

  from equimage.params import get_image_type, set_image_type, get_CIE_params, set_CIE_params, set_max_hist_bins, set_default_hist_bins
  from equimagelab.jupyter.params import set_image_sampling, set_figure_max_width, set_figure_margins, set_table_row_height
"""

# Import relevant parameters from the eQuimage and jupyter modules.

from equimage.params import get_image_type, set_image_type, get_CIE_params, set_CIE_params, set_max_hist_bins, set_default_hist_bins
from .jupyter.params import set_image_sampling, set_figure_max_width, set_figure_margins, set_table_row_height
