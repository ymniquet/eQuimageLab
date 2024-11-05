# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Dash backend for Jupyter-lab interface."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "jupyterlab"
import dash

from . import params
from .utils import prepare_images
from .backend_plotly import __show_image, __show_histograms, __show_statistics

from ..equimage.image import Image

def dashboard(image, histograms = False, statistics = False, sample = 1):
  content = []
  content.append(dash.dcc.Graph(figure = __show_image(image, sample, width = params.maxwidth)))
  if histograms is not False:
    if histograms is True:
      histograms = ""
    figure = __show_histograms(image, channels = histograms, log = True, width = params.maxwidth)
    if figure is not None: content.append(dash.dcc.Graph(figure = figure))
  if statistics is not False:
    if statistics is True:
      statistics = ""
      figure = __show_statistics(image, channels = statistics, width = params.maxwidth, rowheight = params.rowheight)
      if figure is not None: content.append(dash.dcc.Graph(figure = figure))
  app = dash.Dash()
  app.layout = dash.html.Div(content)
  app.run_server(debug = False, use_reloader = False, jupyter_mode = "tab")
