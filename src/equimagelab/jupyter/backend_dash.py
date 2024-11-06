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
from .backend_plotly import _show_image_, _show_histograms_

from ..equimage.image import Image

class Dashboard():

  def __init__(self):
    self.refresh = False
    self.interval = dash.dcc.Interval(id = "update-dashboard", interval = 333, n_intervals = 0)
    self.content = [self.interval]
    self.app = dash.Dash(title = "eQuimageLab dashboard", update_title = None)
    self.app.layout = dash.html.Div(self.content, id = "dashboard")
    self.app.callback(dash.dependencies.Output("dashboard", "children"), dash.dependencies.Input("update-dashboard", "n_intervals"))(self.__update_dashboard)
    self.app.run_server(debug = False, use_reloader = False, jupyter_mode = "external")

  def __update_dashboard(self, n):
    refresh = self.refresh
    self.refresh = False
    return self.content if refresh else dash.no_update

  def update(self, image, histograms = False, statistics = False, sample = 1):
    self.refresh = False
    content = [self.interval]
    content.append(dash.dcc.Graph(figure = _show_image_(image, sample, width = params.maxwidth)))
    if histograms is not False:
      if histograms is True:
        histograms = ""
      figure = _show_histograms_(image, channels = histograms, log = True, width = params.maxwidth)
      if figure is not None: content.append(dash.dcc.Graph(figure = figure))
    if statistics is not False:
      if statistics is True:
        statistics = ""
        table = _show_statistics_(image, channels = statistics, width = params.maxwidth, rowheight = params.rowheight)
        if table is not None: content.append(dash.dcc.Graph(figure = table))
    self.content = content
    self.refresh = True

def _show_statistics_(image, channels, width, rowheight):
  if not issubclass(type(image), Image):
    print("The histograms can only be computed for Image objects.")
    return None
  if channels == "":
    stats = getattr(image, "stats", None)
    if stats is None: stats = image.statistics()
  else:
    stats = image.statistics(channels = channels)
  columns = [[], [], [], [], [], [], [], []]
  for channel in stats.values():
    columns[0].append(channel.name)
    columns[1].append(f"{channel.minimum:.5f}")
    columns[2].append(f"{channel.percentiles[0]:.5f}")
    columns[3].append(f"{channel.percentiles[1]:.5f}")
    columns[4].append(f"{channel.percentiles[2]:.5f}")
    columns[5].append(f"{channel.maximum:.5f}")
    columns[6].append(f"{channel.zerocount} ({100.*channel.zerocount/channel.npixels:.2f}%)")
    columns[7].append(f"{channel.outcount} ({100.*channel.outcount/channel.npixels:.2f}%)")
  align = ["left"]+7*["right"]
  header = dict(values = ["Channel", "Minimum", "25%", "50%", "75%", "Maximum", "Shadowed", "Highlighted"], align = align, height = rowheight)
  cells = dict(values = columns, align = align, height = rowheight)
  table = go.Table(header = header, cells = cells, columnwidth = [1, 1, 1, 1, 1, 1, 1.5, 1.5])
  figure = go.Figure(data = table)
  layout = go.Layout(width = width+params.lmargin+params.rmargin, height = (len(stats)+1)*rowheight+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True))
  figure.update_layout(layout)
  return figure
