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
from .backend_plotly import _show_image_, _show_histograms_, _show_statistics_

from ..equimage.image import Image

class Dashboard():

  def __init__(self, interval = 333):
    self.content = []
    self.refresh = False
    self.app = dash.Dash(title = "eQuimageLab dashboard", update_title = None)
    dashboard = dash.html.Div(self.content, id = "dashboard")
    interval = dash.dcc.Interval(id = "update-dashboard", interval = interval, n_intervals = 0)
    self.app.layout = dash.html.Div([dashboard, interval])
    self.app.callback(dash.dependencies.Output("dashboard", "children"), dash.dependencies.Input("update-dashboard", "n_intervals"))(self.__update_dashboard)
    self.app.run_server(debug = False, use_reloader = False, jupyter_mode = "external")

  def __update_dashboard(self, n):
    refresh = self.refresh
    self.refresh = False
    return self.content if refresh else dash.no_update

  def show(self, images, histograms = False, statistics = False, sample = 1, trans = None):
    self.refresh = False
    if isinstance(images, (list, tuple)):
      nimages = len(images)
      if nimages == 1:
        imgtabs = {"Image": images[0]}
      elif nimages == 2:
        imgtabs = {"Image": images[0], "Reference": images[1]}
      else:
        n = 0
        imgtabs = {}
        for image in images:
          n += 1
          imgtabs[f"Image #{n}"] = image
    elif isinstance(images, dict):
      imgtabs = images
    else:
      imgtabs = {"Image": images}
    tabs = []
    for label, image in imgtabs.items():
      tab = []
      tab.append(dash.dcc.Graph(figure = _show_image_(image, sample = sample)))
      if histograms is not False:
        if histograms is True: histograms = ""
        figure = _show_histograms_(image, channels = histograms, log = True, trans = trans if label == "Reference" else None)
        if figure is not None: tab.append(dash.dcc.Graph(figure = figure))
      if statistics is not False:
        if statistics is True: statistics = ""
        table = _show_statistics_(image, channels = statistics)
        if table is not None: tab.append(dash.dcc.Graph(figure = table))
      tabs.append(dash.dcc.Tab(tab, label = label))
    if len(imgtabs) == 1:
      self.content = tabs
    else:
      self.content = [dash.dcc.Tabs(tabs)]
    self.refresh = True

  def show_t(self, image, histograms = "RGBL", sample = 1):
    if not issubclass(type(image), Image):
      print("The transformations can only be displayed for Image objects.")
    trans = getattr(image, "trans", None)
    if trans is None:
      print("There is no transformation embedded in the input image.")
      return
    reference = trans.input
    channels = trans.xlabel
    for c in channels:
      if c in "RGBVSL":
        if c not in histograms:
          histograms += c
    self.show({"Image": image, "Reference": reference}, histograms = histograms, statistics = histograms, trans = trans, sample = sample)

