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
from dash import Dash, dcc, html
import dash_bootstrap_templates as dbt
import dash_bootstrap_components as dbc

from . import params
from .utils import prepare_images
from .backend_plotly import _figure_image_, _figure_histograms_

from ..equimage.image import Image

class Dashboard():
  """Dashboad class."""

  def __init__(self, interval = 333):
    """Initialize dashboard.

    This dashboard uses Dash to displays image, histograms, statistics, etc... in a
    separate browser tab or window.
    It fetches updates from the Dash server every input interval.

    Args:
      interval (int, optional): The time interval (ms, default 333) between dashboard updates.
    """
    self.content = []
    self.refresh = False
    dbt.load_figure_template("slate")
    self.app = Dash(title = "eQuimageLab dashboard", update_title = None, external_stylesheets = [dbc.themes.SLATE])
    dashboard = html.Div(self.content, id = "dashboard", style = {"width": params.maxwidth+params.lmargin+params.rmargin})
    interval = dcc.Interval(id = "update-dashboard", interval = interval, n_intervals = 0)
    self.app.layout = html.Div([dashboard, interval])
    self.app.callback(dash.dependencies.Output("dashboard", "children"), dash.dependencies.Input("update-dashboard", "n_intervals"))(self.__update_dashboard)
    self.app.run_server(debug = False, use_reloader = False, jupyter_mode = "external")

  def __update_dashboard(self, n):
    """The callback for dashboard updates."""
    refresh = self.refresh
    if refresh: self.refresh = False
    return self.content if refresh else dash.no_update

  def show(self, images, histograms = False, statistics = False, sample = 1, trans = None):
    """Show image(s) on the dashboard.

    Args:
      images: The image(s) as a single/tuple/list/dictionary of Image object(s) or numpy.ndarray.
        If a tuple/list/dictionary, each image is displayed in a separate tab.
        The tabs are named according to the keys for a dictionary. Otherwise, the tabs are named "Image"
        and "Reference" if there are two images, and "Image #1", "Image #2"... if there are more.
        The images must all be Image objects if histograms or statistics is True.
      histograms (optional): If True or a string, show the histograms of the image(s). The string lists the
        channels of the histograms (e.g. "RGBL" for red, green, blue, luma). Default is False.
      statistics (optional): If True or a string, show the statistics of the image(s). The string lists the
        channels of the statistics (e.g. "RGBL" for red, green, blue, luma). Default is False.
      sample (int, optional): Downsampling rate (default 1).
        Only images[:, ::sample, ::sample] are shown, to speed up operations.
      trans (optional): A container with an histogram transformation (see Image.apply_channels), plotted on
        top of the histograms of the "Reference" tab (default None).
    """
    self.refresh = False
    if isinstance(images, (tuple, list)):
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
      tab.append(dcc.Graph(figure = _figure_image_(image, sample = sample, width = params.maxwidth, template = "slate")))
      if histograms is not False:
        if histograms is True: histograms = ""
        figure = _figure_histograms_(image, channels = histograms, log = True, trans = trans if label == "Reference" else None,
                                     width = params.maxwidth, template = "slate")
        if figure is not None: tab.append(dcc.Graph(figure = figure))
      if statistics is not False:
        if statistics is True: statistics = ""
        table = _table_statistics_(image, channels = statistics)
        if table is not None: tab.append(table)
      tabs.append(dbc.Tab(tab, label = label))
    if len(imgtabs) == 1:
      self.content = tabs
    else:
      self.content = [dbc.Tabs(tabs)]
    self.refresh = True

  def show_t(self, image, channels = "RGBL", sample = 1):
    """Show the input and output images of an histogram transformation on the dashboard.

    Displays the input image, histograms, statistics, and transformation curve in tab "Reference",
    and the output image, histograms, and statistics in tab "Image".

    Args:
      image (Image): The output image (must embed a transformation image.trans - see Image.apply_channels).
      channels (str, optional): The channels of the histograms and statistics (default "RGBL" for red,
        green, blue, luma). The channels of the transformation are added if needed.
      sample (int, optional): Downsampling rate (default 1).
        Only image[:, ::sample, ::sample] is shown, to speed up operations.
    """
    if not issubclass(type(image), Image):
      print("The transformations can only be displayed for Image objects.")
    trans = getattr(image, "trans", None)
    if trans is None:
      print("There is no transformation embedded in the input image.")
      return
    reference = trans.input
    for c in trans.channels:
      if c in "RGBVSL":
        if c not in channels:
          channels += c
    self.show({"Image": image, "Reference": reference}, histograms = channels, statistics = channels, trans = trans, sample = sample)

def _table_statistics_(image, channels = ""):
  """Prepare a table with the statistics of an image.

  Args:
    image (Image): The image.
    channels (str, optional): The channels of the histograms (default "" = "RGBL" for red, green, blue, luma).

  Returns:
    dbc.Table: A dash bootstrap components table with the statistics of the image.
  """
  # Prepare statistics.
  if not issubclass(type(image), Image):
    print("The statistics can only be displayed for Image objects.")
    return None
  if channels == "":
    stats = getattr(image, "stats", None)
    if stats is None: stats = image.statistics()
  else:
    stats = image.statistics(channels = channels)
  # Create table.
  header = [html.Thead(html.Tr([html.Th("Channel", style = {"text-align": "left"}), html.Th("Minimum"), html.Th("25%"), html.Th("50%"),
                                html.Th("75%"), html.Th("Maximum"), html.Th("Shadowed"), html.Th("Highlighted")]))]
  rows = []
  for channel in stats.values():
    rows.append(html.Tr([html.Td(channel.name, style = {"text-align": "left"}), html.Td(f"{channel.minimum:.5f}"),
                         html.Td(f"{channel.percentiles[0]:.5f}"), html.Td(f"{channel.percentiles[1]:.5f}"),
                         html.Td(f"{channel.percentiles[2]:.5f}"), html.Td(f"{channel.maximum:.5f}"),
                         html.Td(f"{channel.zerocount} ({100.*channel.zerocount/channel.npixels:.2f}%)"),
                         html.Td(f"{channel.outcount} ({100.*channel.outcount/channel.npixels:.2f}%)")]))
  body = [html.Tbody(rows)]
  table = dbc.Table(header+body, size = "sm", bordered = True, striped = True, style = {"text-align": "right",
                    "width": f"{params.maxwidth}px", "margin": f"32px {params.rmargin}px 32px {params.lmargin}px"})
  return table
