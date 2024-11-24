# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Dash backend for Jupyter-lab interface."""

# TODO:
#  - Bind zooms across tabs.
#  - Update tabs only if necessary.
#  - Shadows/Highlights/Differences.

import os
import threading
import numpy as np
import dash
from dash import Dash, dcc, html
import dash_bootstrap_templates as dbt
import dash_bootstrap_components as dbc
import dash_extensions as dxt

from . import params
from .utils import prepare_images, prepare_images_as_b64strings
from .backend_plotly import _figure_image_, _figure_histograms_

from ..equimage import Image, load_image

class Dashboard():
  """Dashboad class."""

  def __init__(self, interval = 333):
    """Initialize dashboard.

    This dashboard uses Dash to displays image, histograms, statistics, etc... in a
    separate browser tab or window.
    It fetches updates from the Dash server at given intervals.

    Args:db = eql.Dashboard()
      interval (int, optional): The time interval (ms, default 333) between dashboard updates.
    """
    from .. import __packagepath__
    # Initialize data.
    self.content = []
    self.refresh = False
    self.interval = interval
    self.updatelock = threading.Lock()
    # Set-up Dash app.
    dbt.load_figure_template("slate")
    self.app = Dash(title = "eQuimageLab dashboard", update_title = None, external_stylesheets = [dbc.themes.SLATE])
    self.app.layout = self.__layout_dashboard
    self.app.callback(dash.dependencies.Output("dashboard", "children"),
                      dash.dependencies.Input("update-dashboard", "n_intervals"),
                      running = [dash.dependencies.Output("update-dashboard", "disabled"), True, False])(self.__update_dashboard)
    self.app.run_server(debug = False, use_reloader = False, jupyter_mode = "external")
    # Display splash image.
    try:
      splash, meta = load_image(os.path.join(__packagepath__, "images", "splash.png"), verbose = False)
    except:
      pass
    else:
      self.show({"Welcome": splash})

  def __layout_dashboard(self):
    """Lay out dashboard."""
    dashboard = html.Div(self.content, id = "dashboard", style = {"width": params.maxwidth+params.lmargin+params.rmargin})
    interval = dcc.Interval(id = "update-dashboard", interval = self.interval, n_intervals = 0)
    return html.Div([dashboard, interval])

  def __update_dashboard(self, n):
    """Callback for dashboard updates.

    To do: Prevent re-entrance.

    Args:
      n: The number of calls since the start of the application.
    """
    refresh = self.refresh
    if not refresh: return dash.no_update
    with self.updatelock: # Lock on update.
      self.refresh = False
      return self.content

  def show(self, images, histograms = False, statistics = False, sampling = -1, hoverdata = False, trans = None):
    """Show image(s) on the dashboard.

    Args:
      images: The image(s) as a single/tuple/list/dictionary of Image object(s) or numpy.ndarray.
        Each image is displayed in a separate tab. The tabs are labelled according to the keys for
        a dictionary. Otherwise, the tabs are labelled "Image" &"Reference" if there are one or two images,
        and "Image #1", "Image #2"... if there are more.
        The images must all be Image objects if histograms or statistics is True.
      histograms (optional): If True or a string, show the histograms of the image(s). The string lists the
        channels of the histograms (e.g. "RGBL" for red, green, blue, luma). Default is False.
      statistics (optional): If True or a string, show the statistics of the image(s). The string lists the
        channels of the statistics (e.g. "RGBL" for red, green, blue, luma). Default is False.
      sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
        Only images[:, ::sampling, ::sampling] are shown, to speed up display.
      hoverdata (bool, optional): If True, show the image data on hover (default False).
        Warning: Setting hoverdata = True can slow down display a lot !
      trans (optional): A container with an histogram transformation (see Image.apply_channels), plotted on
        top of the histograms of the "Reference" tab (default None).
    """
    self.refresh = False
    if isinstance(images, (tuple, list)):
      nimages = len(images)
      if nimages == 1:
        imgdict = {"Image": images[0]}
      elif nimages == 2:
        imgdict = {"Image": images[0], "Reference": images[1]}
      else:
        n = 0
        imgdict = {}
        for image in images:
          n += 1
          imgdict[f"Image #{n}"] = image
    elif isinstance(images, dict):
      imgdict = images
    else:
      imgdict = {"Image": images}
    tabs = []
    for key, image in imgdict.items():
      tab = []
      tab.append(dcc.Graph(figure = _figure_image_(image, sampling = sampling, width = params.maxwidth, hoverdata = hoverdata, template = "slate")))
      if histograms is not False:
        if histograms is True: histograms = ""
        figure = _figure_histograms_(image, channels = histograms, log = True, width = params.maxwidth,
                                     trans = trans if key == "Reference" else None, template = "slate")
        if figure is not None: tab.append(dcc.Graph(figure = figure))
      if statistics is not False:
        if statistics is True: statistics = ""
        table = _table_statistics_(image, channels = statistics)
        if table is not None: tab.append(table)
      tabs.append(dbc.Tab(tab, label = key))
    with self.updatelock: # Lock on update.
      self.content = [dbc.Tabs(tabs, active_tab = "tab-0")]
      self.refresh = True

  def show_t(self, image, channels = "RGBL", sampling = -1, hoverdata = False):
    """Show the input and output images of an histogram transformation on the dashboard.

    Displays the input image, histograms, statistics, and transformation curve in tab "Reference",
    and the output image, histograms, and statistics in tab "Image".

    Args:
      image (Image): The output image (must embed a transformation image.trans - see Image.apply_channels).
      channels (str, optional): The channels of the histograms and statistics (default "RGBL" for red,
        green, blue, luma). The channels of the transformation are added if needed.
      sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
        Only image[:, ::sampling, ::sampling] is shown, to speed up display.
      hoverdata (bool, optional): If True, show the image data on hover (default False).
        Warning: Setting hoverdata = True can slow down display a lot !
    """
    if not issubclass(type(image), Image): print("The transformations can only be displayed for Image objects.")
    trans = getattr(image, "trans", None)
    if trans is None:
      print("There is no transformation embedded in the input image.")
      return
    reference = trans.input
    if trans.type == "hist":
      for c in trans.channels:
        if c in "RGBVSL" and not c in channels:
          channels += c
    self.show({"Image": image, "Reference": reference}, histograms = channels, statistics = channels,
              sampling = sampling, hoverdata = hoverdata, trans = trans)

  def carousel(self, images, sampling = -1, interval = 2000):
    """Show a carousel of images on the dashboard.

    Args:
      images: The images as a tuple/list/dictionary of Image object(s) or numpy.ndarray.
        The images are captioned according to the keys for a dictionary. Otherwise, the images are captioned
        "Image" and "Reference" if there are two images, and "Image #1", "Image #2"... if there are more.
      sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
        Only images[:, ::sampling, ::sampling] are shown, to speed up display.
      interval (int, optional): The interval (ms) between image switches in the carousel (default 2000).
    """
    self.refresh = False
    if isinstance(images, (tuple, list)):
      nimages = len(images)
      if nimages == 1:
        imgdict = {"Image": images[0]}
      elif nimages == 2:
        imgdict = {"Image": images[0], "Reference": images[1]}
      else:
        n = 0
        imgdict = {}
        for image in images:
          n += 1
          imgdict[f"Image #{n}"] = image
    elif isinstance(images, dict):
      imgdict = images
    else:
      imgdict = {"Image": images}
    n = 0
    items = []
    for key, image in imgdict.items():
      n += 1
      items.append(dict(key = f"{n}", src = prepare_images_as_b64strings(image, sampling = sampling), header = key))
    widget = dbc.Carousel(items = items, controls = True, indicators = True, ride = "carousel", interval = interval, className = "carousel-fade",
             style = {"width": f"{params.maxwidth}px", "margin": f"{params.tmargin}px {params.rmargin}px {params.bmargin}px {params.lmargin}px"})
    with self.updatelock: # Lock on update.
      self.content = [dbc.Tabs([dbc.Tab([widget], label = "Carousel")], active_tab = "tab-0")]
      self.refresh = True

  def slide(self, image1, image2, label1 = "Image", label2 = "Reference", sampling = -1):
    """Compare two images with a "before/after" slider on the dashboard.

    Args:
      image1: The first image as an Image object or numpy.ndarray.
      image2: The second image as an Image object or numpy.ndarray.
      label1 (str, optional): The label of the first image (default "Image").
      label2 (str, optional): The label of the second image (default "Reference").
      sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
        Only image1[:, ::sampling, ::sampling] and image2[:, ::sampling, ::sampling] are shown, to speed up display.
    """
    self.refresh = False
    img1, img2 = prepare_images_as_b64strings(image1, image2, sampling = sampling)
    left   = html.Div([label2],
                      style = {"width": f"{params.lmargin}px", "margin": "0px", "float": "left",
                               "padding": "4px", "writing-mode": "vertical-rl"})
    center = html.Div([dxt.BeforeAfter(before = dict(src = img1), after = dict(src = img2), width = f"{params.maxwidth}")],
                      style = {"width": f"{params.maxwidth}px", "margin": "0px", "float": "left"})
    right  = html.Div([label1],
                      style = {"margin": "0px", "float": "left",
                               "padding": "4px", "writing-mode": "vertical-rl"})
    widget = html.Div([left, center, right],
                      style = {"width": f"{params.maxwidth+params.lmargin+params.rmargin}px",
                               "margin": f"{params.tmargin}px 0px {params.bmargin}px 0px"})
    with self.updatelock: # Lock on update.
      self.content = [dbc.Tabs([dbc.Tab([widget], label = "Compare images")], active_tab = "tab-0")]
      self.refresh = True

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
  exclude01 = False
  for channel in stats.values():
    exclude01 = exclude01 or channel.exclude01
    deco = "\u207d\u2071\u207e" if channel.exclude01 else ""
    if channel.percentiles is not None:
      percentiles = [f"{channel.percentiles[0]:.5f}{deco}", f"{channel.percentiles[1]:.5f}{deco}", f"{channel.percentiles[2]:.5f}{deco}"]
    else:
      percentiles = 3*[f"None{deco}"]
    rows.append(html.Tr([html.Td(channel.name, style = {"text-align": "left"}), html.Td(f"{channel.minimum:.5f}"),
                         html.Td(percentiles[0]), html.Td(percentiles[1]),
                         html.Td(percentiles[2]), html.Td(f"{channel.maximum:.5f}"),
                         html.Td(f"{channel.zerocount} ({100.*channel.zerocount/channel.npixels:.2f}%)"),
                         html.Td(f"{channel.outcount} ({100.*channel.outcount/channel.npixels:.2f}%)")]))
  body = [html.Tbody(rows)]
  table = [dbc.Table(header+body, size = "sm", bordered = True, striped = True, style = {"text-align": "right", "width": f"{params.maxwidth}px"})]
  if exclude01: table.append("\u207d\u2071\u207e Does not include pixels <= 0 or >= 1.")
  return html.Div(table, style = {"margin": f"32px {params.rmargin}px 32px {params.lmargin}px"})
