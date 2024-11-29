# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Dash backend for Jupyter-lab interface."""

# TODO:
#  - Shadows/Highlights/Differences.
#  - Update tabs only if necessary.

import os
import threading
import numpy as np
import dash
from dash import Dash, dcc, html, ctx
import dash_bootstrap_templates as dbt
import dash_bootstrap_components as dbc
import dash_extensions as dxt

from . import params
from .utils import get_image_size, prepare_images, prepare_images_as_b64strings
from .backend_plotly import _figure_prepared_image_, _figure_histograms_

from ..equimage import Image, load_image, get_RGB_luma

class Dashboard():
  """Dashboad class."""

  def __init__(self, interval = 333):
    """Initialize dashboard.

    This dashboard uses Dash to displays image, histograms, statistics, etc... in a
    separate browser tab or window.
    It fetches updates from the Dash server at given intervals.

    Args:
      interval (int, optional): The time interval (ms, default 333) between dashboard updates.
    """
    from .. import __packagepath__
    from dash.dependencies import Input, Output, State, ALL, MATCH
    # Initialize object data.
    self.content = []
    self.images = None
    self.nupdates = 0
    self.refresh = False
    self.synczoom = False
    self.interval = interval
    self.updatelock = threading.Lock()
    # Set-up Dash app.
    dbt.load_figure_template("slate")
    self.app = Dash(title = "eQuimageLab dashboard", update_title = None, external_stylesheets = [dbc.themes.SLATE])
    self.app.layout = self.__layout_dashboard
    # Register callbacks:
    #   - Dashboard update:
    self.app.callback(Output("dashboard", "children"), Output("updateid", "data"), Input("updateinterval", "n_intervals"),
                      running = [Output("updateinterval", "disabled"), True, False])(self.__update_dashboard)
    #   - Image filters:
    self.app.callback(Output({"type": "filters", "index": MATCH}, "value"), Output({"type": "selectedfilters", "index": MATCH}, "data"),
                      Input({"type": "filters", "index": MATCH}, "value"), Input({"type": "selectedfilters", "index": MATCH}, "data"),
                      Input("updateid", "data"), prevent_initial_call = True)(self.__filter_image)
    #   - Image click:
    self.app.callback(Output({"type": "datadiv", "index": MATCH}, "children"), Input({"type": "image", "index": MATCH}, "clickData"),
                      Input("updateid", "data"), prevent_initial_call = True)(self.__click_image)
    #   - Zoom synchronization:
    self.app.callback(Output({"type": "image", "index": ALL}, "relayoutData"), Output({"type": "image", "index": ALL}, "figure"),
                      Input({"type": "image", "index": ALL}, "relayoutData"), State({"type": "image", "index": ALL}, "figure"),
                      prevent_initial_call = True)(self.__sync_zoom)
    # Launch Dash server.
    self.app.run_server(debug = False, use_reloader = False, jupyter_mode = "external")
    # Display splash image.
    try:
      splash, meta = load_image(os.path.join(__packagepath__, "images", "splash.png"), verbose = False)
    except:
      pass
    else:
      self.show({"Welcome": splash}, modifiers = False, hover = False, click = False)

  def __layout_dashboard(self):
    """Lay out dashboard."""
    dashboard = html.Div(self.content, id = "dashboard", style = {"display": "inline-block"})
    updateid = dcc.Store(data = self.nupdates, id = "updateid")
    interval = dcc.Interval(interval = self.interval, n_intervals = 0, id = "updateinterval")
    return html.Div([dashboard, updateid, interval], style = {"margin": "8px"})

  def __update_dashboard(self, n):
    """Callback for dashboard updates.

    Args:
      n: Number of calls since the start of the application.

    Returns:
      The updated dashboard and the updateid.
    """
    refresh = self.refresh
    if not refresh: return dash.no_update, dash.no_update
    with self.updatelock: # Lock on update.
      self.refresh = False
      return self.content, self.nupdates

  def __filter_image(self, values, selected, updateid):
    """Callback for image filters.

    Args:
      values (list):
      selected (list):
      updateid (integer): The unique ID of the dashboard update.

    Returns:
    """
    print(values, selected)
    trigger = ctx.triggered_id # Get the figure that triggered the callback.
    if not trigger: return selected, selected
    if self.images is None or updateid != self.nupdates: return [], [] # The dashboard is out of sync.
    n = trigger["index"] # Image index.
    return values, selected

  def __click_image(self, click, updateid):
    """Callback for image click.

    Prints image coordinates and data at click point.

    Args:
      click (dict): The click event dictionary.
      updateid (integer): The unique ID of the dashboard update.

    Returns:
      The content of the "datadiv" div element with the image coordinates and data at click point.
    """
    trigger = ctx.triggered_id # Get the figure that triggered the callback.
    if not trigger: return []
    if self.images is None or updateid != self.nupdates: return [], [] # The dashboard is out of sync.
    n = trigger["index"] # Image index.
    x = click["points"][0]["x"]
    y = click["points"][0]["y"]
    levels = self.images[n][:, y, x]
    if levels.size == 1:
      return [f"Data at ({x}, {y}): L = {levels[0]:.5f}"]
    else:
      rgbluma = get_RGB_luma()
      luma = rgbluma[0]*levels[0]+rgbluma[1]*levels[1]+rgbluma[2]*levels[2]
      return [f"Data at ({x}, {y}): R = {levels[0]:.5f}, G = {levels[1]:.5f}, B = {levels[2]:.5f}, L = {luma:.5f}"]

  def __sync_zoom(self, fig_relayouts, fig_states):
    """Callback for zoom synchronization.

    Args:
      fig_relayouts: Input figure relayouts.
      fig_states: Input figure states.

    Returns:
      Output figure relayouts and figure states.
    """
    if not self.synczoom: return fig_relayouts, fig_states
    trigger = ctx.triggered_id # Get the figure that triggered the callback.
    if not trigger: return fig_relayouts, fig_states
    n = trigger["index"] # Image index.
    relayout = fig_relayouts[n]
    xauto = relayout.get("xaxis.autorange", False)
    if not xauto:
      x1 = relayout["xaxis.range[0]"] ; x2 = relayout["xaxis.range[1]"]
    yauto = relayout.get("yaxis.autorange", False)
    if not yauto:
      y1 = relayout["yaxis.range[0]"] ; y2 = relayout["yaxis.range[1]"]
    for fig_state in fig_states:
      fig_state["layout"]["xaxis"]["autorange"] = xauto
      if not xauto:
        fig_state["layout"]["xaxis"]["range"] = [x1, x2]
      fig_state["layout"]["yaxis"]["autorange"] = yauto
      if not yauto:
        fig_state["layout"]["yaxis"]["range"] = [y1, y2]
    return [relayout]*len(fig_relayouts), fig_states

  def show(self, images, histograms = False, statistics = False, sampling = -1, modifiers = True, hover = False, click = True, synczoom = True, trans = None):
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
      modifiers (bool, optional): If True (default), add image modifiers menu (R, G, B, L channel
        filters, shadowed/highlighted pixels, images differences).
      hover (bool, optional): If True, show the images data on hover (default False).
        Warning: Setting hover = True can slow down display a lot !
      click (bool, optional): If True, show the images data on click (default True).
      synczoom (bool, optional): If True (default), synchronize zooms over the images (default False).
        Zooms can be synchronized only if all images have the same size.
      trans (optional): A container with an histogram transformation (see Image.apply_channels), plotted on
        top of the histograms of the "Reference" tab (default None).
    """
    self.refresh = False # Stop refreshing dashboard.
    # Build the dictionary of images.
    if isinstance(images, (tuple, list)):
      nimages = len(images)
      if nimages == 1:
        keys = ["Image"]
      elif nimages == 2:
        keys = ["Image", "Reference"]
      else:
        keys = [f"Image #{n}" for n in range(nimages)]
    elif isinstance(images, dict):
      nimages = len(images)
      keys = list(images.keys())
      images = list(images.values())
    else:
      nimages = 1
      keys = ["Image"]
      images = [images]
    # Look for a reference image.
    try:
      reference = keys.index("Reference")
    except:
      reference = None
    # Prepare images.
    pimages = prepare_images(images, sampling = sampling)
    # Check if zooms can be synchronized.
    synczoom = synczoom and nimages > 1
    if synczoom:
      size = pimages[0].shape[1:3]
      for image in pimages[1:]:
        synczoom = (image.shape[1:3] == size)
        if not synczoom: break
    # Set-up tabs.
    tabs = []
    for n in range(nimages):
      tab = []
      figure = _figure_prepared_image_(pimages[n], width = params.maxwidth, hover = hover, template = "slate")
      if click: figure.update_layout(clickmode = "event+select")
      tab.append(dcc.Graph(figure = figure, id = {"type": "image", "index": n}))
      if modifiers:
        options = []
        values = []
        if pimages[n].shape[0] > 1: # Color image.
          options.extend(["R", "G", "B", "L"])
          values.extend(["R", "G", "B"])
        options.extend(["Shadowed", "Highlighted"])
        if reference: options.extend(["Differences"])
        checklist = dcc.Checklist(options = options, value = values, id = {"type": "filters", "index": n},
                                  inline = True, labelStyle = {"margin-right": "16px"})
        selected = dcc.Store(data = values, id = {"type": "selectedfilters", "index": n})
        tab.append(html.Div([html.Div(["Filters:"], style = {"display": "inline-block", "margin-right": "16px"}),
                             html.Div([checklist], style = {"display": "inline-block"}), selected],
                             style = {"width": f"{params.maxwidth}px", "margin": f"0px {params.rmargin}px 0px {params.lmargin}px"}))
      if click:
        tab.append(html.Div([], id = {"type": "datadiv", "index": n},
                   style = {"width": f"{params.maxwidth}px", "margin": f"0px {params.rmargin}px 0px {params.lmargin}px"}))
      if histograms is not False:
        if histograms is True: histograms = ""
        figure = _figure_histograms_(images[n], channels = histograms, log = True, width = params.maxwidth,
                                     trans = trans if keys[n] == "Reference" else None, template = "slate")
        if figure is not None: tab.append(dcc.Graph(figure = figure))
      if statistics is not False:
        if statistics is True: statistics = ""
        table = _table_statistics_(images[n], channels = statistics)
        if table is not None: tab.append(table)
      tabs.append(dbc.Tab(tab, label = keys[n], style = {"border": f"solid {params.border}px black"}))
    # Update dashboard.
    with self.updatelock: # Lock on update.
      # BEWARE TO SIDE EFFECTS: SELF.IMAGES MAY REFERENCE THE ORIGINAL IMAGES.
      self.nupdates += 1
      self.synczoom = synczoom
      self.reference = reference
      self.images = pimages if click else None # No need to register images for the callbacks if click is False.
      self.content = [dbc.Tabs(tabs, active_tab = "tab-0")]
      self.refresh = True

  def show_t(self, image, channels = "RGBL", sampling = -1, hover = False, synczoom = True):
    """Show the input and output images of an histogram transformation on the dashboard.

    Displays the input image, histograms, statistics, and transformation curve in tab "Reference",
    and the output image, histograms, and statistics in tab "Image".

    Args:
      image (Image): The output image (must embed a transformation image.trans - see Image.apply_channels).
      channels (str, optional): The channels of the histograms and statistics (default "RGBL" for red,
        green, blue, luma). The channels of the transformation are added if needed.
      sampling (int, optional): Downsampling rate (defaults to params.sampling if negative).
        Only image[:, ::sampling, ::sampling] is shown, to speed up display.
      hover (bool, optional): If True, show the images data on hover (default False).
        Warning: Setting hover = True can slow down display a lot !
      synczoom (bool, optional): If True (default), synchronize zooms over the images (default False).
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
              sampling = sampling, hover = hover, synczoom = synczoom, trans = trans)

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
    self.refresh = False # Stop refreshing dashboard.
    # Build the dictionary of images.
    if isinstance(images, (tuple, list)):
      nimages = len(images)
      if nimages == 1:
        keys = ["Image"]
      elif nimages == 2:
        keys = ["Image", "Reference"]
      else:
        keys = [f"Image #{n}" for n in range(nimages)]
    elif isinstance(images, dict):
      nimages = len(images)
      keys = list(images.keys())
      images = list(images.values())
    else:
      nimages = 1
      keys = ["Image"]
      images = [images]
    # Set-up carousel.
    items = [dict(key = f"{n}", src = prepare_images_as_b64strings(images[n], sampling = sampling), header = keys[n]) for n in range(nimages)]
    widget = dbc.Carousel(items = items, controls = True, indicators = True, ride = "carousel", interval = interval, className = "carousel-fade",
             style = {"width": f"{params.maxwidth}px", "margin": f"{params.tmargin}px {params.rmargin}px {params.bmargin}px {params.lmargin}px"})
    # Update dashboard.
    with self.updatelock: # Lock on update.
      self.nupdates += 1
      self.images = None # No need to register images for the callbacks.
      self.content = [dbc.Tabs([dbc.Tab([widget], label = "Carousel", style = {"border": f"solid {params.border}px black"})], active_tab = "tab-0")]
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
    self.refresh = False # Stop refreshing dashboard.
    # Set-up before/after widget.
    image1, image2 = prepare_images_as_b64strings((image1, image2), sampling = sampling)
    left   = html.Div([label2],
                      style = {"display": "inline-block", "width": f"{params.lmargin}px",
                               "margin": "0px", "padding": "4px", "vertical-align": "middle", "writing-mode": "vertical-rl"})
    center = html.Div([dxt.BeforeAfter(before = dict(src = image1), after = dict(src = image2), width = params.maxwidth)],
                      style = {"display": "inline-block", "width": f"{params.maxwidth}px",
                               "margin": "0px", "vertical-align": "middle"})
    right  = html.Div([label1],
                      style = {"display": "inline-block",
                               "margin": "0px", "padding": "4px", "vertical-align": "middle", "writing-mode": "vertical-rl"})
    widget = html.Div([left, center, right],
                      style = {"display": "inline-block", "width": f"{params.maxwidth+params.lmargin+params.rmargin}px",
                               "margin": f"{params.tmargin}px 0px {params.bmargin}px 0px"})
    # Update dashboard.
    with self.updatelock: # Lock on update.
      self.nupdates += 1
      self.images = None # No need to register images for the callbacks.
      self.content = [dbc.Tabs([dbc.Tab([widget], label = "Compare images", style = {"border": f"solid {params.border}px black"})], active_tab = "tab-0")]
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
  table = [dbc.Table(header+body, size = "sm", bordered = True, striped = True, style = {"width": f"{params.maxwidth}px", "text-align": "right"})]
  if exclude01: table.append("\u207d\u2071\u207e Does not include pixels <= 0 or >= 1.")
  return html.Div(table, style = {"margin": f"32px {params.rmargin}px 32px {params.lmargin}px"})
