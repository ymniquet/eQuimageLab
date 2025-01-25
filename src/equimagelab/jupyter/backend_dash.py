# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Dash backend for Jupyter Lab interface."""

# TODO:
#  - Update tabs only if necessary.

import os
import threading
import numpy as np
import dash
from dash import Dash, dcc, html
import dash_bootstrap_templates as dbt
import dash_bootstrap_components as dbc
import dash_extensions as dxt

from . import params
from .utils import prepare_images, prepare_images_as_b64strings, shadowed, highlighted, differences
from .backend_plotly import _figure_prepared_image_, _figure_histograms_

from ..equimage import Image, load_image, get_RGB_luma
from ..equimage.image_stats import parse_channels

class Dashboard():
  """Dashboad class."""

  ################
  # Constructor. #
  ################

  def __init__(self, port = 8050, interval = 500, debug = False):
    """Initialize dashboard.

    This dashboard uses Dash to display images, histograms, statistics, etc...
    in a separate browser tab or window.
    It fetches updates from the Dash server at given intervals.

    Args:
      port (int, optional): The port bound to the Dash server (default 8050).
      interval (int, optional): The time interval (ms) between dashboard updates (default 500).
      debug (bool, optional): If True, run Dash in debug mode (default False).
    """
    from .. import __packagepath__
    from dash.dependencies import Input, Output, State, ALL, MATCH
    # Initialize object data.
    self.content = []
    self.nupdates = 0
    self.layout = None
    self.images = None
    self.refresh = False
    self.synczoom = False
    self.interval = interval
    self.updatelock = threading.Lock()
    # Set-up Dash app.
    dbt.load_figure_template("slate")
    self.app = Dash(name = __name__, title = "eQuimageLab dashboard", update_title = None,
                    external_stylesheets = [dbc.themes.SLATE], suppress_callback_exceptions = True)
    self.app.layout = self.__layout_dashboard
    # Register callbacks:
    #   - Dashboard update:
    self.app.callback(Output("dashboard", "children"), Output("updateid", "data"),
                      Input("updateinterval", "n_intervals"),
                      running = [Output("updateinterval", "disabled"), True, False], prevent_initial_call = True)(self.__update_dashboard)
    #   - Image click:
    self.app.callback(Output({"type": "datadiv", "index": MATCH}, "children", allow_duplicate = True),
                      Input({"type": "image", "index": MATCH}, "clickData"),
                      State("updateid", "data"),
                      prevent_initial_call = True)(self.__click_image)
    #   - Image filters:
    self.app.callback(Output({"type": "filters", "index": MATCH}, "value"), Output({"type": "selectedfilters", "index": MATCH}, "data"),
                      Output({"type": "image", "index": MATCH}, "figure", allow_duplicate = True),
                      Input({"type": "filters", "index": MATCH}, "value"),
                      State({"type": "selectedfilters", "index": MATCH}, "data"), State("updateid", "data"),
                      prevent_initial_call = True)(self.__filter_image)
    #   - Local histograms:
    self.app.callback(Output({"type": "offcanvas", "index": MATCH}, "is_open"), Output({"type": "offcanvas", "index": MATCH}, "children"),
                      Input({"type": "histogramsbutton", "index": MATCH}, "n_clicks"),
                      State({"type": "offcanvas", "index": MATCH}, "is_open"), State({"type": "image", "index": MATCH}, "figure"),
                      State("updateid", "data"),
                      prevent_initial_call = True)(self.__local_histograms)
    #   - Images zoom synchronization:
    self.app.callback(Output({"type": "image", "index": ALL}, "relayoutData"), Output({"type": "image", "index": ALL}, "figure"),
                      Input({"type": "image", "index": ALL}, "relayoutData"),
                      prevent_initial_call = True)(self.__sync_image_zoom)
    #   - Histograms zoom tracking:
    self.app.callback(Input({"type": "histograms", "index": ALL}, "relayoutData"), prevent_initial_call = True)(self.__track_histograms_zoom)
    #   - Tab switch.
    self.app.callback(Input("image-tabs", "active_tab"), prevent_initial_call = True)(self.__switch_tab)
    # Launch Dash server.
    self.app.run_server(port = port, debug = debug, use_reloader = False, jupyter_mode = "external")
    # Display splash image.
    try:
      splash, meta = load_image(os.path.join(__packagepath__, "images", "splash.png"), verbose = False)
    except:
      pass
    else:
      self.show({"Welcome": splash}, filters = False, click = False, synczoom = False)

  ##############
  # Callbacks. #
  ##############

  def __layout_dashboard(self):
    """Lay out dashboard."""
    with self.updatelock: # Lock while updating layout.
      dashboard = html.Div(self.content, id = "dashboard", className = "dashboard-inner")
      updateid = dcc.Store(data = self.nupdates, id = "updateid")
      interval = dcc.Interval(interval = self.interval, n_intervals = 0, id = "updateinterval")
      return html.Div([dashboard, updateid, interval], className = "dashboard-outer")

  def __update_dashboard(self, n_intervals):
    """Callback for dashboard updates.

    Args:
      n_intervals: The number of update intervals elapsed since the start of the application.

    Returns:
      The updated dashboard and the unique ID of the update.
    """
    if not self.refresh or n_intervals <= 0: return dash.no_update, dash.no_update
    with self.updatelock: # Lock on callback.
      self.refresh = False
      return self.content, self.nupdates

  def __click_image(self, click, updateid):
    """Callback for image click.

    Prints image coordinates and data at click point.

    Args:
      click (dict): The click event dictionary.
      updateid (integer): The unique ID of the displayed dashboard update.

    Returns:
      The content of the "datadiv" div element with the image coordinates and data at click point.
    """
    trigger = dash.ctx.triggered_id # Get the component that triggered the callback.
    if not trigger: return []
    with self.updatelock: # Lock on callback.
      if self.images is None or updateid != self.nupdates: return [] # The dashboard is out of sync.
      n = trigger["index"] # Image index.
      x = click["points"][0]["x"]
      y = click["points"][0]["y"]
      data = self.images[n][y, x]
      if data.size > 1:
        rgbluma = get_RGB_luma()
        luma = rgbluma[0]*data[0]+rgbluma[1]*data[1]+rgbluma[2]*data[2]
        return [f"Data at (x = {x}, y = {y}): ", html.Span(f"R = {data[0]:.5f}", className = "red"), ", ",
                                                 html.Span(f"G = {data[1]:.5f}", className = "green"), ", ",
                                                 html.Span(f"B = {data[2]:.5f}", className = "blue"), ", ",
                                                 html.Span(f"L = {luma:.5f}", className = "luma"), "."]
      else:
        return [f"Data at (x = {x}, y = {y}): ", html.Span(f"L = {data:.5f}", className = "luma"), "."]

  def __filter_image(self, current, previous, updateid):
    """Callback for image filters.

    Apply selected filters (R, G, B, L channel filters, shadowed/highlighted pixels, images differences) to
    the current image.

    Args:
      current (list): The currently selected filters.
      previous (list): The previously selected filters.
      updateid (integer): The unique ID of the displayed dashboard update.

    Returns:
      The curated filters (twice, as currently selected and new previous), and the filtered figure.
    """

    def filter_channels(image, channels):
      """Apply channel filters to the input image."""
      if image.ndim > 2:
        if "L" in channels: # Return luma.
          rgbluma = get_RGB_luma()
          return rgbluma[0]*image[:, : , 0]+rgbluma[1]*image[:, : , 1]+rgbluma[2]*image[:, : , 2]
        else: # Filter out R, G, B channels.
          output = image.copy()
          if "R" not in channels: output[:, :, 0] = 0.
          if "G" not in channels: output[:, :, 1] = 0.
          if "B" not in channels: output[:, :, 2] = 0.
        return output
      else: # Return luma.
        return image

    trigger = dash.ctx.triggered_id # Get the component that triggered the callback.
    if not trigger: return previous, previous, dash.no_update
    with self.updatelock: # Lock on callback.
      if self.images is None or updateid != self.nupdates: return [], [], dash.no_update # The dashboard is out of sync.
      # Update filters list.
      current = set(current)
      previous = set(previous)
      toggled = current^previous
      for t in toggled: # There shall be only one checkbox toggled, actually.
        if t == "L":
          if "L" in current:
            current.difference_update({"R", "G", "B"})
          else:
            current.update({"R", "G", "B"})
        elif t in ["R", "G", "B"]:
          if not current & {"R", "G", "B"}:
            current.update("L")
          else:
            current.difference_update({"L"})
        elif t == "S":
          current.difference_update({"H", "D"})
        elif t == "H":
          current.difference_update({"S", "D"})
        elif t == "D":
          current.difference_update({"S", "H"})
        else:
          raise ValueError(f"Error, unknown filter {t}.")
      # Apply selected filters to the image.
      n = trigger["index"] # Image index.
      image = filter_channels(self.images[n], current)
      if current & {"S", "H", "D"}:
        reference = filter_channels(self.images[self.reference], current) if self.reference is not None else None
        if "S" in current:
          image = shadowed(image, reference)
        elif "H" in current:
          image = highlighted(image, reference)
        else:
          image = differences(image, reference)
      # Return filtered image as a patch.
      figure_patch = dash.Patch()
      figure_patch["data"][0]["source"] = prepare_images_as_b64strings(image, sampling = 1)
      current = list(current)
      return current, current, figure_patch

  def __local_histograms(self, n_clicks, is_open, figure, updateid):
    """Callback for local histograms.

    Shows the histograms of the displayed area of the current image.

    Args:
      n_clicks (list): The number of clicks on the histograms button.
      is_open (bool): The status of the off-canvas showing the local histograms.
      figure (dict): The figure associated with the histograms button.
      updateid (integer): The unique ID of the displayed dashboard update.

    Returns:
      The status and content of the off-canvas showing the local histograms.
    """
    if is_open: return False, [] # Close the off-canvas.
    if n_clicks <= 0: return False, []
    trigger = dash.ctx.triggered_id # Get the component that triggered the callback.
    if not trigger: return False, []
    with self.updatelock: # Lock on callback.
      if self.images is None or updateid != self.nupdates: return False, [] # The dashboard is out of sync.
      # Get x and y axes range.
      xmin, xmax = figure["layout"]["xaxis"]["range"]
      xmin = np.ceil(xmin) ; xmax = np.floor(xmax)
      ymax, ymin = figure["layout"]["yaxis"]["range"]
      ymin = np.ceil(ymin) ; ymax = np.floor(ymax)
      if (xmax-xmin)*(ymax-ymin) < 256: return False, [] # Area too small for histograms.
      # Compute local histograms using eQuimage.
      n = trigger["index"] # Image index.
      image = Image(self.images[n], channels = -1).crop(xmin, xmax, ymin, ymax)
      figure = _figure_histograms_(image, channels = "RGBL", log = True, width = params.maxwidth, template = "slate")
      content = [dcc.Graph(figure = figure)]
      return True, content

  def __sync_image_zoom(self, relayouts):
    """Callback for images zoom synchronization.

    Args:
      relayouts (list): Input relayouts of all images.

    Returns:
      Output relayouts and figure patches for all images.
    """
    nimages = len(relayouts)
    if not self.synczoom: return [dash.no_update]*nimages, [dash.no_update]*nimages
    trigger = dash.ctx.triggered_id # Get the component that triggered the callback.
    if not trigger: return [dash.no_update]*nimages, [dash.no_update]*nimages
    with self.updatelock: # Lock on callback.
      n = trigger["index"] # Image index.
      relayout = relayouts[n]
      xauto = relayout.get("xaxis.autorange", False)
      if not xauto:
        xmin = relayout.get("xaxis.range[0]", None)
        xmax = relayout.get("xaxis.range[1]", None)
        if xmin is None or xmax is None:
          return [dash.no_update]*nimages, [dash.no_update]*nimages # Unexpected relayout structure; Discard event.
      yauto = relayout.get("yaxis.autorange", False)
      if not yauto:
        ymin = relayout.get("yaxis.range[0]", None)
        ymax = relayout.get("yaxis.range[1]", None)
        if ymin is None or ymax is None:
          return [dash.no_update]*nimages, [dash.no_update]*nimages # Unexpected relayout structure; Discard event.
      figure_patch = dash.Patch()
      figure_patch["layout"]["xaxis"]["autorange"] = xauto
      figure_patch["layout"]["yaxis"]["autorange"] = yauto
      if xauto:
        self.xrange = None
      else:
        self.xrange = [xmin, xmax]
        figure_patch["layout"]["xaxis"]["range"] = self.xrange
      if yauto:
        self.yrange = None
      else:
        self.yrange = [ymin, ymax]
        figure_patch["layout"]["yaxis"]["range"] = self.yrange
      return [relayout]*nimages, [figure_patch]*nimages

  def __track_histograms_zoom(self, relayouts):
    """Callback for histograms zoom tracking.

    Tracks x axis changes on histograms.

    Args:
      relayouts (dict): Input relayouts of all histograms.
    """
    if not self.synczoom: return # This comes along with image zoom synchronization.
    trigger = dash.ctx.triggered_id # Get the component that triggered the callback.
    if not trigger: return
    with self.updatelock: # Lock on callback.
      n = trigger["index"] # Image index.
      relayout = relayouts[n]
      xauto = relayout.get("xaxis.autorange", False)
      if xauto:
        self.hrange[n] = None
      else:
        xmin = relayout.get("xaxis.range[0]", None)
        xmax = relayout.get("xaxis.range[1]", None)
        if xmin is None or xmax is None: return # Unexpected relayout structure; Discard event.
        self.hrange[n] = [xmin, xmax]

  def __switch_tab(self, tab):
    """Callback for tab switches.

    This callback just stores the current tab name.

    Args:
      tab (string): The name of the current tab.
    """
    with self.updatelock: # Lock on callback.
      self.activetab = tab

  ############
  # Layouts. #
  ############

  ### Tabs layout.

  def show(self, images, histograms = False, statistics = False, sampling = -1, filters = True, click = True, synczoom = True, trans = None):
    """Show image(s) on the dashboard.

    Args:
      images: The image(s) as a single/tuple/list/dictionary of equimage.Image object(s) or numpy.ndarray.
        Each image is displayed in a separate tab. The tabs are labelled according to the keys for
        a dictionary. Otherwise, the tabs are labelled "Image" & "Reference" if there are one or two images,
        and "Image #1", "Image #2"... if there are more.
        The images must all be equimage.Image objects if histograms or statistics is True.
      histograms (optional): If True or a string, show the histograms of the image(s). The string lists the
        channels of the histograms (e.g. "RGBL" for red, green, blue, luma). Default is False.
      statistics (optional): If True or a string, show the statistics of the image(s). The string lists the
        channels of the statistics (e.g. "RGBL" for red, green, blue, luma). Default is False.
      sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
        Only images[::sampling, ::sampling] are shown, to speed up display.
      filters (bool, optional): If True (default), add image filters menu (R, G, B, L channel filters,
        shadowed/highlighted pixels, images differences, local histograms).
      click (bool, optional): If True, show image data on click (default True).
      synczoom (bool, optional): If True (default), synchronize zooms over images.
        Zooms will be synchronized only if all images have the same size.
      trans (optional): A container with an histogram transformation (see equimage.Image.apply_channels),
        plotted on top of the histograms of the "Reference" tab (default None).
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
        keys = [f"Image #{n+1}" for n in range(nimages)]
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
    if synczoom:
      imagesize = pimages[0].shape[0:2]
      for image in pimages[1:]:
        synczoom = (image.shape[0:2] == imagesize)
        if not synczoom: break
    if not synczoom: imagesize = None
    # Try to preserve existing axes if already in tabs layout and synczoom is consistently True.
    preserveaxes = synczoom and self.layout == "tabs" and self.synczoom and self.imagesize == imagesize
    if preserveaxes:
      xrange = self.xrange # Image x & y ranges.
      yrange = self.yrange
    else:
      xrange = None
      yrange = None
    hrange = [None]*nimages # Histograms x ranges.
    # Set-up tabs.
    tabs = []
    for n in range(nimages):
      tab = []
      figure = _figure_prepared_image_(pimages[n], width = params.maxwidth, hover = False, template = "slate")
      if xrange is not None: figure.update_layout(xaxis_range = xrange)
      if yrange is not None: figure.update_layout(yaxis_range = yrange)
      if click: figure.update_layout(clickmode = "event+select")
      tab.append(dcc.Graph(figure = figure, id = {"type": "image", "index": n}))
      if filters:
        options = []
        values = []
        if pimages[n].ndim > 2: # Color image.
          options.extend([{"label": html.Span("R", className = "red lm1"), "value": "R"},
                          {"label": html.Span("G", className = "green lm1"), "value": "G"},
                          {"label": html.Span("B", className = "blue lm1"), "value": "B"},
                          {"label": html.Span("L", className = "luma lm1 rm4"), "value": "L"}])
          values.extend(["R", "G", "B"])
        options.extend([{"label": html.Span("Shadowed", className = "lm1"), "value": "S"},
                        {"label": html.Span("Highlighted", className = "lm1"), "value": "H"}])
        if reference is not None and pimages[n].shape == pimages[reference].shape:
          options.extend([{"label": html.Span("Differences", className = "lm1"), "value": "D"}])
        checklist = dcc.Checklist(options = options, value = values, id = {"type": "filters", "index": n},
                                  inline = True, labelClassName = "rm4")
        selected = dcc.Store(data = values, id = {"type": "selectedfilters", "index": n})
        button = dbc.Button("Local histograms", color = "primary", size = "sm", n_clicks = 0, id = {"type": "histogramsbutton", "index": n})
        offcanvas = dbc.Offcanvas([], title = "Histograms of the displayed area of the image:", placement = "top", close_button = True, keyboard = True,
                                  id = {"type": "offcanvas", "index": n}, style = {"height": "auto", "bottom": "initial"}, is_open = False)
        tab.append(html.Div([html.Div(["Filters:"], className = "rm4"), html.Div([checklist]),  html.Div([button], className = "flushright"),
                             selected, offcanvas], className = "flex center tm1 bm1",
                             style = {"width": f"{params.maxwidth}px", "margin-left": f"{params.lmargin}px", "margin-right": f"{params.rmargin}px"}))
      if click:
        tab.append(html.Div([], id = {"type": "datadiv", "index": n}, className = "tm1 bm1",
                   style = {"width": f"{params.maxwidth}px", "margin-left": f"{params.lmargin}px", "margin-right": f"{params.rmargin}px"}))
      if histograms is not False:
        if histograms is True: histograms = ""
        figure = _figure_histograms_(images[n], channels = histograms, log = True, width = params.maxwidth,
                                     trans = trans if keys[n] == "Reference" else None, template = "slate")
        if figure is not None:
          if preserveaxes and keys[n] in self.keys:
            hrange[n] = self.hrange[self.keys.index(keys[n])]
            if hrange[n] is not None: figure.update_layout(xaxis_range = hrange[n])
          tab.append(dcc.Graph(figure = figure, id = {"type": "histograms", "index": n}))
      if statistics is not False:
        if statistics is True: statistics = ""
        table = _table_statistics_(images[n], channels = statistics)
        if table is not None: tab.append(table)
      tabs.append(dbc.Tab(tab, label = keys[n], tab_id = keys[n], className = "tab"))
    # Set active tab. Keep current tab open if possible.
    activetab = self.activetab if self.layout == "tabs" and self.activetab in keys else keys[0]
    # Update dashboard.
    with self.updatelock: # Lock on update.
      # BEWARE TO SIDE EFFECTS: SELF.IMAGES MAY REFERENCE THE ORIGINAL IMAGES.
      self.nupdates += 1
      self.layout = "tabs"
      self.keys = keys
      self.xrange = xrange
      self.yrange = yrange
      self.hrange = hrange
      self.synczoom = synczoom
      self.imagesize = imagesize
      self.reference = reference
      self.activetab = activetab
      self.images = pimages if click or filters else None # No need to register images for the callbacks if click and filters are False.
      self.content = [dbc.Tabs(tabs, active_tab = activetab, id = "image-tabs")]
      self.refresh = True

  def show_t(self, image, channels = "RGBL", sampling = -1, filters = True, click = True, synczoom = True):
    """Show the input and output images of an histogram transformation on the dashboard.

    Displays the input image, histograms, statistics, and transformation curve in tab "Reference",
    and the output image, histograms, and statistics in tab "Image".

    Args:
      image (equimage.Image): The output image (must embed a transformation image.trans -
        see equimage.Image.apply_channels).
      channels (str, optional): The channels of the histograms and statistics (default "RGBL" for red,
        green, blue, luma). The channels of the transformation are added if needed.
      sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
        Only image[::sampling, ::sampling] is shown, to speed up display.
      filters (bool, optional): If True (default), add image filters menu (R, G, B, L channel filters,
        shadowed/highlighted pixels, images differences, local histograms).
      click (bool, optional): If True, show image data on click (default True).
      synczoom (bool, optional): If True (default), synchronize zooms over images.
    """
    if not issubclass(type(image), Image): print("The transformations can only be displayed for Image objects.")
    trans = getattr(image, "trans", None)
    if trans is None:
      print("There is no transformation embedded in the input image.")
      return
    reference = trans.input
    if trans.type == "hist":
      keys = parse_channels(channels)
      for key in parse_channels(trans.channels, errors = False):
        if not key in keys: channels += key
    self.show({"Image": image, "Reference": reference}, histograms = channels, statistics = channels,
              sampling = sampling, filters = filters, click = click, synczoom = synczoom, trans = trans)

  ### Carousel layout.

  def carousel(self, images, sampling = -1, interval = 2000):
    """Show a carousel of images on the dashboard.

    Args:
      images: The images as a tuple/list/dictionary of equimage.Image object(s) or numpy.ndarray.
        The images are labelled according to the keys for a dictionary. Otherwise, the images are labelled
        "Image" and "Reference" if there are two images, and "Image #1", "Image #2"... if there are more.
      sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
        Only images[::sampling, ::sampling] are shown, to speed up display.
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
        keys = [f"Image #{n+1}" for n in range(nimages)]
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
    tab = dbc.Tab([widget], label = "Carousel", className = "tab")
    # Update dashboard.
    with self.updatelock: # Lock on update.
      self.nupdates += 1
      self.layout = "carousel"
      self.images = None # No need to register images for the callbacks.
      self.content = [dbc.Tabs([tab], active_tab = "tab-0")]
      self.refresh = True

  ### Before/After layout.

  def slider(self, image1, image2, label1 = "Image", label2 = "Reference", sampling = -1):
    """Compare two images with a "before/after" slider on the dashboard.

    Args:
      image1: The "after" image as an equimage.Image object or numpy.ndarray.
      image2: The "before"" image as an equimage.Image object or numpy.ndarray.
      label1 (str, optional): The label of the first image (default "Image").
      label2 (str, optional): The label of the second image (default "Reference").
      sampling (int, optional): Downsampling rate (defaults to `jupyter.params.sampling` if negative).
        Only image1[::sampling, ::sampling] and image2[::sampling, ::sampling] are shown, to speed up display.
    """
    self.refresh = False # Stop refreshing dashboard.
    # Set-up before/after widget.
    image1, image2 = prepare_images_as_b64strings((image1, image2), sampling = sampling)
    baslider = dxt.BeforeAfter(after = dict(src = image1), before = dict(src = image2), width = f"{params.maxwidth}")
    left   = html.Div([label1], className = "ba-left", style = {"width": f"{params.lmargin}px"})
    middle = html.Div([baslider], className = "ba-middle", style = {"width": f"{params.maxwidth}px"})
    right  = html.Div([label2], className = "ba-right", style = {"width": f"{params.rmargin}px"})
    widget = html.Div([left, middle, right], className = "inline",
                      style = {"margin": f"{params.tmargin}px 0px {params.bmargin}px 0px"})
    tab = dbc.Tab([widget], label = "Compare images", className = "tab")
    # Update dashboard.
    with self.updatelock: # Lock on update.
      self.nupdates += 1
      self.layout = "slider"
      self.images = None # No need to register images for the callbacks.
      self.content = [dbc.Tabs([tab], active_tab = "tab-0")]
      self.refresh = True

#####################
# Helper functions. #
#####################

def _table_statistics_(image, channels = ""):
  """Prepare a table with the statistics of an image.

  Args:
    image (equimage.Image): The image.
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
  header = html.Thead(html.Tr([html.Th("Channel", className = "left"), html.Th("Minimum"), html.Th("25%"), html.Th("50%"),
                               html.Th("75%"), html.Th("Maximum"), html.Th("Shadowed"), html.Th("Highlighted")]))
  rows = []
  exclude01 = False
  for channel in stats.values():
    exclude01 = exclude01 or channel.exclude01
    deco = "\u207d\u2071\u207e" if channel.exclude01 else ""
    if channel.percentiles is not None:
      percentiles = [f"{channel.percentiles[0]:.5f}{deco}", f"{channel.percentiles[1]:.5f}{deco}", f"{channel.percentiles[2]:.5f}{deco}"]
    else:
      percentiles = 3*[f"None{deco}"]
    rows.append(html.Tr([html.Td(channel.name, className = "bold left", style = {"color": f"{channel.color}"}), html.Td(f"{channel.minimum:.5f}"),
                         html.Td(percentiles[0]), html.Td(percentiles[1]), html.Td(percentiles[2]), html.Td(f"{channel.maximum:.5f}"),
                         html.Td(f"{channel.zerocount} ({100.*channel.zerocount/channel.npixels:.2f}%)"),
                         html.Td(f"{channel.outcount} ({100.*channel.outcount/channel.npixels:.2f}%)")]))
  body = html.Tbody(rows)
  table = [dbc.Table([header]+[body], size = "sm", bordered = True, striped = True, className = "right", style = {"width": f"{params.maxwidth}px"})]
  if exclude01: table.append("\u207d\u2071\u207e Does not include pixels <= 0 or >= 1.")
  return html.Div(table, className = "tm8 bm8", style = {"margin-left": f"{params.lmargin}px", "margin-right": f"{params.rmargin}px"})
