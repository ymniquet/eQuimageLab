# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.4.1 / 2025.05.30
# Doc OK.

"""Plotly backend for JupyterLab interface.

The following symbols are imported in the equimagelab namespace for convenience:
  "show", "show_t", "show_histograms", "show_statistics", "light_curve".
"""

__all__ = ["show", "show_t", "show_histograms", "show_statistics", "light_curve"]

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "jupyterlab"

from equimage import Image
from equimage.image_stats import parse_channels

from . import params
from .utils import format_images

#####################
# Helper functions. #
#####################

def _figure_formatted_image_(image, dx = 1, dy = 1, width = -1, hover = False, template = "plotly_dark"):
  """Prepare a ploty figure for the input (formatted) image.

  Args:
    image (numpy.ndarray): The prepared formatted (namely, processed by :meth:`jupyter.utils.format_images() <.format_images>`).
    dx (int, optional): The size of a pixel along x (default 1).
    dy (int, optional): The size of a pixel along y (default 1).
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    hover (bool, optional): If True, show the image data on hover (default False).
      Warning: setting hover = True can slow down display a lot !
    template (str, optional): The template for the figure (default "plotly_dark").

  Returns:
    plotly.graph_objects.Figure: A plotly figure with the image.
  """
  if width <= 0: width = params.maxwidth
  # Plot image.
  x = np.arange(0, image.shape[1])*dx
  y = np.arange(0, image.shape[0])*dy
  figure = px.imshow(image, x = x, y = y, zmin = 0., zmax = 1., aspect = "equal", binary_string = not hover)
  figure.update_traces(name = "", hovertemplate = "(%{x}, %{y}): %{z}" if hover else "(%{x}, %{y})")
  layout = go.Layout(template = template,
                     width = width+params.lmargin+params.rmargin, height = width*image.shape[0]/image.shape[1]+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True))
  figure.update_layout(layout)
  return figure

def _figure_image_(image, sampling = -1, width = -1, hover = False, template = "plotly_dark"):
  """Prepare a ploty figure for the input image.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    sampling (int, optional): The downsampling rate (defaults to `jupyter.params.sampling` if negative).
      Only image[::sampling, ::sampling] is shown, to speed up display.
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    hover (bool, optional): If True, show the image data on hover (default False).
      Warning: setting hover = True can slow down display a lot !
    template (str, optional): The template for the figure (default "plotly_dark").

  Returns:
    plotly.graph_objects.Figure: A plotly figure with the image.
  """
  if sampling <= 0: sampling = params.sampling
  return _figure_formatted_image_(format_images(image, sampling = sampling), dx = sampling, dy = sampling, width = width, hover = hover, template = template)

def _figure_histograms_(image, channels = "", log = True, width = -1, xlabel = "Level", trans = None, template = "plotly_dark"):
  """Prepare a plotly figure with the histograms of an image.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    channels (str, optional): The channels of the histograms (default "" = "RGBL" for red, green,
      blue, luma).
    log (bool, optional): If True (default), plot the histogram counts in log scale.
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    xlabel (str, optional): The x axis label of the plot (default "Level").
    trans (optional): A container with an histogram transformation (see :meth:`Image.apply_channels() <.apply_channels>`),
      plotted on top of the histograms (default None).
    template (str, optional): The template for the figure (default "plotly_dark").

  Returns:
    plotly.graph_objects.Figure: A plotly figure with the histograms of the image.
  """
  if width <= 0: width = params.maxwidth
  # Prepare histograms.
  if not issubclass(type(image), Image): image = Image(image, channels = -1)
  if channels == "":
    hists = getattr(image, "hists", None)
    if hists is None: hists = image.histograms()
  else:
    hists = image.histograms(channels = channels)
  # Set-up lines.
  mline = dict(color = params.mlinecolor)
  mline2 = dict(color = params.mlinecolor, width = 2)
  mlinedot1 = dict(color = params.mlinecolor, dash = "dot", width = 1)
  mlinedash1 = dict(color = params.mlinecolor, dash = "dash", width = 1)
  mlinedashdot1 = dict(color = params.mlinecolor, dash = "dashdot", width = 1)
  # Plot histograms.
  figure = make_subplots(specs = [[dict(secondary_y = trans is not None, r = -0.06)]])
  updatemenus = []
  n = len(hists)
  for channel in hists.values():
    midpoints = (channel.edges[1:]+channel.edges[:-1])/2.
    figure.add_trace(go.Scatter(x = midpoints, y = channel.counts, name = channel.name, mode = "lines", line = dict(color = channel.color, width = 2)),
                     secondary_y = False)
  figure.add_vline(x = 0., line = mlinedashdot1, secondary_y = False)
  figure.add_vline(x = 1., line = mlinedashdot1, secondary_y = False)
  figure.update_xaxes(title_text = xlabel, ticks = "inside", rangemode = "tozero")
  figure.update_yaxes(title_text = "Count", ticks = "inside", rangemode = "tozero", secondary_y = False)
  if log:
    figure.update_yaxes(type = "log", secondary_y = False)
    active =  0
  else:
    active = -1
  # Add lin/log toggle button.
  ybutton = 1.+.025*1024./width
  buttons = [dict(label = "lin/log", method = "relayout", args = [{"yaxis.type": "log"}], args2 = [{"yaxis.type": "linear"}])]
  updatemenus.append(dict(type = "buttons", buttons = buttons, active = active, showactive = False,
                          xanchor = "left", x = 0., yanchor = "bottom", y = ybutton))
  # Plot transformation.
  if trans is not None:
    if trans.type == "hist":
      cef = np.log(np.maximum(np.gradient(trans.y, trans.x), 1.e-12))
      figure.add_trace(go.Scatter(x = trans.x, y = trans.y, name = trans.ylabel, mode = "lines", line = mline2, showlegend = False),
                       secondary_y = True)
      m = 2 if hasattr(trans, "xm") else 1
      if m == 2:
        figure.add_trace(go.Scatter(x = trans.xm, y = trans.ym, mode = "markers", marker = dict(size = 8, color = params.mlinecolor), showlegend = False),
                         secondary_y = True)
      figure.add_trace(go.Scatter(x = trans.x, y = cef, name = f"log {trans.ylabel}'", mode = "lines", line = mline2, showlegend = False, visible = False),
                       secondary_y = True)
      figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 0.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False),
                       secondary_y = True)
      figure.add_trace(go.Scatter(x = [0., 1.], y = [1., 1.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False),
                       secondary_y = True)
      figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 1.], name = "", mode = "lines", line = mlinedot1, showlegend = False),
                       secondary_y = True)
      if hasattr(trans, "xticks"):
        for xtick in trans.xticks:
          figure.add_vline(x = xtick, line = mlinedash1, secondary_y = True)
      ftitle = f"{trans.ylabel}({trans.channels})"
      ceftitle = f"log {trans.ylabel}'({trans.channels})"
      figure.update_yaxes(title_text = ftitle, title_font = mline, ticks = "inside", tickfont = mline, showgrid = False, rangemode = "tozero", secondary_y = True)
      # Add f/log f' toggle button.
      buttons = [dict(label = "f/log f'", method = "update",
                      args  = [{"visible": n*[True]+m*[False]+[True , True, False, False]}, {"yaxis2.title": ceftitle}],
                      args2 = [{"visible": n*[True]+m*[True ]+[False, True, True , True ]}, {"yaxis2.title": ftitle}])]
      xbutton = .066*1024./width
      updatemenus.append(dict(type = "buttons", buttons = buttons, active = -1, showactive = False,
                              xanchor = "left", x = xbutton, yanchor = "bottom", y = ybutton))
    elif trans.type == "hue":
      figure.add_trace(go.Scatter(x = trans.x, y = trans.y, name = trans.ylabel, mode = "lines", line = mline2, showlegend = False),
                       secondary_y = True)
      figure.add_trace(go.Scatter(x = trans.xm, y = trans.ym, name = trans.ylabel, mode = "markers", marker = dict(size = 16, color = trans.cm), showlegend = False),
                       secondary_y = True)
      figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 0.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False),
                       secondary_y = True)
      figure.update_xaxes(range = [0., 1.])
      figure.update_yaxes(title_text = trans.ylabel, title_font = mline, ticks = "inside", tickfont = mline, showgrid = False, rangemode = "tozero", secondary_y = True)
    else:
      print(f"Can not handle transformations of type '{trans.type}'.")
  # Finalize layout.
  xlegend = 1.+.05*1024./width
  layout = go.Layout(template = template,
                     width = width+params.lmargin+params.rmargin, height = width/3+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True),
                     legend = dict(xanchor = "left", x = xlegend, yanchor = "top", y = 1.))
  figure.update_layout(layout, updatemenus = updatemenus)
  return figure

def _figure_statistics_(image, channels = "", width = -1, rowheight = -1, template = "plotly_dark"):
  """Prepare a plotly table with the statistics of an image.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    channels (str, optional): The channels of the statistics (default "" = "RGBL" for red, green,
      blue, luma).
    width (int, optional): The width of the table (defaults to `jupyter.params.maxwidth` if negative).
    rowheight (int, optional): The height of the rows (default to jupyter.params.rowheight if negative).
    template (str, optional): The template for the figure (default "plotly_dark").

  Returns:
    plotly.graph_objects.Figure: A plotly figure with the table of the statistics of the image.
  """
  if width <= 0: width = params.maxwidth
  if rowheight <= 0: rowheight = params.rowheight
  # Prepare statistics.
  if not issubclass(type(image), Image): image = Image(image, channels = -1)
  if channels == "":
    stats = getattr(image, "stats", None)
    if stats is None: stats = image.statistics()
  else:
    stats = image.statistics(channels = channels)
  # Create table.
  columns = [[], [], [], [], [], [], [], []]
  for channel in stats.values():
    columns[0].append(channel.name)
    columns[1].append(f"{channel.minimum:.5f}")
    if channel.percentiles is not None:
      columns[2].append(f"{channel.percentiles[0]:.5f}")
      columns[3].append(f"{channel.percentiles[1]:.5f}")
      columns[4].append(f"{channel.percentiles[2]:.5f}")
    else:
      columns[2].append("None")
      columns[3].append("None")
      columns[4].append("None")
    columns[5].append(f"{channel.maximum:.5f}")
    columns[6].append(f"{channel.zerocount} ({100.*channel.zerocount/channel.npixels:.2f}%)")
    columns[7].append(f"{channel.outcount} ({100.*channel.outcount/channel.npixels:.2f}%)")
  align = ["left"]+7*["right"]
  header = dict(values = ["Channel", "Minimum", "25%", "50%", "75%", "Maximum", "Shadowed", "Highlighted"], align = align, height = rowheight)
  cells = dict(values = columns, align = align, height = rowheight)
  table = go.Table(header = header, cells = cells, columnwidth = [1, 1, 1, 1, 1, 1, 1.5, 1.5])
  # Create figure.
  figure = go.Figure(data = table)
  layout = go.Layout(template = template,
                     width = width+params.lmargin+params.rmargin, height = (len(stats)+1)*rowheight+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = 32, t = 32, autoexpand = True))
  figure.update_layout(layout)
  return figure

#####################
# Plotly interface. #
#####################

def show(image, histograms = False, statistics = False, sampling = -1, width = -1, hover = False, renderer = None):
  """Show an image using plotly.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    histograms (optional): If True or a string, show the histograms of the image. The string lists
      the channels of the histograms (see :meth:`Image.histograms() <.histograms>`). True is
      substituted with "RGBL" (red, green, blue, luma). Default is False.
    statistics (optional): If True or a string, show the statistics of the image. The string lists
      the channels of the statistics (see :meth:`Image.statistics() <.statistics>`). True is
      substituted with "RGBL" (red, green, blue, luma). Default is False.
    sampling (int, optional): The downsampling rate (defaults to `jupyter.params.sampling` if negative).
      Only image[::sampling, ::sampling] is shown, to speed up display.
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    hover (bool, optional): If True, show the image data on hover (default False).
      Warning: setting hover = True can slow down display a lot !
    renderer (str, optional): The plotly renderer (default None = "jupyterlab").
  """
  figure = _figure_image_(image, sampling = sampling, width = width, hover = hover)
  figure.show(renderer)
  if histograms is not False:
    if histograms is True: histograms = ""
    show_histograms(image, channels = histograms, width = width, renderer = renderer)
  if statistics is not False:
    if statistics is True: statistics = ""
    show_statistics(image, channels = statistics, width = width, renderer = renderer)

def show_histograms(image, channels = "", log = True, width = -1, xlabel = "Level", trans = None, renderer = None):
  """Plot the histograms of an image using plotly.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    channels (str, optional): The channels of the histograms (default "" = "RGBL" for red, green,
      blue, luma). See :meth:`Image.histograms() <.histograms>`.
    log (bool, optional): If True (default), plot the histogram counts in log scale.
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    xlabel (str, optional): The x axis label of the plot (default "Level").
    trans (optional): A container with an histogram transformation (see :meth:`Image.apply_channels() <.apply_channels>`),
      plotted on top of the histograms (default None).
    renderer (str, optional): The plotly renderer (default None = "jupyterlab").
  """
  figure = _figure_histograms_(image, channels = channels, log = log, width = width, xlabel = xlabel, trans = trans)
  if figure is not None: figure.show(renderer)

def show_statistics(image, channels = "", width = -1, rowheight = -1, renderer = None):
  """Display a table with the statistics of an image using plotly.

  Args:
    image: An Image object or numpy.ndarray with shape (height, width, 3) (for a color image),
      (height, width, 1) or (height, width) (for a grayscale image).
    channels (str, optional): The channels of the statistics (default "" = "RGBL" for red, green,
      blue, luma). See :meth:`Image.statistics() <.statistics>`.
    width (int, optional): The width of the table (defaults to `jupyter.params.maxwidth` if negative).
    rowheight (int, optional): The height of the rows (default to `jupyter.params.rowheight` if negative).
    renderer (str, optional): The plotly renderer (default None = "jupyterlab").
  """
  figure = _figure_statistics_(image, channels = channels, width = width, rowheight = rowheight)
  if figure is not None: figure.show(renderer)

def show_t(image, channels = "RGBL", sampling = -1, width = -1, hover = False, renderer = None):
  """Show an image embedding an histogram transformation using plotly.

  Displays the input histograms with the transformation curve, the output histograms, and the output
  image.

  Args:
    image (Image): The output image (must embed a transformation image.trans -
      see :meth:`Image.apply_channels() <.apply_channels>`).
    channels (str, optional): The channels of the histograms (default "" = "RGBL" for red, green,
      blue, luma). The channels of the transformation are automatically appended.
      See :meth:`Image.histograms() <.histograms>`.
    sampling (int, optional): The downsampling rate (defaults to `jupyter.params.sampling` if negative).
      Only image[::sampling, ::sampling] is shown, to speed up display.
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    hover (bool, optional): If True, show the image data on hover (default False).
      Warning: setting hover = True can slow down display a lot !
    renderer (str, optional): The plotly renderer (default None = "jupyterlab").
  """
  if not issubclass(type(image), Image):
    print("The transformations can only be displayed for Image objects.")
    return
  trans = getattr(image, "trans", None)
  if trans is None:
    print("There is no transformation embedded in the input image.")
    return
  reference = trans.input
  keys = parse_channels(channels)
  for key in parse_channels(trans.channels, errors = False):
    if not key in keys: channels += key
  show_histograms(reference, channels = channels, log = True, width = width, xlabel = "Input level", trans = trans, renderer = renderer)
  show_histograms(image, channels = channels, log = True, width = width, xlabel = "Output level", renderer = renderer)
  show(image, histograms = False, statistics = False, sampling = sampling, width = width, hover = hover, renderer = renderer)

def light_curve(image, reference, maxpoints = 32768, width = -1, renderer = None):
  """Plot light curve (scatter plot of an output image channel vs an input reference channel).

  Args:
    image (numpy.ndarray): The output image channel (luma, ...) as an array with shape (height, width).
    reference (numpy.ndarray): The input reference channel as an array with shape (height, width).
    maxpoints (int, optional): The maximum number of points in the scatter plot. The image and
      reference will be sampled accordingly (default 32768).
    width (int, optional): The width of the figure (defaults to `jupyter.params.maxwidth` if negative).
    renderer (str, optional): The plotly renderer (default None = "jupyterlab").
  """
  if width <= 0: width = params.maxwidth
  # Prepare light curve.
  if image.ndim != 2 or reference.ndim != 2:
    raise ValueError("Error, image and reference must be 2D arrays.")
  if image.shape != reference.shape:
    raise ValueError("Error, image and reference must have the same shape.")
  input = reference.ravel()
  output = image.ravel()
  step = int(np.ceil(len(input)/maxpoints))
  # Set-up lines.
  mlinedot1 = dict(color = params.mlinecolor, dash = "dot", width = 1)
  mlinedashdot1 = dict(color = params.mlinecolor, dash = "dashdot", width = 1)
  # Plot light curve.
  figure = go.Figure()
  figure.add_trace(go.Scatter(x = input[::step], y = output[::step],
                              name = "Light curve", mode = "markers", marker = dict(size = 4, color = "lightslategray"), showlegend = False))
  figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 0.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False))
  figure.add_trace(go.Scatter(x = [0., 1.], y = [1., 1.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False))
  figure.add_trace(go.Scatter(x = [0., 0.], y = [0., 1.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False))
  figure.add_trace(go.Scatter(x = [1., 1.], y = [0., 1.], name = "", mode = "lines", line = mlinedashdot1, showlegend = False))
  figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 1.], name = "", mode = "lines", line = mlinedot1, showlegend = False))
  figure.update_xaxes(title_text = "Input level", ticks = "inside", rangemode = "tozero")
  figure.update_yaxes(title_text = "Output level", ticks = "inside", rangemode = "tozero")
  layout = go.Layout(template = "plotly_dark",
                     width = width+params.lmargin+params.rmargin, height = width+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True))
  figure.update_layout(layout)
  figure.show(renderer)
