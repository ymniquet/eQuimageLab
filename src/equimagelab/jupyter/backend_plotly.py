# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Plotly backend for Jupyter-lab interface."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "jupyterlab"

from . import params
from .utils import prepare_images

from ..equimage.image import Image

def _show_image_(image, sample = 1, width = params.maxwidth):
  img = prepare_images(image, sample = sample)
  if img.shape[0] == 1:
    img = img[0]
  else:
    img = np.moveaxis(img, 0, -1)
  raster = px.imshow(img, zmin = 0., zmax = 1., aspect = "equal", binary_string = True)
  figure = go.Figure(data = raster)
  layout = go.Layout(width = width+params.lmargin+params.rmargin, height = width*img.shape[0]/img.shape[1]+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True))
  figure.update_layout(layout)
  return figure

def show(image, histograms = False, statistics = False, sample = 1, width = params.maxwidth, renderer = None):
  figure = _show_image_(image, sample = sample, width = width)
  figure.show(renderer)
  if histograms is not False:
    if histograms is True: histograms = ""
    show_histograms(image, channels = histograms, width = width, renderer = renderer)
  if statistics is not False:
    if statistics is True: statistics = ""
    show_statistics(image, channels = statistics, width = width, renderer = renderer)

def _show_histograms_(image, channels = "", log = True, trans = None, width = params.maxwidth):
  if not issubclass(type(image), Image):
    print("The histograms can only be displayed for Image objects.")
    return None
  if channels == "":
    hists = getattr(image, "hists", None)
    if hists is None: hists = image.histograms()
  else:
    hists = image.histograms(channels = channels)
  figure = make_subplots(specs = [[{"secondary_y": trans is not None, "r": -0.06}]])
  for channel in hists.values():
    midpoints = (channel.edges[1:]+channel.edges[:-1])/2.
    figure.add_trace(go.Scatter(x = midpoints, y = channel.counts, name = channel.name, mode = "lines", line = dict(color = channel.color, width = 2)), secondary_y = False)
  figure.update_xaxes(title_text = "Level", ticks = "inside", rangemode = "tozero")
  figure.update_yaxes(title_text = "Count", ticks = "inside", secondary_y = False)
  if log:
    figure.update_yaxes(type = "log", secondary_y = False)
  else:
    figure.update_yaxes(rangemode = "tozero", secondary_y = False)
  if trans is not None:
    figure.add_trace(go.Scatter(x = trans.x, y = trans.y, mode = "lines", line = dict(color = "purple", width = 2), showlegend = False), secondary_y = True)
    figure.add_trace(go.Scatter(x = [0., 1.], y = [0., 1.], mode = "lines", line = dict(color = "purple", width = 1, dash = "dot"), showlegend = False), secondary_y = True)
    figure.update_yaxes(title_text = trans.ylabel, titlefont = dict(color = "purple"), ticks = "inside", tickfont = dict(color = "purple"),
                        showgrid = False, rangemode = "tozero", secondary_y = True)
  layout = go.Layout(width = width+params.lmargin+params.rmargin, height = width/3+params.bmargin+params.tmargin,
                     margin = go.layout.Margin(l = params.lmargin, r = params.rmargin, b = params.bmargin, t = params.tmargin, autoexpand = True))
  figure.update_layout(layout, legend = dict(xanchor = "left", x = 1.05, yanchor = "top", y = 1.))
  return figure

def show_histograms(image, channels = "", log = True, trans = None, width = params.maxwidth, renderer = None):
  figure = _show_histograms_(image, channels = channels, log = log, trans = trans, width = width)
  if figure is not None: figure.show(renderer)

def _show_statistics_(image, channels = "", width = params.maxwidth, rowheight = params.rowheight):
  if not issubclass(type(image), Image):
    print("The statistics can only be displayed for Image objects.")
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

def show_statistics(image, channels = "", width = params.maxwidth, rowheight = params.rowheight, renderer = None):
  figure = _show_statistics_(image, channels = channels, width = width, rowheight = rowheight)
  if figure is not None: figure.show(renderer)

def show_t(image, histograms = "RGBL", sample = 1, width = params.maxwidth, renderer = None):
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
  show_histograms(reference, channels = histograms, log = True, trans = trans, width = width, renderer = renderer)
  show_histograms(image, channels = histograms, log = True, width = width, renderer = renderer)
  show(image, histograms = False, statistics = False, sample = sample, width = width, renderer = renderer)
