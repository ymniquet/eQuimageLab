# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Jupyter-lab display management."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "jupyterlab"

from . import params
from ..equimage.image import Image

def show(image, histograms = None, statistics = None, width = params.maxwidth, height = params.maxheight, sample = 1, renderer = None):
  if isinstance(image, Image):
    image = image.get_image(channels = -1)
  if image.ndim == 2:
    sampled = image[::sample, ::sample]
  elif image.shape[2] == 1:
    sampled = image[::sample, ::sample, 0]
  else:
    sampled = image[::sample, ::sample, :]
  figure = px.imshow(sampled, zmin = 0., zmax = 1., aspect = "equal", binary_string = True)
  layout = go.Layout(autosize = True, height = height) #, margin = go.layout.Margin(l = 0, r = 0, b = 0, t = 0))
  widget = go.FigureWidget(data = figure) # Fails to account for layout ??
  widget.update_layout(layout)
  widget.show(renderer)

def show_statistics(image, channels = None, renderer = None):
  if channels is None:
    stats = getattr(image, "stats", None)
    if stats is None: stats = image.statistics()
  else:
    stats = image.statistics(channels)
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
  header = dict(values = ["Channel", "Minimum", "25%", "Median (50%)", "75%", "Maximum", "Shadowed", "Highlighted"], align = align)
  cells = dict(values = columns, align = align)
  table = go.Table(header = header, cells = cells, columnwidth = [1, 1, 1, 1, 1, 1, 1.5, 1.5])
  widget = go.FigureWidget(data = table)
  widget.show(renderer)

def show_histograms(image, log = False, channels = None, renderer = None):
  if channels is None:
    hists = getattr(image, "hists", None)
    if hists is None: hists = image.histograms()
  else:
    hists = image.histograms(channels)
  widget = go.FigureWidget()
  for channel in hists.values():
    x = (channel.edges[1:]+channel.edges[:-1])/2.
    widget.add_trace(go.Scatter(x = x, y = channel.counts, name = channel.name, line = dict(color = channel.color, width = 2)))
  if log: widget.update_yaxes(type = "log")
  widget.update_layout(xaxis_title = "level", yaxis_title = "count")
  widget.show(renderer)

def compare(images, width = params.maxwidth, height = params.maxheight, sample = 1, renderer = None):
  return None

def shadowed(image):
  return None

def highlighted(image):
  return None

# def red, blue, green, value, luma, lightness...
