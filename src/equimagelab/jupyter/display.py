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
  widget = go.FigureWidget(data = figure) #, layout = layout) # Fails to account for layout ??
  widget.update_layout(layout)
  widget.show(renderer)

def compare(images, width = params.maxwidth, height = params.maxheight, sample = 1, renderer = None):
  return None

def shadowed(image):
  return None

def highlighted(image):
  return None

# def red, blue, green, value, luma, lightness...
