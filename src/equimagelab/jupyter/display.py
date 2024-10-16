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

def show(image, histograms = None, statistics = None, width = params.maxwidth, height = params.maxheight, sample = 1, renderer = None):
  if image.ndim == 2:
    image = image[::sample, ::sample]
  else:
    image = np.moveaxis(image[:, ::sample, ::sample], 0, -1)
  figure = px.imshow(image, zmin = 0., zmax = 1., aspect = "equal", binary_string = True)
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
