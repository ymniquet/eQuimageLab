# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Image statistics & histograms."""

import numpy as np

from . import params

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  def statistics(self, channels = "RGBL"):
    """Compute image statistics of channels 'channels' of a RGB image or HSV image.
       'channels' is a combination of the keys "R" (for red), "G" (for green), "B" (for blue), "V" (for HSV value),
       "S" (for HSV saturation), and "L" (for luma). For a HSV image, only the statistics of the value and saturation
       can be computed.
       Return stats[key] for key in channels, with:
         - stats[key].name = channel name ("Red", "Green", "Blue", "Value", "Luma" or "Saturation", provided for convenience).
         - stats[key].width = image width (provided for convenience).
         - stats[key].height = image height (provided for convenience).
         - stats[key].npixels = number of image pixels = image width*image height (provided for convenience).
         - stats[key].minimum = minimum value in channel key.
         - stats[key].maximum = maximum value in channel key
         - stats[key].median = pr50 = median value in channel key (excluding pixels <= 0 and >= 1).
         - stats[key].percentiles = (pr25, pr50, pr75) = the 25th, 50th and 75th percentiles in channel key (excluding pixels <= 0 and >= 1).
         - stats[key].zerocount = number of pixels <= 0 in channel key.
         - stats[key].oorcount = number of pixels  > 1 (out-of-range) in channel key.
        The statistics are also embedded in the object as self.stats."""
    class Container: pass # An empty container class.
    width, height = self.width_height()
    npixels = width*height
    stats = {}
    for key in channels:
      if stats.get(key, None) is not None: # Already computed.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if key == "R":
        self.check_color_model("RGB")
        name = "Red"
        channel = self[0]
      elif key == "G":
        self.check_color_model("RGB")     
        name = "Green"
        channel = self[1]
      elif key == "B":
        self.check_color_model("RGB")
        name = "Blue"
        channel = self[2]
      elif key == "V":
        name = "Value"
        channel = self.value()
       elif key == "S":
        name = "Saturation"
        channel = self.saturation()
      elif key == "L"
        name = "Luma"
        channel = self.luma()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      stats[key] = Container()
      stats[key].name = name
      stats[key].width = width
      stats[key].height = height
      stats[key].npixels = npixels
      stats[key].minimum = channel.min()
      stats[key].maximum = channel.max()
      mask = (channel >= params.IMGTOL) & (channel <= 1.-params.IMGTOL)
      if np.any(mask):
        stats[key].percentiles = np.percentile(channel[mask], [25., 50., 75.])
        stats[key].median = stats[key].percentiles[1]
      else:
        stats[key].percentiles = None
        stats[key].median = None
      stats[key].zerocount = np.sum(channel < params.IMGTOL)
      stats[key].outcount = np.sum(channel > 1.+params.IMGTOL)
    self.stats = stats
    return stats

  def histograms(self, channels = "RGBL", nbins = None):
    """Compute histograms of channels 'channels' of a RGB or HSV image.
       'channels' is a combination of the keys "R" (for red), "G" (for green), "B" (for blue), "V" (for HSV value),
       "S" (for HSV saturation), and "L" (for luma). For a HSV image, only the histograms of the value and saturation
       can be computed. 'nbins' is the number of bins in the [0, 1] range (automatically adjusted if None).
       Return hists[key] for key in channels, with:
         - hists[key].name = channel name ("Red", "Green", "Blue", "Value", "Luma" or "Saturation", provided for convenience).
         - hists[key].color = suggested line color for plots. 
         - hists[key].edges = histogram bins edges.
         - hists[key].counts = histogram bins counts.
        The histograms are also embedded in the object as self.hists."""
    class Container: pass # An empty container class.
    hists = {}
    for key in channels: 
      if hists.get(key, None) is not None: # Already computed.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if key == "R":
        self.check_color_model("RGB")
        name = "Red"
        color = "red"
        channel = self[0]
      elif key == "G":
        self.check_color_model("RGB")     
        name = "Green"
        color = "green"
        channel = self[1]
      elif key == "B":
        self.check_color_model("RGB")
        name = "Blue"
        color = "blue"
        channel = self[2]
      elif key == "V":
        name = "Value"
        color = "black"
        channel = self.value()
       elif key == "S":
        name = "Saturation"
        color = "orange"
        channel = self.saturation()
      elif key == "L"
        name = "Luma"
        color = "gray"
        channel = self.luma()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      minimum = channel.min()
      maximum = channel.max()
      if nbins is None:
        median = channel.median()
        span = min(median-minimum, maximum-median)
        span = max(span, 1.e-3)
        nbinsc = int(round(128./span))
      else:
        nbinsc = nbins
      nbinsc = int(round(nbinsc*(maximum-minimum)))
      if nbinsc > 32768: # Limit number of bins.
        nbinsc = 32768
        print(f"Warning, limiting the number of bins to {nbinsc} for channel '{key}'.")
      hists[key] = Container(l)
      hists[key].name = name
      hists[key].color = color
      hists[key].counts, hists[key].edges = np.histogram(channel, bins = nbinsc, range = (minimum, maximum), density = False)
    self.hists = hists
    return hists
    