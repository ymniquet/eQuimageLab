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

  def statistics(self, channels = "RGBVL"):
    """Compute image statistics for channels 'channels', a combination of the keys "R" (for red), "G" (for green), "B" (for blue),
       "V" (for HSV value), "L" (for luma) and "S" (for HSV saturation). Return stats[key] for key in channels, with:
         - stats[key].name = channel name ("Red", "Green", "Blue", "Value", "Luma" or "Saturation", provided for convenience).
         - stats[key].width = image width (provided for convenience).
         - stats[key].height = image height (provided for convenience).
         - stats[key].npixels = number of image pixels = image width*image height (provided for convenience).
         - stats[key].minimum = minimum value in channel key.
         - stats[key].maximum = maximum value in channel key.
         - stats[key].percentiles = (pr25, pr50, pr75) = the 25th, 50th and 75th percentiles in channel key (excluding pixels <= 0 and >= 1).
         - stats[key].median = pr50 = median value in channel key (excluding pixels <= 0 and >= 1).
         - stats[key].zerocount = number of pixels <= 0 in channel key.
         - stats[key].oorcount = number of pixels  > 1 (out-of-range) in channel key."""
    class Container: pass # An empty container class.
    width, height = self.size()
    npixels = width*height
    stats = {}
    for key in channels:
      if key == "R":
        name = "Red"
        channel = self.rgb[0]
      elif key == "G":
        name = "Green"
        channel = self.rgb[1]
      elif key == "B":
        name = "Blue"
        channel = self.rgb[2]
      elif key == "V":
        name = "Value"
        channel = self.value()
      elif key == "L":
        name = "Luma"
        channel = self.luma()
      elif key == "S":
        name = "Saturation"
        channel = self.saturation()
      else:
        raise ValueError(f"Error, invalid channel '{key}'.")
      stats[key] = Container()
      stats[key].name = name
      stats[key].width = width
      stats[key].height = height
      stats[key].npixels = npixels
      stats[key].minimum = channel.min()
      stats[key].maximum = channel.max()
      mask = (channel >= IMGTOL) & (channel <= 1.-IMGTOL)
      if np.any(mask):
        stats[key].percentiles = np.percentile(channel[mask], [25., 50., 75.])
        stats[key].median = stats[key].percentiles[1]
      else:
        stats[key].percentiles = None
        stats[key].median = None
      stats[key].zerocount = np.sum(channel < IMGTOL)
      stats[key].outcount = np.sum(channel > 1.+IMGTOL)
    return stats

  def histograms(self, channels = "RGBVL", nbins = 256):
    """Return image histograms for channels 'channels', a combination of the keys "R" (for red), "G" (for green), "B" (for blue),
       "V" (for HSV value), "L" (for luma) and "S" (for HSV saturation). Return a tuple (edges, counts), where edges(nbins) are
       the bin edges and counts(len(channels), nbins) are the bin counts for all channels. 'nbins' is the number of bins in the
       range [0, 1]."""
    minimum = min(0., self.rgb.min())
    maximum = max(1., self.rgb.max())
    nbins = int(round(nbins*(maximum-minimum)))
    counts = np.empty((len(channels), nbins))
    ic = 0
    for key in channels:
      if key == "R":
        channel = self.rgb[0]
      elif key == "G":
        channel = self.rgb[1]
      elif key == "B":
        channel = self.rgb[2]
      elif key == "V":
        channel = self.value()
      elif key == "L":
        channel = self.luma()
      elif key == "S":
        channel = self.saturation()
      else:
        raise ValueError(f"Error, invalid channel '{key}'.")
      counts[ic], edges = np.histogram(channel, bins = nbins, range = (minimum, maximum), density = False)
      ic += 1
    return edges, counts
    