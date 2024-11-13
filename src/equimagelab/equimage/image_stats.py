# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01
# DOC+MCI.

"""Image statistics & histograms."""

import numpy as np

from . import params
from . import helpers

#####################################
# For inclusion in the Image class. #
#####################################

class Mixin:
  """To be included in the Image class."""

  def statistics(self, channels = "RGBL"):
    """Compute statistics of selected channels of the image.

    Args:
      channels (str, optional): A combination of the keys "R" (for red), "G" (for green), "B" (for blue),
        "V" (for HSV value), "S" (for HSV saturation), and "L" (for luma). For a HSV image, only the
        statistics of the value and saturation can be computed. Default is "RGBL".

    Returns:
      dict: stats[key] for key in channels, with:
        - stats[key].name = channel name ("Red", "Green", "Blue", "Value", "Luma" or "Saturation", provided for convenience).
        - stats[key].width = image width (provided for convenience).
        - stats[key].height = image height (provided for convenience).
        - stats[key].npixels = number of image pixels = image width*image height (provided for convenience).
        - stats[key].minimum = minimum level.
        - stats[key].maximum = maximum level.
        - stats[key].median = pr50 = median level (excluding pixels <= 0 and >= 1).
        - stats[key].percentiles = (pr25, pr50, pr75) = the 25th, 50th and 75th percentiles (excluding pixels <= 0 and >= 1).
        - stats[key].zerocount = number of pixels <= 0.
        - stats[key].outcount = number of pixels > 1 (out-of-range).

    Note: The statistics are also embedded in the object as self.stats.
    """
    width, height = self.get_size()
    npixels = width*height
    stats = {}
    for key in channels:
      if stats.get(key, None) is not None: # Already computed.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if key == "R":
        self.check_color_model("RGB", "gray")
        name = "Red"
        channel = self.image[0]
      elif key == "G":
        self.check_color_model("RGB", "gray")
        name = "Green"
        channel = self.image[1] if self.colormodel == "RGB" else self.image[0]
      elif key == "B":
        self.check_color_model("RGB", "gray")
        name = "Blue"
        channel = self.image[2] if self.colormodel == "RGB" else self.image[0]
      elif key == "V":
        name = "Value"
        channel = self.value()
      elif key == "S":
        name = "Saturation"
        channel = self.saturation()
      elif key == "L":
        name = "Luma"
        channel = self.luma()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      stats[key] = helpers.Container()
      stats[key].name = name
      stats[key].width = width
      stats[key].height = height
      stats[key].npixels = npixels
      stats[key].minimum = np.min(channel)
      stats[key].maximum = np.max(channel)
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
    """Compute histograms of selected channels of the image.

    Args:
      channels (str, optional): A combination of the keys "R" (for red), "G" (for green), "B" (for blue),
        "V" (for HSV value), "S" (for HSV saturation), and "L" (for luma). For a HSV image, only the
        statistics of the value and saturation can be computed. Default is "RGBL".
      nbins (int, optional): Number of bins in the histograms (auto if None, default).

    Returns:
      dict: hists[key] for key in channels, with:
        - hists[key].name = channel name ("Red", "Green", "Blue", "Value", "Luma" or "Saturation", provided for convenience).
        - hists[key].color = suggested line color for plots.
        - hists[key].edges = histogram bins edges.
        - hists[key].counts = histogram bins counts.

    Note: The histograms are also embedded in the object as self.hists.
    """
    hists = {}
    for key in channels:
      if hists.get(key, None) is not None: # Already computed.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if key == "R":
        self.check_color_model("RGB", "gray")
        name = "Red"
        color = "red"
        channel = self.image[0]
      elif key == "G":
        self.check_color_model("RGB", "gray")
        name = "Green"
        color = "green"
        channel = self.image[1] if self.colormodel == "RGB" else self.image[0]
      elif key == "B":
        self.check_color_model("RGB", "gray")
        name = "Blue"
        color = "blue"
        channel = self.image[2] if self.colormodel == "RGB" else self.image[0]
      elif key == "V":
        name = "Value"
        color = "darkslategray"
        channel = self.value()
      elif key == "S":
        name = "Saturation"
        color = "orange"
        channel = self.saturation()
      elif key == "L":
        name = "Luma"
        color = "lightslategray"
        channel = self.luma()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      minimum = np.min(channel)
      maximum = np.max(channel)
      if nbins is None:
        median = np.median(channel)
        span = min(median-minimum, maximum-median)
        span = max(span, 1.e-3)
        nbinsc = int(round(128/span))
      else:
        nbinsc = nbins
      nbinsc = int(round(nbinsc*(maximum-minimum)))
      if nbinsc > 32768: # Limit the number of bins.
        nbinsc = 32768
        print(f"Warning, limiting the number of bins to {nbinsc} for channel '{key}'.")
      hists[key] = helpers.Container()
      hists[key].name = name
      hists[key].color = color
      hists[key].counts, hists[key].edges = np.histogram(channel, bins = nbinsc, range = (minimum, maximum), density = False)
    self.hists = hists
    return hists
