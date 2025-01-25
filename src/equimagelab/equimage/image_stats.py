# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.1.1 / 2025.01.25
# Sphinx OK.

"""Image statistics & histograms."""

import numpy as np

from . import params
from . import helpers

def parse_channels(channels, errors = True):
  """Parse channel keys.

  Args:
    channels (str): A combination of the keys "R" (for red), "G" (for green), "B" (for blue), "V" (for HSV value),
      "S" (for HSV saturation), "L" (for luma), and "L*" (for lightness).
    errors (bool, optional): If False, discard unknown channel keys; If True (default), raise a ValueError.

  Returns:
    list: The list of channel keys.
  """
  keys = []
  prevkey = None
  for key in channels:
    ok = False
    if key in ["R", "G", "B", "V", "S", "L"]:
      keys.append(key)
      ok = True
    elif key == "*":
      if prevkey == "L":
        keys[-1] += key
        ok = True
    elif key == " ":
      ok = True # Skip spaces.
    if not ok and errors: raise ValueError(f"Error, unknown channel '{key}'.")
    prevkey = key
  return keys

#####################################
# For inclusion in the Image class. #
#####################################

class MixinImage:
  """To be included in the Image class."""

  def histograms(self, channels = "RGBL", nbins = None, recompute = False):
    """Compute histograms of selected channels of the image.

    The histograms are both returned and embedded in the object as self.hists. Histograms already registered in self.hists
    are not recomputed unless required.

    Args:
      channels (str, optional): A combination of the keys "R" (for red), "G" (for green), "B" (for blue), "V" (for HSV value),
        "S" (for HSV saturation), "L" (for luma), and "L*" (for lightness). For a HSV image, only the histograms of the
        value and saturation can be computed. If it ends with a "+", channels gets appended with the keys already computed
        and stored in self.hists. Default is "RGBL".
      nbins (int, optional): Number of bins within [0, 1] in the histograms. Set to `equimage.params.maxhistbins` if negative,
        and computed from the image statistics using Scott's rule if zero. If None, defaults to `equimage.params.defhistbins`.
      recompute (bool, optional): If False (default), the histograms already registered in self.hists are not recomputed
        (provided they match nbins). If True, all histograms are recomputed.

    Returns:
      dict: hists[key] for key in channels, with:

        - hists[key].name = channel name ("Red", "Green", "Blue", "Value", "Saturation", "Luma" or "Lightness", provided for convenience).
        - hists[key].nbins = number of bins within [0, 1].
        - hists[key].edges = histogram bins edges.
        - hists[key].counts = histogram bins counts.
        - hists[key].color = suggested line color for plots.
    """
    if nbins is None: nbins = params.defhistbins
    if nbins == 0:
      if not recompute and hasattr(self, "hists"): # Retrieve the number of bins from the existing histograms.
        nbins = list(self.hists.values())[0].nbins
      else: # Compute the number of bins using Scott's rule.
        width, height = self.get_size()
        npixels = width*height
        if self.colormodel == "RGB":
          channel = (self.image[0]+self.image[1]+self.image[2])/3.
        elif self.colormodel == "HSV":
          channel = self.image[2]
        elif self.colormodel == "gray":
          channel = self.image[0]
        else:
          self.color_model_error()
        nbins = int(np.ceil(npixels**(1./3.)/(3.5*np.std(channel))))
    elif nbins < 0:
      nbins = params.maxhistbins
    nbins = min(max(nbins, 16), params.maxhistbins)
    if not hasattr(self, "hists"): self.hists = {} # Register empty histograms in the object, if none already computed.
    if channels and channels[-1] == "+":
      keys = parse_channels(channels[:-1])
      for key in self.hists.keys(): # Add missing keys.
        if not key in keys: keys.append(key)
    else:
      keys = parse_channels(channels)
    hists = {}
    for key in keys:
      if key in hists: # Already selected.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if not recompute and key in self.hists: # Already computed.
        if self.hists[key].nbins == nbins:
          hists[key] = self.hists[key]
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
      elif key == "L*":
        name = "Lightness"
        color = "lightsteelblue"
        channel = self.lightness()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      mmin = np.floor(np.min(channel)*nbins)
      mmax = np.ceil (np.max(channel)*nbins)
      mbins = int(round(mmax-mmin))
      hists[key] = helpers.Container()
      hists[key].name = name
      hists[key].nbins = nbins
      hists[key].counts, hists[key].edges = np.histogram(channel, bins = mbins, range = (mmin/nbins, mmax/nbins), density = False)
      hists[key].color = color
    self.hists = hists
    return hists

  def statistics(self, channels = "RGBL", exclude01 = None, recompute = False):
    """Compute statistics of selected channels of the image.

    The statistics are both returned and embedded in the object as self.stats. Statistics already registered in self.stats are
    not recomputed unless required.

    Args:
      channels (str, optional): A combination of the keys "R" (for red), "G" (for green), "B" (for blue), "V" (for HSV value),
        "S" (for HSV saturation), "L" (for luma), and "L*" (for lightness). For a HSV image, only the histograms of the
        value and saturation can be computed. If it ends with a "+", channels gets appended with the keys already computed
        and stored in self.hists. Default is "RGBL".
      exclude01 (bool, optional): If True, exclude pixels <= 0 or >= 1 from the median and percentiles.
        Defaults to `equimage.params.exclude01` if None.
      recompute (bool, optional): If False (default), the statistics already registered in self.stats are not recomputed.
        If True, all statistics are recomputed.

    Returns:
      dict: stats[key] for key in channels, with:

        - stats[key].name = channel name ("Red", "Green", "Blue", "Value", "Saturation", "Luma" or "Lightness", provided for convenience).
        - stats[key].width = image width (provided for convenience).
        - stats[key].height = image height (provided for convenience).
        - stats[key].npixels = number of image pixels = image width*image height (provided for convenience).
        - stats[key].minimum = minimum level.
        - stats[key].maximum = maximum level.
        - stats[key].percentiles = (pr25, pr50, pr75) = the 25th, 50th and 75th percentiles.
        - stats[key].median = pr50 = median level.
        - stats[key].zerocount = number of pixels <= 0.
        - stats[key].outcount = number of pixels > 1 (out-of-range).
        - stats[key].exclude01 = True if pixels >= 0 or <= 1 have been excluded from the median and percentiles, False otherwise.
        - stats[key].color = suggested text color for display.
    """
    epsilon = helpers.fpaccuracy(self.dtype)
    if exclude01 is None: exclude01 = params.exclude01
    if not hasattr(self, "stats"): self.stats = {} # Register empty statistics in the object, if none already computed.
    if channels and channels[-1] == "+":
      keys = parse_channels(channels[:-1])
      for key in self.hists.keys(): # Add missing keys.
        if not key in keys: keys.append(key)
    else:
      keys = parse_channels(channels)
    width, height = self.get_size()
    npixels = width*height
    stats = {}
    for key in keys:
      if key in stats: # Already selected.
        print(f"Warning, channel '{key}' selected twice or more...")
        continue
      if not recompute and key in self.stats: # Already computed.
        if self.stats[key].exclude01 == exclude01:
          stats[key] = self.stats[key]
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
      elif key == "L*":
        name = "Lightness"
        color = "lightsteelblue"
        channel = self.lightness()
      else:
        raise ValueError(f"Error, unknown channel '{key}'.")
      stats[key] = helpers.Container()
      stats[key].name = name
      stats[key].width = width
      stats[key].height = height
      stats[key].npixels = npixels
      stats[key].minimum = np.min(channel)
      stats[key].maximum = np.max(channel)
      if exclude01:
        mask = (channel >= epsilon) & (channel <= 1.-epsilon)
        stats[key].percentiles = np.percentile(channel[mask], [25., 50., 75.]) if np.any(mask) else None
      else:
        stats[key].percentiles = np.percentile(channel, [25., 50., 75.])
      stats[key].median = stats[key].percentiles[1] if stats[key].percentiles is not None else None
      stats[key].zerocount = np.sum(channel < epsilon)
      stats[key].outcount = np.sum(channel > 1.+epsilon)
      stats[key].exclude01 = exclude01
      stats[key].color = color
    self.stats = stats
    return stats
