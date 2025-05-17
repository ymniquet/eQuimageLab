# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from equimagelab import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "eQuimageLab"
copyright = "2024, Yann-Michel Niquet"
author = "Yann-Michel Niquet"
version = __version__
release = version

# -- Path to sources ---------------------------------------------------------

sys.path.insert(0, os.path.abspath("../src"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.autosummary",
              "nbsphinx", "sphinx_sitemap", "sphinxcontrib.googleanalytics"]

html_show_sourcelink = False
autodoc_member_order = "bysource"
autodoc_default_options = {"ignore-module-all": True}
nbsphinx_execute = "auto"

html_baseurl = "https://astro.ymniquet.fr/codes/equimagelab/docs/"
sitemap_url_scheme = "{link}"
sitemap_excludes = ["equimagelab_launcher.html", "modules.html",
                    "search.html", "genindex.html", "py-modindex.html"]

googleanalytics_id = "GTM-TBRJL3X9"
googleanalytics_enabled = False

templates_path = ["_templates"]
exclude_patterns = ["notebooks/dashboard_layouts.ipynb"]

# -- Include __init__ in the documentation -----------------------------------

def skip(app, what, name, obj, would_skip, options):
  if name == "__init__":
    return False
  return would_skip

def setup(app):
  app.connect("autodoc-skip-member", skip)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
