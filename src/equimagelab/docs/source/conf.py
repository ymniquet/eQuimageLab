# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "eQuimageLab"
copyright = "2024, Yann-Michel Niquet"
author = "Yann-Michel Niquet"
release = "1.0.0"

# -- Path to sources ---------------------------------------------------------

sys.path.insert(0, os.path.abspath("../../.."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "nbsphinx"]

autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []

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
