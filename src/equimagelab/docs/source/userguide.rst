eQuimageLab user guide
======================

eQuimageLab is a Python package to process astronomical images in `JupyterLab <https://jupyter.org/>`_ notebooks.

Installation
------------

eQuimageLab is developed for Python 3 kernels.

This guide assumes that you are familiar with the Python programming language and that Python 3 is installed on your machine.

The eQuimageLab package is available on `PyPI <https://https://pypi.org/project/eQuimageLab/>`_ (last stable release) and on `GitHub <https://github.com/ymniquet/eQuimage>`_ (development version).

To install the latest stable release, open a linux or windows shell and type:

.. code-block:: bash

  pip install --user eQuimageLab

pip will download and install the eQuimageLab package from PyPI, as well as all dependencies (Jupiter Lab, plotly, dash...). If you run Python in a virtual environment, you can remove the --user option.

Running eQuimageLab
--------------------

To use eQuimageLab, you can simply start a JupyterLab server from a shell:

.. code-block:: bash

  jupyter lab

and import eQuimageLab:

.. code-block:: ipython3

  import equimagelab as eql

This is, however, little convenient - in particular, because you may want to run JupyterLab from a given directory in order to have an easy access to its files. Therefore, the eQuimageLab package includes a launcher with a graphical user interface that helps you start JupyterLab. You can run this launcher from a shell:

.. code-block:: bash

  equimagelab

or add and configure this `icon <https://astro.ymniquet.fr/codes/equimagelab/icons/icon.ico>`_ on your desktop.

The launcher menu provides four options:

  - `New notebook`: Create a new JupyterLab notebook from a default template. The launcher asks for the directory and name of this new notenook, and starts JupyterLab from this directory. The default template contains minimal code to import and set up eQuimageLab.
  - `Open notebook`: Open an existing JupyterLab notebook. The launcher asks for the directory and name of this notenook, and starts JupyterLab from this directory.
  - `Open directory`: Start JupyterLab from a given directory (with an empty notebook).
  - `Quit`: Quit the launcher.

First steps with eQuimageLab
----------------------------

Have a look at this minimal `notebook <notebooks/minimal.ipynb>`_: It imports and sets up eQuimageLab, then loads and displays a png image.

Note that the notebook is displayed with a "light" theme in this documentation, but that eQuimageLab is designed to run with a "dark" theme (better adapted to astronomical images). You can switch to the "JupyterLab Dark" theme in the Settings/Theme menu of Jupyer Lab.

This minimal notebook:

  - ``import equimagelab as eqlab``: imports eQuimageLab as ``eqlab``. The cell outputs a link to this documentation and the definition of the `Luma` (a weighted average of the RGB components of an image).
  - ``eqlab.params.set_figure_max_width(768)``: sets the maximal display width to 768 pixels (default is 1024 pixels). Adjust this width to work comfortably on your screen; this only affects image display, not image processing.
  - ``eqlab.params.set_image_sampling(1)``: sets image sampling to 1 (which is anyway the default). Only one every sampling lines and columns of the images are shown to speed up display. Set sampling > 1 if  you deal with very large images; this only affects image display, not image processing.
  - ``original, meta = eqlab.load_image("NGC6888.png")``: loads the png file "NGC6888.png". This returns an :class:`equimagelab.equimage.Image` object (the image container of eQuimageLab) and a dictionary of meta-data of the image (if any). Images are represented as numpy arrays of floats within [0, 1] - namely, this RGB image with width 2400 pixels and height 1800 pixels is stored as the array ``original.image`` with shape (3, 1800, 2400); the RGB channels are the leading axis.
  - ``eqlab.show(original, histograms = True, statistics = True)``: displays the ``original`` image with its histograms and statistics [minimum, 25%, 50% (median) and 75% percentiles, maximum, number of shadowed (<= 0) and highlighted (>= 1) pixels in each channel]. You can click on the lin/log button of the histograms to switch from linear to logarithmic count axis.

See the function :func:`equimagelab.show` for more details about image display. You can display histograms and statistics separately with :func:`equimagelab.show_histograms` and :func:`equimagelab.show_statistics`. If your image results from an histogram transformation (e.g., histogram stretch), you can display the input and output histograms as well as the transformation curve with :func:`equimagelab.show_t`.

The dashboard
-------------

You can also display images in a separate dashboard.

Have a look at this `notebook <notebooks/dashboard.ipynb>`_: This is the same as the previous one, except for the ``dbrd = eqlab.Dashboard()`` line in the first cell that starts the dashboard. This dashboard can be opened in a separate tab by clicking on the link provided in the output of the cell. The image is now displayed on the dashboard with the ``dbrd.show(original, histograms = True, statistics = True)`` instruction in the last cell.

.. hint::

  The dashboard refreshes automatically. If it stops refreshing, click the "reload" button of your browser.

...



