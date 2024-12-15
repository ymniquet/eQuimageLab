eQuimageLab user guide
======================

eQuimageLab is a Python package to process astronomical images in `Jupyter Lab <https://jupyter.org/>`_ notebooks.

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

To use eQuimageLab, you can simply start a Jupyter Lab server from a shell:

.. code-block:: bash

  jupyter lab

and import eQuimageLab:

.. code-block:: Python

  import equimagelab as eql

This is, however, little convenient - in particular, because you may want to run Jupyter Lab from a given directory in order to have an easy access to its files. Therefore, the eQuimageLab package includes a launcher with a graphical user interface that helps you start Jupyter Lab. You can run this launcher from a shell:

.. code-block:: bash

  equimagelab

or add and configure this `icon <https://astro.ymniquet.fr/codes/equimagelab/icons/icon.ico>`_ on your desktop.

The launcher menu provides four options:

  - `New notebook`: Create a new Jupyter Lab notebook from a default template. The launcher asks for the directory and name of this new notenook, and starts Jupyter Lab from this directory. The default template contains minimal code to import and set up eQuimageLab.
  - `Open notebook`: Open an existing Jupyter Lab notebook. The launcher asks for the directory and name of this notenook, and starts Jupyter Lab from this directory.
  - `Open directory`: Start Jupyter Lab from a given directory (with an empty notebook).
  - `Quit`: Quit the launcher.

First steps with eQuimageLab
----------------------------

...


