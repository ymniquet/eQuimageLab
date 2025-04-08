First steps with eQuimageLab
----------------------------

Have a look at this minimal `notebook <notebooks/minimal.ipynb>`_: It imports and sets up eQuimageLab, then loads and displays a png image.

This minimal notebook

  - ``import equimagelab as eqlab``: imports eQuimageLab as `eqlab`. The cell outputs a link to this documentation and the definition of the *luma* (a weighted average of the RGB components of an image, see :doc:`composite`).
  - :py:meth:`eqlab.params.set_figure_max_width(768) <equimagelab.jupyter.params.set_figure_max_width>`: sets the maximal display width to 768 pixels (default is 1024 pixels). Adjust this width to work comfortably on your screen; this only affects image display, not image processing.
  - :py:meth:`eqlab.params.set_image_sampling(1) <equimagelab.jupyter.params.set_image_sampling>`: sets image sampling to 1 (which is anyway the default). Only one every `sampling` rows and columns of the images are shown to speed up display. Set `sampling` > 1 if  you deal with very large images; this only affects image display, not image processing.
  - :py:meth:`original, meta = eqlab.load_image("NGC6888.png") <equimagelab.equimage.image_io.load_image>`: loads the png file "NGC6888.png". This returns an :py:class:`Image <equimagelab.equimage.image.Image>` object (the image container of eQuimageLab, see :doc:`image`) and a dictionary of meta-data of the image (if any). Images are represented as numpy arrays of floats within [0, 1] - namely, this RGB image with width 2400 pixels and height 1800 pixels is stored as the array ``original.image`` with shape (3, 1800, 2400) and the RGB channels as leading axis.
  - :py:meth:`eqlab.show(original, histograms = True, statistics = True) <equimagelab.jupyter.backend_plotly.show>`: displays the `original` image with its histograms and statistics [minimum, 25%, 50% (median) and 75% percentiles, maximum, number of shadowed (≤ 0) and highlighted (≥ 1) pixels in each channel]. You can zoom in the images and histograms with the mouse (double click to unzoom). You can also click on the *lin/log* button of the histograms to switch between linear and logarithmic count axis.

eQuimageLab uses the `Plotly <https://plotly.com/>`_ package to display figures in JupyterLab output cells. See the function :py:meth:`equimagelab.show() <equimagelab.jupyter.backend_plotly.show>` for more details; you can display histograms and statistics separately with :py:meth:`equimagelab.show_histograms() <equimagelab.jupyter.backend_plotly.show_histograms>` and :py:meth:`equimagelab.show_statistics() <equimagelab.jupyter.backend_plotly.show_statistics>`. If your image results from a histogram transformation (e.g., histogram stretch), you can display the input and output histograms as well as the transformation curve with :py:meth:`equimagelab.show_t() <equimagelab.jupyter.backend_plotly.show_t>`.

.. hint::

  Note that the notebooks are displayed with a "light" theme in this documentation, but that eQuimageLab is designed to run with a "dark" theme (better adapted to astronomical images). You can switch to the "JupyterLab Dark" theme in the Settings/Theme menu of JupyerLab.
