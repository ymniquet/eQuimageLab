First steps with eQuimageLab
----------------------------

Have a look at this minimal `notebook <notebooks/minimal.ipynb>`_: It imports and sets up eQuimageLab, then loads and displays a png image.

.. hint::

  Note that the notebook is displayed with a "light" theme in this documentation, but that eQuimageLab is designed to run with a "dark" theme (better adapted to astronomical images). You can switch to the "JupyterLab Dark" theme in the Settings/Theme menu of Jupyer Lab.

This minimal notebook:

  - ``import equimagelab as eqlab``: imports eQuimageLab as ``eqlab``. The cell outputs a link to this documentation and the definition of the `Luma` (a weighted average of the RGB components of an image).
  - ``eqlab.params.set_figure_max_width(768)``: sets the maximal display width to 768 pixels (default is 1024 pixels). Adjust this width to work comfortably on your screen; this only affects image display, not image processing.
  - ``eqlab.params.set_image_sampling(1)``: sets image sampling to 1 (which is anyway the default). Only one every sampling lines and columns of the images are shown to speed up display. Set sampling > 1 if  you deal with very large images; this only affects image display, not image processing.
  - ``original, meta = eqlab.load_image("NGC6888.png")``: loads the png file "NGC6888.png". This returns an :class:`equimagelab.equimage.Image` object (the image container of eQuimageLab) and a dictionary of meta-data of the image (if any). Images are represented as numpy arrays of floats within [0, 1] - namely, this RGB image with width 2400 pixels and height 1800 pixels is stored as the array ``original.image`` with shape (3, 1800, 2400); the RGB channels are the leading axis.
  - ``eqlab.show(original, histograms = True, statistics = True)``: displays the ``original`` image with its histograms and statistics [minimum, 25%, 50% (median) and 75% percentiles, maximum, number of shadowed (<= 0) and highlighted (>= 1) pixels in each channel]. You can click on the lin/log button of the histograms to switch from linear to logarithmic count axis.

See the function :func:`equimagelab.show` for more details about image display. You can display histograms and statistics separately with :func:`equimagelab.show_histograms` and :func:`equimagelab.show_statistics`. If your image results from an histogram transformation (e.g., histogram stretch), you can display the input and output histograms as well as the transformation curve with :func:`equimagelab.show_t`.
