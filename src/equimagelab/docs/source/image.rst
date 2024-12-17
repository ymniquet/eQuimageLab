The Image class
---------------

Image I/O
^^^^^^^^^

Images can be read from/saved on disk with the commands:

.. currentmodule:: equimagelab.equimage.image_io

.. autosummary::

   load_image
   save_image

eQuimageLab can handle png, tiff and fits files.

The image in an :py:class:`Image <equimagelab.equimage.image.Image>` object can also be saved with the method :py:class:`Image.save <equimagelab.equimage.image_io.MixinImage.save>`.

Histograms and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

The histograms and statistics of an image can be computed with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stats.MixinImage

.. autosummary::

   histograms
   statistics

They can be displayed in JupyterLab cells or on the dashboard with the relevant commands (see :doc:`firststeps` and :doc:`dashboard`).
