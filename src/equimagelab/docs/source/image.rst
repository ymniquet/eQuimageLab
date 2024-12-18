The Image class
---------------

TODO: Document the params modules


Description
^^^^^^^^^^^

An image is stored in an :py:class:`Image <equimagelab.equimage.image.Image>` object as ``Image.image``, a :py:class:`numpy.ndarray` with data type :py:class:`numpy.float32`:

  - Color images are represented as arrays with shape (3, H, W), where W is the width and H the height of the image in pixels. The leading axis spans the color channels (see :ref:`Color spaces and models` below).
  - Grayscale images are represented as arrays with shape (1, H, W).

.. note::
  This convention differs from the usual representation of images as arrays with shape (H, W, 3) or (H, W). It is, however, more convenient to represent the image as such a superposition of "layers" or "channels" for many operations. See the discussion about the :py:meth:`Image.get_image <equimagelab.equimage.image.Image.get_image>` method below.

Color spaces and models
^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`Image <equimagelab.equimage.image.Image>` class embeds ``Image.colorspace`` and ``Image.colormodel`` attributes for the color space and model of the image. The ``colorspace`` attribute can be:

  - "lRGB" for the linear RGB color space.
  - "sRGB" for the `sRGB <https://en.wikipedia.org/wiki/SRGB>`_ color space.

The ``colormodel`` attribute can be:

  - "gray": grayscale image with one single channel within [0, 1].
  - "RGB": the 3 channels of the image are the red, blue, and green (RGB) levels within [0, 1].
  - "HSV": the 3 channels of the image are the `hue, value, and saturation <https://en.wikipedia.org/wiki/HSL_and_HSV>`_ within [0, 1].

The default color space of an image is "sRGB" and the default color model is "RGB".

.. note::

  In the lRGB color space, the luminance is simply proportional to the RGB values of a pixel. This is usually the color space of your raw images, because cameras are, in principle, rather linear light detectors.

  In the (non-linear) sRGB color space - which is the default color space of most screens - the luminance is *not* proportional to the RGB values of a pixel. The sRGB color space is indeed adapted to our eyes and brain that behave as non-linear light detectors. Therefore, the lRGB color space does not appear linear to the human eye (namely, a pixel with a value of 1 does not appear ten times brighter than a pixel with a value of 0.1), whereas the sRGB color space does (approximately) so.

  In principal, an original linear RGB image shall be converted into a sRGB image before any transformation (with, e.g., the :py:meth:`Image.sRGB <equimagelab.equimage.image_colorspaces.MixinImage.sRGB>` method). As this amounts (roughly) to a power law stretch, which will ultimately be mimicked by the midtone or generalized hyperbolic stretches applied later, this conversion is often dropped out, and the lRGB image is direcly "imported" in the sRGB color space of your screen. eQuimageLab does so by default, as all files are loaded as "sRGB" images irrespective of their actual color profile (see :ref:`Image I/O` below). This can, however, change the color balance of your image. At present, you can make a "clean" conversion of a lRGB into a sRGB image by resetting ``Image.colormodel`` to "lRGB" after reading the file and calling the :py:meth:`Image.sRGB <equimagelab.equimage.image_colorspaces.MixinImage.sRGB>` method.

.. note::

  The HSV color model is best suited for many color transformations (color saturation, etc...). Some operations can, however, not be applied to HSV images. eQuimageLab does not convert automatically HSV into RGB then back to HSV to apply such operations. You need to to it yourself with the :py:meth:`Image.RGB <equimagelab.equimage.image_colorspaces.MixinImage.RGB>` and :py:meth:`Image.HSV <equimagelab.equimage.image_colorspaces.MixinImage.HSV>` methods. HSV images can not, moreover, be displayed in JupyterLab cells or on the dashboard (you need, again, to convert them to RGB to do so).


Accessing the image
^^^^^^^^^^^^^^^^^^^

We recommend - a usual "good practice" in object-oriented applications - that you do not access directly ``Image.image`` (which could break your notebooks if the application interface changes). Use instead the method :py:meth:`Image <equimagelab.equimage.image.Image.get_image>`, which can return the image data in any of the two representations.

+ This is a view...

Image I/O
^^^^^^^^^

Images can be loaded from/saved on disk with the commands:

.. currentmodule:: equimagelab.equimage.image_io

.. autosummary::

   load_image
   save_image

eQuimageLab can handle png, tiff and fits files. As discussed above, they are (for now) all loaded and saved as sRGB images with a RGB or grayscale color model.

The image in an :py:class:`Image <equimagelab.equimage.image.Image>` object can also be saved with the method :py:class:`Image.save <equimagelab.equimage.image_io.MixinImage.save>`.

Histograms and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

The histograms and statistics of an image can be computed with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stats.MixinImage

.. autosummary::

   histograms
   statistics

They can be displayed in JupyterLab cells or on the dashboard with the relevant commands (see :doc:`firststeps` and :doc:`dashboard`).
