The Image class
---------------

Description
^^^^^^^^^^^

An image is stored in an :py:class:`Image <equimagelab.equimage.image.Image>` object as ``Image.image``, a :py:class:`numpy.ndarray` with data type :py:class:`numpy.float32` (single precision 32 bits floats) or :py:class:`numpy.float64` (double precision 64 bits floats):

  - Color images are represented as arrays with shape (3, H, W), where W is the width and H the height of the image in pixels. The leading axis spans the color channels (see `colormodel` below).
  - Grayscale images are represented as arrays with shape (1, H, W).

.. note::
  This convention differs from the usual representation of images as arrays with shape (H, W, 3) or (H, W). It is, however, more convenient to represent an image as such a superposition of "layers" or "channels" for many operations.

The :py:class:`Image <equimagelab.equimage.image.Image>` class also embeds ``Image.colorspace`` and ``Image.colormodel`` attributes for the color space and model of the image. The `colorspace` attribute can be:

  - "lRGB" for the linear RGB (lRGB) color space.
  - "sRGB" for the `sRGB <https://en.wikipedia.org/wiki/SRGB>`_ color space.

The `colormodel` attribute can be:

  - "gray": grayscale image with one single channel within [0, 1].
  - "RGB": the 3 channels of the image are the red, blue, and green levels within [0, 1].
  - "HSV": the 3 channels of the image are the `hue, value, and saturation <https://en.wikipedia.org/wiki/HSL_and_HSV>`_ within [0, 1].

The default color space of an image is "sRGB" and the default color model is "RGB". You won't need to worry about the color space and model of your images for most operations. See the section `More about color spaces`_ for details.

.. note::

  The HSV color model is best suited for some color transformations (color saturation, etc...). Many operations can not, however, be applied to HSV images. eQuimageLab does not automatically convert back and forth between HSV and RGB to apply such operations. You need to do it yourself with the :py:meth:`Image.RGB() <equimagelab.equimage.image_colorspaces.MixinImage.RGB>` and :py:meth:`Image.HSV() <equimagelab.equimage.image_colorspaces.MixinImage.HSV>` methods. HSV images can not, moreover, be displayed in JupyterLab cells and on the dashboard (well, they can, but the outcome is fancy, as they are dealt with as RGB images !). You need, again, to convert them into RGB images to do so.

The data type of the images (:py:class:`numpy.float32` or :py:class:`numpy.float64`) can be set with :py:func:`equimagelab.set_image_type() <equimagelab.equimage.params.set_image_type>` and inquired with :py:func:`equimagelab.get_image_type() <equimagelab.equimage.params.get_image_type>`. The :py:class:`numpy.float64` type is more accurate, but doubles the memory footprint of the images and significantly increases computation times. The default image type is :py:class:`numpy.float32`.

.. hint::
  You may design your notebook with the :py:class:`numpy.float32` type, then rerun it with the :py:class:`numpy.float64` type for best accuracy.

Creating and accessing images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An existing image represented as a :py:class:`numpy.ndarray` array `imgarray` can be embedded in an :py:class:`Image <equimagelab.equimage.image.Image>` object using the constructor ``image = Image(imgarray)``. You can specify the channel axis for color images [``image = Image(imgarray, channels = 0)`` if `imgarray` has (default) shape (3, H, W), and ``image = Image(imgarray, channels = -1)`` if `imgarray` has shape (H, W, 3)].

The :py:class:`Image <equimagelab.equimage.image.Image>` class behaves as a :py:class:`numpy.ndarray` for basic and Numpy operations. Namely, you can add, substract, multiply :py:class:`Image <equimagelab.equimage.image.Image>` objects, and apply all Numpy functions:

.. code-block:: ipython3

  image = (image1+image2)/2
  maxRGB = np.max(image, axis = (1, 2)) # Maximum R/G/B levels.
  fancy = np.sin(image) # For fun...

Therefore, you won't need to access the ``Image.image`` data for most purposes. If you need to do so anyway, we recommend that you use the :py:meth:`Image.get_image() <equimagelab.equimage.image.Image.get_image>` method, which returns the data as a :py:class:`numpy.ndarray` with shape (H, W, 3) or (3, H, W) (see the `channels` kwarg).

.. warning::

  By default the :py:meth:`Image.get_image() <equimagelab.equimage.image.Image.get_image>` method returns (if possible) a **view** on the ``Image.image`` data. Therefore, the instructions

  .. code-block:: ipython3

    data = image.get_image()
    data[0] *= 0.9 # Multiply the red channel by 0.9 to adjust color balance.

  modify both the `data` array and the original `image` object. Use ``data = image.get_image(copy = True)`` if you wish a **copy** of the image data.

You can inquire about the size, number of channels, color space and model of an image with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image.Image

.. autosummary::

   get_shape
   get_size
   get_nc
   get_color_space
   get_color_model

You can make a copy of an :py:class:`Image <equimagelab.equimage.image.Image>` object with the method:

.. autosummary::

   copy

Loading and saving images
^^^^^^^^^^^^^^^^^^^^^^^^^

Images can be loaded from/saved on disk with the functions:

.. currentmodule:: equimagelab.equimage.image_io

.. autosummary::

   load_image
   save_image

eQuimageLab can handle png, tiff and fits files.

The image in an :py:class:`Image <equimagelab.equimage.image.Image>` object can also be saved with the method :py:class:`Image.save <equimagelab.equimage.image_io.MixinImage.save>`.

More about color spaces
^^^^^^^^^^^^^^^^^^^^^^^

In the linear RGB (lRGB) color space, the intensity of light is directly proportional to the RGB values of a pixel. This is usually the color space of our raw images, because cameras are, in principle, rather linear light detectors.

However, our eyes and brain behave as a non-linear light detector. We are more sensitive (able to discern contrast) at low than at high intensities. Therefore, the lRGB color space does not look linear to us (namely, a green pixel with a value of 0.5 does not appear five times brighter than a green pixel with a value of 0.1). This complicates the design of computer-generated images.

Therefore, screens, printers, etc... make use of a non-linear color space (where the intensity is not proportional to the RGB values) that appears approximately linear to the human eyes. The "standard" on most screens and on the web is the `sRGB <https://en.wikipedia.org/wiki/SRGB>`_ color space. The relations between the lRGB and sRGB levels define the so-called transfer function or "gamma":

.. math::

  l &= \left(\frac{s+0.055}{1.055}\right)^{2.4}\text{ if }s > 0.04045\text{ else }\frac{s}{12.92}

  s &= 1.055l^{1/2.4}-0.055\text{ if }l > 0.0031308\text{ else }12.92l

where :math:`l \in [0, 1]` is a linear R, G, or B component and :math:`s` is the corresponding sRGB component.

Linear RGB images can be converted into sRGB images with the :py:meth:`Image.sRGB() <equimagelab.equimage.image_colorspaces.MixinImage.sRGB>` method, and sRGB images into lRGB images with the :py:meth:`Image.lRGB() <equimagelab.equimage.image_colorspaces.MixinImage.lRGB>` method.

.. note::

  In principle, a linear RGB image shall be converted into a sRGB image before processing. As this amounts (roughly) to a power law stretch, which would ultimately be lumped with the midtone or hyperbolic stretches applied later, this conversion is often left out, and the lRGB image is direcly "imported" in the sRGB color space of the screen (that is, setting :math:`s\equiv l`). eQuimageLab actually does so, since all files are loaded as sRGB images by default, irrespective of their color profile (see `Loading and saving images`_ above). This can, however, alter the color balance of the image. If you wish to make a proper conversion of a lRGB into a sRGB image, use the :py:meth:`Image.sRGB() <equimagelab.equimage.image_colorspaces.MixinImage.sRGB>` method:

  .. code-block:: ipython3

    lRGBimage, meta = eqlab.load_image("NGC6942.fit", colorspace = "lRGB")
    sRGBimage = lRGBimage.sRGB()

.. warning::

  Linear RGB images are displayed "as is" in JupyterLab cells and on the dashboard, without conversion to the sRGB color space of the screen. If you need af faithful representation of a lRGB image, you must convert it into a sRGB image with the :py:meth:`Image.sRGB() <equimagelab.equimage.image_colorspaces.MixinImage.sRGB>` method before display.
