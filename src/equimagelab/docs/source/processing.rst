Processing images with eQuimageLab
----------------------------------

This section provides an overview of the capabilities of eQuimageLab.

Image geometry
^^^^^^^^^^^^^^

The geometry of the image (size, orientation, ...) can be transformed with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_geometry.MixinImage

.. autosummary::

   flip_height
   flip_width
   resize
   rescale
   crop

Colors
^^^^^^

The following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class transform the colors of an image:

.. currentmodule:: equimagelab.equimage.image_colors.MixinImage

.. autosummary::

   negative
   grayscale
   color_balance
   color_saturation
   SCNR

Histogram stretching
^^^^^^^^^^^^^^^^^^^^

The image histograms can be stretched with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stretch.MixinImage

.. autosummary::

   set_black_point
   clip_shadow_highlight
   set_dynamic_range
   asinh_stretch
   ghyperbolic_stretch
   midtone_stretch
   gamma_stretch
   adjust_midtone_levels

Additionally, the image histograms can be stretched with the following functions, which can be applied either to an :py:class:`Image <equimagelab.equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimagelab.equimage.image_stretch

.. autosummary::

   mts
   ghs

Finally, eQuimageLab provides the following interface to the `Scikit-Image <https://scikit-image.org/>`_ implementation of `CLAHE <https://en.wikipedia.org/wiki/Adaptive_histogram_equalization>`_ as a method of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_skimage.MixinImage

.. autosummary::

   CLAHE

Image filters
^^^^^^^^^^^^^

eQuimageLab provides filters for image convolution, enhancement or noise reduction as methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class. Some of them are interfaces to the `Scikit-Image <https://scikit-image.org/>`_ package.

Convolutions
""""""""""""

From `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimagelab.equimage.image_skimage.MixinImage

.. autosummary::

   gaussian_filter
   butterworth_filter

Image enhancement
"""""""""""""""""

.. currentmodule:: equimagelab.equimage.image_filters.MixinImage

.. autosummary::

   sharpen
   remove_hot_pixels

and, from `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimagelab.equimage.image_skimage.MixinImage

.. autosummary::

   unsharp_mask

Noise reduction
"""""""""""""""

From `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimagelab.equimage.image_skimage.MixinImage

.. autosummary::

   estimate_noise
   wavelets_filter
   bilateral_filter
   total_variation
   non_local_means

Miscellaneous operations
^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`Image <equimagelab.equimage.image.Image>` class also provides the following methods that perform miscellaneous operations (clipping, blending images, ...):

.. currentmodule:: equimagelab.equimage.image_utils.MixinImage

.. autosummary::

   clip
   blend
   scale_pixels

The :py:meth:`scale_pixels` can, in particular, be used to stretch the image without changing the hues (the R/G/B ratios) - but it may result in some out-of-range components. Actually, the :py:class:`Image <equimagelab.equimage.image.Image>` class also provides some useful tests:

.. autosummary::

   is_out_of_range

Also, you can instantiate a new black or empty image with the same size as an :py:class:`Image <equimagelab.equimage.image.Image>` object with the methods:

.. autosummary::

   black
   empty

You may also clip or blend images with the following functions, which can be applied either to an :py:class:`Image <equimagelab.equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimagelab.equimage.image_utils

.. autosummary::

   clip
   blend


Edition with external softwares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External softwares (Gimp, Siril, Starnet++) can be run from eQuimageLab to perform specialized operations.

The generic method to edit an :py:class:`Image <equimagelab.equimage.image.Image>` object with external software is :py:meth:`Image.edit_with <equimagelab.equimage.image_editors.MixinImage.edit_with>`. This method is implemented for `Gimp <https://www.gimp.org/>`_, `Siril <https://siril.org/>`_ and `Starnet++ <https://www.starnetastro.com/>`_:

.. currentmodule:: equimagelab.equimage.image_editors.MixinImage

.. autosummary::

   edit_with_gimp
   edit_with_siril
   starnet
