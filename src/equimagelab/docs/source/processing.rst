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
   set_shadow_highlight
   set_dynamic_range
   harmonic_stretch
   gharmonic_stretch
   midtone_stretch
   midtone_transfer
   garcsinh_stretch
   ghyperbolic_stretch
   gpowerlaw_stretch
   gamma_stretch
   curve_stretch

Additionally, the image histograms can be stretched with the following functions, which can be applied either to an :py:class:`Image <equimagelab.equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimagelab.equimage.image_stretch

.. autosummary::

   hms
   mts
   ghs

The following function provides an alternative parametrization of the harmonic stretch [based on target (input, output) levels instead of a stretch factor]:

.. autosummary::

   Dharmonic_through

Moreover, eQuimageLab provides an implementation of the `statistical stretch of SETI astro <https://www.setiastro.com/statistical-stretch>`_ as a method of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stretch.MixinImage

.. autosummary::

   statistical_stretch

Finally, eQuimageLab provides an interface to the `Scikit-Image <https://scikit-image.org/>`_ implementation of `CLAHE <https://en.wikipedia.org/wiki/Adaptive_histogram_equalization>`_ as a method of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

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
   LDBS

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

The :py:class:`Image <equimagelab.equimage.image.Image>` class also features the following methods that perform miscellaneous operations (clipping, blending images, ...):

.. currentmodule:: equimagelab.equimage.image_utils.MixinImage

.. autosummary::

   clip
   blend
   scale_pixels

The :py:meth:`scale_pixels` method can, in particular, be used to stretch the image without changing the hues (the R/G/B ratios) - but this may result in some out-of-range RGB components (see :doc:`composite`). Actually, the :py:class:`Image <equimagelab.equimage.image.Image>` class also provides the useful test:

.. autosummary::

   is_out_of_range

Moreover, you can instantiate a new black or empty image with the same size as an :py:class:`Image <equimagelab.equimage.image.Image>` object with the methods:

.. autosummary::

   black
   empty

You may also clip or blend images with the following functions, which can be applied either to an :py:class:`Image <equimagelab.equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimagelab.equimage.image_utils

.. autosummary::

   clip
   blend

Image masks
^^^^^^^^^^^

Masks can be used to apply operations to selected parts of an image. In the simplest form, a mask is a 2D :py:class:`numpy.ndarray` with the same width and height as the image, and only 0's and 1's. Then the instruction ``output = (1-mask)*input+mask*transform(input)`` (or equivalently ``output = input.blend(transform(input), mask)``) returns an `output` image which is the transform of the `input` image wherever `mask` is 1, and the original `input` image wherever it is 0. The edges of the mask may be smoothed (vary gradually from 0 to 1) for a soft transition between the original and transformed images.

At present, eQuimageLab can construct "threshold" masks that are 1 whereever some function ``filter(image)`` is greater than a threshold:

.. currentmodule:: equimagelab.equimage.image_masks

.. autosummary::

   threshold_fmask

The input `filtered` argument is a 2D :py:class:`numpy.ndarray` that contains ``filter(image)``. In particular, the :py:class:`Image <equimagelab.equimage.image.Image>` class provides the following useful filters:

.. currentmodule:: equimagelab.equimage.image_masks.MixinImage

.. autosummary::

   filter

See the :doc:`examples` for use cases. "Lasso selection" of regions of interest will come soon.

Edition with external softwares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External softwares (Gimp, Siril, Starnet++, ...) can be run from eQuimageLab to perform specialized operations.

The generic method to edit an :py:class:`Image <equimagelab.equimage.image.Image>` object with an external software is :py:meth:`Image.edit_with() <equimagelab.equimage.image_editors.MixinImage.edit_with>`. This method is implemented for `Gimp <https://www.gimp.org/>`_, `Siril <https://siril.org/>`_ and `Starnet++ <https://www.starnetastro.com/>`_:

.. currentmodule:: equimagelab.equimage.image_editors.MixinImage

.. autosummary::

   edit_with_gimp
   edit_with_siril
   starnet

.. warning::

  The softwares Gimp, Siril and Starnet++ must be in the PATH to be run from eQuimageLab.
