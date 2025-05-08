Processing images with eQuimageLab
----------------------------------

This section provides an overview of the capabilities of eQuimageLab.

Image geometry
^^^^^^^^^^^^^^

The geometry of the image (size, orientation, ...) can be transformed with the following methods of the :py:class:`Image <equimage.image.Image>` class:

.. currentmodule:: equimage.image_geometry.MixinImage

.. autosummary::

   flipud
   fliplr
   rot90
   resize
   rescale
   crop

Colors
^^^^^^

The following methods of the :py:class:`Image <equimage.image.Image>` class transform the colors of an image:

.. currentmodule:: equimage.image_colors.MixinImage

.. autosummary::

   negative
   grayscale
   neutralize_background
   color_balance
   match_RGB
   mix_RGB
   color_temperature
   HSX_color_saturation
   CIE_chroma_saturation
   rotate_HSX_hue
   rotate_CIE_hue
   SCNR

Also,the following function returns a HSV wheel as an :py:class:`Image <equimage.image.Image>` object to test color transformations:

.. currentmodule:: equimage.image_colors

.. autosummary::

   HSV_wheel

Histogram stretching
^^^^^^^^^^^^^^^^^^^^

The image histograms can be stretched with the following methods of the :py:class:`Image <equimage.image.Image>` class:

.. currentmodule:: equimage.image_stretch.MixinImage

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
   spline_stretch

Additionally, the image histograms can be stretched with the following functions, which can be applied either to an :py:class:`Image <equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimage.image_stretch

.. autosummary::

   hms
   mts
   ghs

The following function provides an alternative parametrization of the harmonic stretch [based on target (input, output) levels instead of a stretch factor]:

.. autosummary::

   Dharmonic_through

Moreover, eQuimageLab provides an implementation of the `statistical stretch of SETI astro <https://www.setiastro.com/statistical-stretch>`_ as a method of the :py:class:`Image <equimage.image.Image>` class:

.. currentmodule:: equimage.image_stretch.MixinImage

.. autosummary::

   statistical_stretch

Finally, eQuimageLab provides an interface to the `Scikit-Image <https://scikit-image.org/>`_ implementation of `CLAHE <https://en.wikipedia.org/wiki/Adaptive_histogram_equalization>`_ as a method of the :py:class:`Image <equimage.image.Image>` class:

.. currentmodule:: equimage.image_skimage.MixinImage

.. autosummary::

   CLAHE

Image filters
^^^^^^^^^^^^^

eQuimageLab provides filters for image convolution, enhancement or noise reduction as methods of the :py:class:`Image <equimage.image.Image>` class. Some of them are interfaces to the `Scikit-Image <https://scikit-image.org/>`_ package.

Convolutions
""""""""""""

From `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimage.image_skimage.MixinImage

.. autosummary::

   gaussian_filter
   butterworth_filter

Noise reduction
"""""""""""""""

From `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimage.image_skimage.MixinImage

.. autosummary::

   estimate_noise
   wavelets_filter
   bilateral_filter
   total_variation
   non_local_means

Image enhancement
"""""""""""""""""

.. currentmodule:: equimage.image_filters.MixinImage

.. autosummary::

   sharpen
   remove_hot_pixels
   remove_hot_cold_pixels
   LDBS

and, from `Scikit-Image <https://scikit-image.org/>`_:

.. currentmodule:: equimage.image_skimage.MixinImage

.. autosummary::

   unsharp_mask

Miscellaneous operations
^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`Image <equimage.image.Image>` class also features the following methods that perform miscellaneous operations (clipping, blending images, ...):

.. currentmodule:: equimage.image_utils.MixinImage

.. autosummary::

   clip
   blend
   scale_pixels

The :py:meth:`scale_pixels` method can, in particular, be used to stretch the image without changing the hues (the R/G/B ratios) - but this may result in some out-of-range RGB components (see :doc:`composite`). The presence of out-of-range pixels can be tested with:

.. autosummary::

   is_out_of_range

Moreover, you can instantiate a new black or empty image with the same size as an :py:class:`Image <equimage.image.Image>` object with the methods:

.. autosummary::

   black
   empty

You may also clip or blend images with the following functions, which can be applied either to an :py:class:`Image <equimage.image.Image>` object or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimage.image_utils

.. autosummary::

   clip
   blend

Image masks
^^^^^^^^^^^

Binary and float masks can be used to apply operations to selected parts of an image. A binary mask is a boolean :py:class:`numpy.ndarray` with the same size as the image, which defines a False/True flag for each pixel. A float mask is a :py:class:`numpy.ndarray` with the same size as the image, which defines in the same spirit a 0/1 coefficient for each pixel (or, more generally, a weight ranging from 0 to 1). For example, given some :py:class:`Image <equimage.image.Image>` object `image` and binary mask `bmask`,

.. code-block:: python

  selection = np.where(bmask, image, 0.)

returns a new :py:class:`Image <equimage.image.Image>` object `selection`, equal to `image` where `bmask` is True and to zero (black) where `bmask` is False. Likewise, given a float mask `fmask` and some transformation function `transform`,

.. code-block:: python

  output = (1-fmask)*image+fmask*transform(image) # Or equivalently output = image.blend(transform(image), fmask)

returns the transformed image where `fmask` is 1, and the original image where `fmask` is 0. The edges of the mask may be smoothed (made vary gradually from 0 to 1) for a soft transition between the original and transformed images.

eQuimageLab can construct "threshold" float and binary masks that are 1 or True wherever some function ``filter(image)`` is greater than a threshold:

.. currentmodule:: equimage.image_masks

.. autosummary::

   threshold_fmask
   threshold_bmask

The input `filtered` argument is a 2D :py:class:`numpy.ndarray` that contains ``filter(image)``. In particular, the :py:class:`Image <equimage.image.Image>` class provides the following useful filters (local average, median, ...):

.. currentmodule:: equimage.image_masks.MixinImage

.. autosummary::

   filter

The function

.. currentmodule:: equimage.image_masks

.. autosummary::

   shape_bmask

and method of the :py:class:`Image <equimage.image.Image>` class

.. currentmodule:: equimage.image_masks.MixinImage

.. autosummary::

   shape_bmask

enable the construction of binary masks with rectangular, elliptic and polygonal shapes. The rectangle/ellipse/polygon can be selected with the mouse on the dashboard, then its coordinates copied to the notebook. See :doc:`dashboard` for details.

A binary mask can be extended/eroded with the function:

.. currentmodule:: equimage.image_masks

.. autosummary::

   extend_bmask

It can be converted into a float mask with the function:

.. autosummary::

   float_mask

A mask can be smoothed with the function:

.. autosummary::

   smooth_mask

Binary masks are converted into float masks for that purpose.

Edition with external softwares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External softwares (Gimp, Siril, ...) can be run from eQuimageLab to perform specialized operations.

The generic method to edit an :py:class:`Image <equimage.image.Image>` object with an external software is :py:meth:`Image.edit_with() <equimage.image_editors.MixinImage.edit_with>`. This method is implemented for `Gimp <https://www.gimp.org/>`_ and `Siril <https://siril.org/>`_:

.. currentmodule:: equimage.image_editors.MixinImage

.. autosummary::

   edit_with_gimp
   edit_with_siril

.. warning::

  The softwares Gimp and Siril must be in the PATH to be run from eQuimageLab.

Star transformations
^^^^^^^^^^^^^^^^^^^^

The stars can be removed from an image with `Starnet++ <https://www.starnetastro.com/>`_ and resynthetized with `Siril <https://siril.org/>`_ using the following methods of the :py:class:`Image <equimage.image.Image>` class:

.. currentmodule:: equimage.image_stars.MixinImage

.. autosummary::

   starnet
   resynthetize_stars_siril
   reduce_stars

.. warning::

  The softwares Starnet++ and Siril must be in the PATH to be run from eQuimageLab.
