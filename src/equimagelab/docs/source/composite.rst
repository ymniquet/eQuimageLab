Composite channels
------------------

The "composite" channels are functions of the RGB components that bring specific informations about the image. This includes the luminance, lightness, luma, value and saturation (the latter two being actually channels of the HSV color model).

See the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class for a complete overview of the available composite channels:

.. currentmodule:: equimagelab.equimage.image_colorspaces.MixinImage

.. autosummary::

   luminance
   lightness
   luma
   value
   saturation

as well as the following functions of eQuimageLab, which can be applied either to an :py:class:`Image <equimagelab.equimage.image.Image>` object (with a RGB color model) or to a :py:class:`numpy.ndarray`:

.. currentmodule:: equimagelab.equimage.image_colorspaces

.. autosummary::

   luma
   value
   saturation

Luminance, lightness, luma and value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our eyes catch the "brightness" or "lightness" of an image more precisely than its colors. Therefore, it can be beneficial, for example, to reduce noise on the lightness agressively, even if this slightly degrades color rendering.

Yet how bright does an image or pixel look ?

This is a complex question because our eyes are not equally sensitive to the red, blue, and green components of the image. This led to the definition of a *perceptual* lightness :math:`L^*`, defined in the **linear RGB color space** as:

.. math::

  L^* = 116{\rm Y}^{1/3}-16\text{ if }{\rm Y} > 0.008856\text{ else }L^* = 903.3{\rm Y}

where Y is the luminance:

.. math::

  {\rm Y} = 0.2126{\rm lR}+0.7152{\rm lG}+0.0722{\rm lB}

and lR, lG, lB are the linear RGB components of the image. The definitions of Y and :math:`L^*` account for the non-linear and non-homogeneous response of the eyes. They highlight, for example, that we are far more sentitive to green than to red and blue light. Note that :math:`L^*` conventionally ranges within [0, 100] instead of [0, 1].

The method :py:meth:`Image.lightness <equimagelab.equimage.image_colorspaces.MixinImage.lightness>` returns the lightness :math:`L^*` of all pixels of an image (in both lRGB and sRGB color spaces; the sRGB components are converted to lRGB in the latter case).

While :math:`L^*` is the best measure of the brightness of a pixel, it is expensive to compute (since our images usually live in the sRGB color space, whereas :math:`L^*` is defined in the lRGB color space). Therefore, alternate, *approximate* measures of the brightness have been introduced:

  - The *luma* of a pixel L = 0.2126R+0.7152G+0.0722B. In the lRGB color space, the luma is the luminance L = Y (but has nothing to do with the lightness !). In the sRGB color space, the luma (that somehow accounts for the non-linear and non-homogeneous response of the eye) is often used as a convenient substitute for the lightness :math:`L\equiv L^*/100` (but is not as accurate). The method :py:meth:`Image.luma <equimagelab.equimage.image_colorspaces.MixinImage.luma>` returns the luma L of all pixels of an image (calculated from the lRGB or sRGB components depending on the color space). Also, the RGB coefficients of the luma can be tweaked with the :py:func:`set_RGB_luma <equimagelab.equimage.params.set_RGB_luma>` function (and inquired with :py:func:`get_RGB_luma <equimagelab.equimage.params.get_RGB_luma>`). Depending on your purposes, it may be more convenient to work with L = (R+G+B)/3.

  - The *value* of a pixel V = max(R, G, B). This is a basic ingredient of the HSV color model, but a really poor measure of the lightness ! The method :py:meth:`Image.value <equimagelab.equimage.image_colorspaces.MixinImage.value>` returns the value V of all pixels of an image (available for both RGB and HSV images).

Saturation
^^^^^^^^^^

The saturation is also an ingredient of the HSV color model. It is defined as S = 1-min(R, G, B)/max(R, G, B), where min(R, G, B) is the minimum RGB component, and max(R, G, B) the maximum RGB component of a pixel.

Therefore, S = 0 for a black, gray or white pixel (RGB = XXX), and S = 1 for pure red (RGB = 100), yellow (RGB = 110), green (RGB = 010), cyan (RGB = 011), blue (RGB = 001), and magenta (RGB = 101) pixels (as well as for all colors interpolating between two successive ones). It is a measure of the "purity" or "strength" of the color. Decreasing the saturation of a pixel makes it look more grayish, while increasing the saturation makes it look more vivid.

Histograms and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

The histograms and statistics of the RGB and composite channels can be computed with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stats.MixinImage

.. autosummary::

   histograms
   statistics

They histograms can be displayed in JupyterLab cells or on the dashboard with the relevant commands (see :doc:`firststeps` and :doc:`dashboard`).

Also see the following functions of eQuimageLab about histograms:

.. currentmodule:: equimagelab

.. autosummary::

   equimage.params.set_max_hist_bins
   equimage.params.set_default_hist_bins

Working with composite channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many operations (e.g., histograms stretching) can be applied separately to the R, G, B channels of a color image (see the generic :py:meth:`Image.apply_channels <equimagelab.equimage.image_colorspaces.MixinImage.apply_channels>` method).
