Composite channels
------------------

The "composite" channels are mixtures of the RGB components that bring specific informations about the image. This includes the luminance, lightness, luma, value and saturation (the latter two being actually borrowed from the HSV color model).

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

Our eyes catch the "brightness" or "lightness" of an image more accurately than its colors. Therefore, it can be beneficial, for example, to reduce noise on the lightness agressively even if this slightly degrades color rendering.

Yet how bright does an image or pixel look ?

This is a complex question because our eyes are not equally sensitive to the red, blue, and green components of the image. This led to the definition of a *perceptual* lightness :math:`L^*`, defined in the **linear RGB color space** as:

.. math::

  L^* = 116{\rm Y}^{1/3}-16\text{ if }{\rm Y} > 0.008856\text{ else }L^* = 903.3{\rm Y}

where Y is the luminance:

.. math::

  {\rm Y} = 0.2126{\rm lR}+0.7152{\rm lG}+0.0722{\rm lB}

and lR, lG, lB are the linear RGB components of the image. The definitions of Y and :math:`L^*` account for the non-linear and non-homogeneous response of the eyes. They highlight, for example, that we are far more sentitive to green than to red and blue light. Note that :math:`L^*` conventionally ranges within [0, 100] instead of [0, 1].

The method :py:meth:`Image.lightness() <equimagelab.equimage.image_colorspaces.MixinImage.lightness>` returns the lightness :math:`L^*` of all pixels of a lRGB or sRGB image (the sRGB components being converted to lRGB for that purpose).

While :math:`L^*` is the best measure of the brightness of a pixel, it is expensive to compute (since our images usually live in the sRGB color space, whereas :math:`L^*` is defined in the lRGB color space). Therefore, alternate, *approximate* measures of the brightness have been introduced:

  - The *luma* of a pixel L = 0.2126R+0.7152G+0.0722B. In the lRGB color space, the luma is the luminance L = Y (but has nothing to do with the lightness !). In the sRGB color space, the luma (which somehow accounts for the non-linear and non-homogeneous response of the eye) is often used as a convenient substitute for the lightness :math:`L\equiv L^*/100` (but is not as accurate). The method :py:meth:`Image.luma() <equimagelab.equimage.image_colorspaces.MixinImage.luma>` returns the luma L of all pixels of an image (calculated from the lRGB or sRGB components depending on the color space). Also, the RGB coefficients of the luma can be tweaked with the :py:func:`set_RGB_luma() <equimagelab.equimage.params.set_RGB_luma>` function (and inquired with :py:func:`get_RGB_luma() <equimagelab.equimage.params.get_RGB_luma>`). Depending on your purposes, it may be more convenient to work with L = (R+G+B)/3.

  - The *value* of a pixel V = max(R, G, B). This is a key component of the HSV color model, but a really poor measure of the lightness ! The method :py:meth:`Image.value() <equimagelab.equimage.image_colorspaces.MixinImage.value>` returns the value V of all pixels of an image (available for both RGB and HSV images).

Saturation
^^^^^^^^^^

The saturation is also a component of the HSV color model. It is defined as S = 1-min(R, G, B)/max(R, G, B), where min(R, G, B) is the minimum RGB component, and max(R, G, B) the maximum RGB component of a pixel.

Therefore, S = 0 for a black, gray or white pixel (RGB = XXX), and S = 1 for pure red (RGB = 100), yellow (RGB = 110), green (RGB = 010), cyan (RGB = 011), blue (RGB = 001), and magenta (RGB = 101) pixels (as well as for all colors interpolating between two successive ones). This is best shown on the "HSV wheel of colors" below, where saturation increases from the center (S = 0) to the edges (S = 1).

.. figure:: images/HSVwheel.png
   :figwidth: 40%
   :width: 100%
   :align: center
   :alt: The HSV wheel

   The "HSV wheel" of colors. Saturation increases from the center to the edge of the wheel.

The saturation is, therefore, a measure of the "purity" or "strength" of the color. Decreasing the saturation of a pixel makes it look more grayish, while increasing the saturation makes it look more vivid.

Histograms and statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

The histograms and statistics of the RGB and composite channels can be computed with the following methods of the :py:class:`Image <equimagelab.equimage.image.Image>` class:

.. currentmodule:: equimagelab.equimage.image_stats.MixinImage

.. autosummary::

   histograms
   statistics

The histograms can be displayed in JupyterLab cells or on the dashboard with the relevant commands (see :doc:`firststeps` and :doc:`dashboard`).

Also see the following functions of eQuimageLab about histograms:

.. currentmodule:: equimagelab

.. autosummary::

   equimage.params.set_max_hist_bins
   equimage.params.set_default_hist_bins

Working with composite channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations (e.g., histograms stretches) can be applied separately to the R, G, B channels of a color image, or to a composite channel (see the generic :py:meth:`Image.apply_channels() <equimagelab.equimage.image_colorspaces.MixinImage.update_channels>` and :py:meth:`Image.apply_channels() <equimagelab.equimage.image_colorspaces.MixinImage.apply_channels>` methods).

**In particular, operations on the value V and luma L of an image are designed to preserve the ratios between the RGB components, hence to preserve the hue and saturation (the "apparent" color).**

Therefore, stretching the luma protects the colors of the image, whereas stretching the RGB components separately usually tends to "bleach" the image. Stretching the value also protects the colors, but tends to mess up the lightness (see :doc:`image` section).

However, acting on the luma L can bring some RGB components out of the [0, 1] range.

Let us take the midtone transformation T(x) = 0.761x/(0.522x+0.239) as an example (see the :py:meth:`Image.midtone_stretch() <equimagelab.equimage.image_stretch.MixinImage.midtone_stretch>` method). The transformation T(x) maps [0, 1] onto [0, 1] and does not, therefore, produce out-of-range pixels when applied to the R, G, B channels separately.

Let us now consider a pixel with components (R = 0.4, G = 0.2, B = 0.6) and luma L = 0.2126R+0.7152G+0.0722B = 0.271. Under transformation T, the luma of this pixel doubles and becomes L' = T(L) = 0.543. Accordingly, the new RGB components of the pixel are (R' = 0.8, G' = 0.4, B' = 1.2). While L' is still within bounds, B' is not.

Such out-of-range pixels are cut-off when displayed or saved in png and tiff files.

There are three options to deal with the out-of-range pixels:

  1. Leave "as is": If you are confident that further processing will bring back these pixels in the [0, 1] range (or are satisfied with the look of the image), you can simply... do nothing about them.

  2. Desaturate at constant luma: decrease the saturation of the out-of-range pixels while keeping the luma constant until all components fall back in the [0, 1] range. This preserves the intent of the stretch (the luma is unchanged) but tends to bleach the out-of-range pixels. In the present case, the transformed pixel becomes (R' = 0.722, G' = 0.443, B' = 1) and the saturation decreases from S' = 0.667 to S' = 0.557.

  3. Blend each out-of-range pixel with (T(R), T(G), T(B)), so that all components fall back in the [0, 1] range. This does preserve neither the luma nor the hue, and also tends to bleach the out-of-range pixels. In the present case, (T(R), T(G), T(B)) = (0.680, 0.443, 0.827) and the transformed pixel becomes (R' = 0.736, G' = 0.423, B' = 1). The output luma is L' = 0.531 and the output saturation is S' = 0.577.

In eQuimageLab, these three options correspond to different choices for the kwarg `channels` of the transformation: 1) `channels` = "L", 2) `channels` = "Ls" and 3) `channels` = "Lb".

.. note::

  eQuimageLab applies operations to the lightness :math:`L^*` in the :math:`L^*a^*b^*` `color space & model <https://en.wikipedia.org/wiki/CIELAB_color_space>`_. This color space & model is only used internally for that purpose, and is not available in the eQuimageLab interface (at present).
