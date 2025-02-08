The channels argument in eQuimageLab
------------------------------------

Many operations in eQuimageLab can be applied to specific channels specified by the `channels` argument (also see :doc:`composite`).
`channels` can be:

  - An empty string: Apply the operation to all channels.
  - A combination of "1", "2", "3" (or equivalently "R", "G", "B" for RGB images): Apply the operation to the first/second/third channel.
  - "H": Apply the operation to the HSV/HSL hue (RGB, HSV and HSL images).
  - "V": Apply the operation to the HSV value (RGB, HSV and grayscale images).
  - "S": Apply the operation to the HSV saturation (RGB and HSV images).
  - "L'": Apply the operation to the HSL lightness (RGB, HSL and grayscale images).
  - "S'": Apply the operation to the HSL saturation (RGB and HSL images).
  - "L": Apply the operation to the luma (RGB and grayscale images).
  - "Ls": Apply the operation to the luma, and protect highlights by desaturation (see below).
  - "Lb": Apply the operation to the luma, and protect highlights by blending (see below).
  - "Ln": Apply the operation to the luma, and protect highlights by normalization (see below).
  - "L*": Apply the operation to the CIE lightness :math:`L^*` (CIELab and CIELuv images; equivalent to "L*ab" for RGB and grayscale images).
  - "L*ab": Apply the operation to the CIE lightness :math:`L^*` in the CIELab color space (CIELab, RGB and grayscale images).
  - "L*uv": Apply the operation to the CIE lightness :math:`L^*` in the CIELuv color space (CIELuv, RGB and grayscale images).

Note on "Ls" and "Lb":
""""""""""""""""""""""

When applying an operation `f` to the luma, the RGB components of the image are rescaled by the ratio `f` (luma)/luma. This preserves the hue and HSV saturation, but may bring some RGB components out-of-range even though `f`\(luma) fits within [0, 1]. These out-of-range components can be regularized with three highlights protection methods:

  - "Desaturation": The out-of-range pixels are desaturated at constant hue and luma (namely, the out-of-range components are decreased while the in-range components are increased so that the hue and luma are preserved). This tends to bleach the out-of-range pixels. `f`\(luma) must fit within [0, 1] to make use of this highlights protection method.
  - "Blending": The out-of-range pixels are blended with `f`\(RGB) (the same operation applied to the RGB channels). This tends to bleach the out-of-range pixels too. `f`\(RGB) must fit within [0, 1] to make use of this highlights protection method.
  - "Normalization": The whole ``output`` image is rescaled so that all pixels fall back in the [0, 1] range (``output`` → ``output/max(1., np.max(output))``).
