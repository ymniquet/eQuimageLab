  def clip_channels(self, f, channels):
    """Clip selected 'channels' of the image in the [0, 1] range.
       The 'channels' can be:
         - An empty string, "L", "Lp": Clip all channels.
         - "V": Clip all channels (RGB images) or the value (HSV images).
         - "S": Clip all channels (RGB images) or the saturation (HSV images).
         - A combination of "R", "G", "B": Clip the R/G/B channels (RGB images)."""
    if channels in ["", "L", "Lp"]:
      return self.clip()
    elif channels == "V":
      if self.colormodel == "RGB":
        return self.clip()
      elif self.colormodel == "HSV":
        hsv_image = self.copy()
        hsv_image[2] = utils.clip(self[2])
        return hsv_image
      else:
        self.color_model_error()
    elif channels == "S":
      if self.colormodel == "RGB":
        return self.clip()
      elif self.colormodel == "HSV":
        hsv_image = self.copy()
        hsv_image[1] = utils.clip(self[1])
        return hsv_image
      else:
        self.color_model_error()
    else:
      selected = [False, False, False]
      for c in channels:
        if c == "R":
          ic = 0
        elif c == "G":
          ic = 1
        elif c == "B":
          ic = 2
        else:
          raise ValueError(f"Error, unknown or incompatible channel '{c}'.")
        selected[ic] = True
      self.check_color_model("RGB")
      if all(selected):
        return self.clip()
      else:
        output = self.empty()
        for ic in range(3):
          if selected[ic]:
            output[ic] = utils.clip(self[ic])
          else:
            output[ic] = self[ic]
        return output

