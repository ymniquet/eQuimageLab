# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Launcher for eQuimageLab."""

import os
import inspect
import tkinter
import subprocess

def run():
  splash = tkinter.Tk()
  canvas = Canvas(canvas, width = 800, height = 600)
  canvas.pack()
  packagepath = os.path.dirname(inspect.getabsfile(inspect.currentframe()))
  image = tkinter.PhotoImage(file = os.path.join(packagepath, "..", "images", "splash.png"))
  canvas.create_image(0, 0, anchor = tkinter.NW, image = image)
  mainloop = tkinter.mainloop()

  #subprocess.run(["jupyter", "lab"])
