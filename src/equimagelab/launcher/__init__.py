# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.10.01

"""Launcher for eQuimageLab."""

import os
import sys
import shutil
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import subprocess
from pathlib import Path
from PIL import Image, ImageTk

def run():
  """Run eQuimageLab."""

  def open_notebook():
    """Select and open a jupyter notebook."""
    # Open file selection dialog.
    notebook = tkinter.filedialog.askopenfilename(title = "Open notebook", filetypes = [("Jupyter notebooks", "*.ipynb")], initialdir = Path.home())
    if notebook == "": return
    # Run jupyter lab.
    run_jupyter_lab(notebook)

  def create_new_notebook():
    """Create and open a new jupyter notebook."""
    # Open file selection dialog.
    notebook = tkinter.filedialog.asksaveasfilename(title = "Save new notebook as", filetypes = [("Jupyter notebooks", "*.ipynb")],
                                                    initialdir = Path.home(), initialfile = "eqlab.ipynb", defaultextension = ".ipynb")
    if notebook == "": return
    # Create notebook.
    try:
      shutil.copyfile(os.path.join(__packagepath__, "equimagelab.ipynb"), notebook)
    except Exception as err:
      tkinter.messagebox.showerror("Error", f"Failed to create notebook {notebook}:\n{str(err)}")
      return
    # Run jupyter lab.
    run_jupyter_lab(notebook)

  def run_jupyter_lab(notebook):
    """Run jupyter lab on input notebook."""
    try:
      subprocess.Popen(["jupyter", "lab", notebook])
    except Exception as err:
      tkinter.messagebox.showerror("Error", f"Failed to run jupyter lab:\n{str(err)}")
      return
    quit()

  def quit():
    """Quit launcher."""
    root.destroy()
    sys.exit(0)

  from .. import __packagepath__
  # Open Tk window.
  root = tkinter.Tk()
  root.title("eQuimageLab")
  canvas = tkinter.Canvas(root, width = 800, height = 600)
  canvas.pack(side = "top")
  image = Image.open(os.path.join(__packagepath__, "images", "splash.png")).resize((800, 600))
  imagetk = ImageTk.PhotoImage(image)
  canvas.create_image(0, 0, anchor = "nw", image = imagetk)
  menu = tkinter.Menu(root)
  menu.add_command(label = "Open notebook", command = open_notebook)
  menu.add_command(label = "New notebook", command = create_new_notebook)
  menu.add_command(label = "Quit", command = quit)
  root.config(menu = menu)
  root.mainloop()
