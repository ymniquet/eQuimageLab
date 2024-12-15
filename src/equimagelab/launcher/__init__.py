# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
# Author: Yann-Michel Niquet (contact@ymniquet.fr).
# Version: 1.0.0 / 2024.12.15

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
  """Run eQuimageLab launcher."""

  def new_notebook():
    """Create and open a new jupyter notebook."""
    # Open file selection dialog.
    notebook = tkinter.filedialog.asksaveasfilename(title = "Save new notebook as", filetypes = [("Jupyter notebooks", "*.ipynb")],
                                                    initialdir = Path.home(), initialfile = "eqlab.ipynb", defaultextension = ".ipynb")
    if not notebook: return
    # Create notebook.
    template = os.path.join(__packagepath__, "launcher", "equimagelab.ipynb")
    print(f"Copying {template} as {notebook}...")
    try:
      shutil.copyfile(template, notebook)
    except Exception as err:
      tkinter.messagebox.showerror("Error", f"Failed to create notebook {notebook}:\n{str(err)}")
      return
    # Run jupyter lab.
    run_jupyter_lab(notebook)

  def open_notebook():
    """Select and open a jupyter notebook."""
    # Open file selection dialog.
    notebook = tkinter.filedialog.askopenfilename(title = "Open notebook", filetypes = [("Jupyter notebooks", "*.ipynb")], initialdir = Path.home())
    if not notebook: return
    # Run jupyter lab.
    run_jupyter_lab(notebook)

  def open_directory():
    """Run jupyter lab in a selected directory."""
    # Open directory selection dialog.
    directory = tkinter.filedialog.askdirectory(title = "Open directory", initialdir = Path.home())
    if not directory: return
    # Run jupyter lab.
    run_jupyter_lab(f"--notebook-dir={directory}")

  def run_jupyter_lab(options = ""):
    """Run jupyter lab.

    Args:
      options (str): Options for the jupyter lab command (default "").
    """
    print(f"Running jupyter lab {options}...")
    try:
      subprocess.Popen(["jupyter", "lab", options])
    except Exception as err:
      tkinter.messagebox.showerror("Error", f"Failed to run jupyter lab:\n{str(err)}")
      return
    quit()

  def quit():
    """Quit launcher."""
    root.destroy()
    sys.exit(0)

  from .. import __packagepath__
  # Open root Tk window.
  root = tkinter.Tk()
  root.title("eQuimageLab")
  # Configure menu.
  menu = tkinter.Menu(root)
  menu.add_command(label = "New notebook", command = new_notebook)
  menu.add_command(label = "Open notebook", command = open_notebook)
  menu.add_command(label = "Open directory", command = open_directory)
  menu.add_command(label = "Quit", command = quit)
  root.config(menu = menu)
  # Display splash screen.
  canvas = tkinter.Canvas(root, width = 800, height = 600)
  canvas.pack(side = "top")
  try:
    image = Image.open(os.path.join(__packagepath__, "images", "splash.png")).resize((800, 600))
  except:
    pass
  else:
    imagetk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor = "nw", image = imagetk)
  # Start Tk main loop.
  root.mainloop()
