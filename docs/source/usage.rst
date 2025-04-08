Running eQuimageLab
--------------------

To use eQuimageLab, you can simply run JupyterLab from a shell:

.. code-block:: bash

  jupyter lab

and import eQuimageLab:

.. code-block:: ipython3

  import equimagelab as eql

This is, however, little convenient - in particular, because you may want to start JupyterLab from a given directory in order to have direct access to its content. Therefore, the eQuimageLab package includes a launcher with a graphical user interface that helps you start JupyterLab according to your needs. You can run this launcher from a shell:

.. code-block:: bash

  eQuimageLab

or add and configure this `icon <https://astro.ymniquet.fr/codes/equimagelab/icons/icon.ico>`_ on your desktop.

The launcher menu provides four options:

  - *New notebook*: Create a new JupyterLab notebook from a default template. The launcher asks for the directory and name of this new notenook, and starts JupyterLab from this directory. The default template contains minimal code to import and set up eQuimageLab.
  - *Open notebook*: Open an existing JupyterLab notebook. The launcher asks for the directory and name of this notenook, and starts JupyterLab from this directory. You can also open an existing notebook from the command line with ``eQuimageLab notebook``.
  - *Open directory*: Start JupyterLab from a given directory (with an empty notebook).
  - *Quit*: Quit the launcher.
