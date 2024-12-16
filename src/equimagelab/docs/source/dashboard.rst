The dashboard
-------------

You can also display images in a separate dashboard managed with `Dash <https://dash.plotly.com/>`_.

Have a look at this `notebook <notebooks/dashboard.ipynb>`_. This is the same as the previous one, except for

  - The :py:class:`dbrd = eqlab.Dashboard() <equimagelab.jupyter.backend_dash.Dashboard>` line in the first cell that starts the dashboard. This dashboard can be opened in a separate browser tab by clicking on the link provided in the output of the cell.
  - The :py:meth:`dbrd.show(original, histograms = True, statistics = True) <equimagelab.jupyter.backend_dash.Dashboard.show>` instruction in the last cell that displays the image on the dashboard.

...

.. hint::

  The dashboard refreshes automatically. If it stops refreshing, click the "reload" button of your browser.

.. warning::

  The dashboard is managed by a Dash application running in background and serving data on port 8050. There can only be one application bound to that port; if you get the error message

  *Address already in use. Port 8050 is in use by another program. Either identify and stop that program, or start the server with a different port.*

  another Dash application (from a previous or concurrent eQuimageLab session) may be running on your machine.

  On Linux or Mac OSX, you can get the process ID (<PID>) of the application bound to port 8050 with the shell command ``lsof -i:8050``, and kill it (if you know what you're doing) with ``kill <PID>``. On Windows, you can likewise ``netstat -aon | find "8050"`` and ``taskkill /PID <PID>`` in a command prompt.

  **To avoid such issues, always quit JupyterLab through the File/Shutdown menu in order not to leave a stale Dash application running in background.**


