The dashboard
-------------

You can also display images in a separate dashboard.

Have a look at this `notebook <notebooks/dashboard.ipynb>`_: This is the same as the previous one, except for the ``dbrd = eqlab.Dashboard()`` line in the first cell that starts the dashboard. This dashboard can be opened in a separate tab by clicking on the link provided in the output of the cell. The image is now displayed on the dashboard with the ``dbrd.show(original, histograms = True, statistics = True)`` instruction in the last cell.

.. hint::

  The dashboard refreshes automatically. If it stops refreshing, click the "reload" button of your browser.

...


.. warning::

  The dashboard is managed by a Dash/Flask application running in background and serving data on port 8050. There can only be on such server on port 8050.
