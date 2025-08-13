Installation
============

Requirements
------------

xamrex requires Python 3.8 or later and the following dependencies:

* xarray >= 2023.1.0
* numpy >= 1.20
* dask[array] >= 2021.1

Install from Source
-------------------

Currently, xamrex is available for installation from source:

.. code-block:: bash

   git clone https://github.com/your-repo/xamrex.git
   cd xamrex
   pip install -e .

Development Installation
------------------------

For development, clone the repository and install in editable mode with test dependencies:

.. code-block:: bash

   git clone https://github.com/your-repo/xamrex.git
   cd xamrex
   pip install -e .

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   python tests/test_xamrex.py

Or to run tests from the project root:

.. code-block:: bash

   python -m tests.test_xamrex

Verification
------------

To verify the installation, try importing xamrex:

.. code-block:: python

   import xamrex
   print(xamrex.__version__)

You should also be able to use the xarray backend:

.. code-block:: python

   import xarray as xr
   # This should work without errors if you have AMReX plotfile data
   # ds = xr.open_dataset('path/to/plotfile', engine='amrex', level=0)
