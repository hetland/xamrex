Installation
============

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/hetland/xamrex.git
   cd xamrex
   pip install -e .

Requirements
------------

- Python ≥ 3.8
- xarray ≥ 2023.1.0
- numpy ≥ 1.20
- dask ≥ 2021.1
- pandas ≥ 1.5.0

Optional Dependencies
---------------------

For grid-aware operations:

.. code-block:: bash

   pip install xgcm

For plotting:

.. code-block:: bash

   pip install matplotlib

Verification
------------

Test the installation:

.. code-block:: python

   import xarray as xr
   import xamrex
   
   print(xamrex.__version__)
   
   # Check that backend is registered
   ds = xr.open_dataset('path/to/plotfile', engine='amrex', level=0)
