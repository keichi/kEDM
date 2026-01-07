"""
Python bindings for kEDM
------------------------

.. currentmodule:: kedm

.. autosummary::
   :toctree: _generate

   ccm
   edim
   simplex
   eval_simplex
   smap
   eval_smap
   xmap
   get_kokkos_config
"""

import os

if "OMP_PROC_BIND" not in os.environ:
    os.environ["OMP_PROC_BIND"] = "close"

from ._kedm import *

__version__ = "0.10.0"
__version_info__ = (0, 10, 0)
