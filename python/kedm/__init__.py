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

from ._kedm import *

__version__ = "0.9.3"
__version_info__ = (0, 9, 3)
