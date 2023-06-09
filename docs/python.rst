.. automodule:: kedm

.. warning::

   Due to a bug in libgomp (GCC's OpenMP runtime), kEDM hangs in a forked
   process. This includes calling kEDM from a ``multiprocessing.Pool`` started
   in `fork` mode (the default). A workaround is to use `spawn` mode instead.
