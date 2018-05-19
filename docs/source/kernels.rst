.. _kernels:

Kernels
========


Create kernels
---------------
New kernels can be defined by creating a closure object where the inner function takes
the spatial and/or temporal frequencies as argument, depending on whether it is a spatial,
temporal, or spatiotemporal kernel.

An example is shown below:

  .. code-block:: python

       def create_my_kernel():
           def evaluate(w, kx, ky):
               # implementation
           return evaluate


Available kernels
-----------------

Spatial
'''''''
* Gaussian
* Difference of Gaussian
* Dirac delta

.. automodule:: pylgn.kernels.spatial
   :members:
   :undoc-members:


Temporal
''''''''
* Dirac delta
* Biphasic
* Difference of exponentials


.. automodule:: pylgn.kernels.temporal
   :members:
   :undoc-members:
