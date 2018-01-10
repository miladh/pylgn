.. _quick:


Getting Started
===============

Install
-------

With `Anaconda <http://continuum.io/downloads>`_ or
`Miniconda <http://conda.pydata.org/miniconda.html>`_::

    conda install -c defaults -c conda-forge -c cinpla pylgn


Minimal example
---------------
This example shows a minimal network consisting of a ganglion cell population and a relay cell population.
A space-time separable impulse-response function is assumed for the ganglion cell, with a spatial part modeled as a difference of Gaussian (DoG) function while the temporal part is a delta function.
The connectivity kernel between ganglion cells and relay cells is also assumed to be space-time separable, with a spatial part modeled as a Gaussian function while the temporal part is a delta function.
The stimulus is full-field grating.

The complete code and a step-by-step explanation is given below:

.. code-block:: python

          import pylgn
          import pylgn.kernels.spatial as spl
          import pylgn.kernels.temporal as tpl

          # create network
          network = pylgn.Network()

          # create integrator
          integrator = network.create_integrator(nt=5, nr=7, dt=1, dr=1)

          # create neurons
          ganglion = network.create_ganglion_cell()
          relay = network.create_relay_cell()

          # create kernels
          Krg_r = spl.create_gauss_ft()
          Krg_t = tpl.create_delta_ft()

          # connect neurons
          network.connect(ganglion, relay, (Krg_r, Krg_t))

          # create stimulus
          k_g = integrator.spatial_angular_freqs[3]
          w_g = -integrator.temporal_angular_freqs[1]
          stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=w_g,
                                                                wavenumber=k_g,
                                                                orient=0.0)
          network.set_stimulus(stimulus)

          # compute
          network.compute_response(relay)

          # visulize
          pylgn.plot.animate_cube(relay.response, title="Relay cell response")


Create network
''''''''''''''
First step is to import pyLGN, including the spatial and temporal kernels, and create a network:

.. testcode::


          import pylgn
          import pylgn.kernels.spatial as spl
          import pylgn.kernels.temporal as tpl

          network = pylgn.Network()


Create integrator
'''''''''''''''''
Next we create an integrator with :math:`2^{nt}` and :math:`2^{ns}` spatial and temporal points, respectively. The temporal and spatial resolutions are :code:`dt=1` (ms) and :code:`dr=0.1` (deg), respectively. Note that if units are not given for the resolutions, "ms" and "deg" are used by default.



.. testcode::

          integrator = network.create_integrator(nt=5, nr=7, dt=1, dr=0.1)


Create neurons
''''''''''''''
Cells can be added to the network using :code:`create_<name>_cell()` method:

.. testcode::

          ganglion = network.create_ganglion_cell()
          relay = network.create_relay_cell()

.. note::
    The impulse-response function of ganglion cells can be set in two ways:

    * It can either be given as an argument :code:`kernel` when the neuron object is created using :py:meth:`~pylgn.core.Network.create_ganglion_cell`. If no argument is given, a spatial DoG function and a temporal delta function is used.

    * The second option is to use the :py:meth:`~pylgn.core.Ganglion.set_kernel` method after that the neuron object is created.



The various neuron attributes are stored in a dictionary on the neuron objects:

    .. code-block:: python

              >>> print(ganglion.annotations)
              {'background_response': array(0.0) * 1/s, 'kernel': {'spatial': {'center': {'params': {'A': 1, 'a': array(0.62) * deg}, 'type': 'create_gauss_ft'}, 'surround': {'params': {'A': 0.85, 'a': array(1.26) * deg}, 'type': 'create_gauss_ft'}, 'type': 'create_dog_ft'}, 'temporal': {'params': {'delay': array(0.0) * ms}, 'type': 'create_delta_ft'}}}


Connect neurons
'''''''''''''''
We use a separable kernel between the ganglion cells and relay cells.
The :py:meth:`~pylgn.core.Network.connect` method has the following signature: :code:`connect(source, target, kernel, weight)`, where source and target are the source and target neurons, respectively, kernel is the connectivity kernel, and weight is the connection weight (default is 1).
If a separable kernel is used a tuple consisting of the spatial and temporal part is given as kernel. The order of kernels in the tuple does not matter.

.. testcode::

          Krg_r = spl.create_gauss_ft()
          Krg_t = tpl.create_delta_ft()

          network.connect(ganglion, relay, (Krg_r, Krg_t))

.. note::
    The kernel parameters can be received using:

    .. code-block:: python

            >>> print(pylgn.closure_params(Krg_r))
            {'params': {'A': 1, 'a': array(0.62) * deg}, 'type': 'create_gauss_ft'}


Create stimulus
'''''''''''''''
A full-field grating stimulus has several parameters including angular frequency, spatial frequency, and orientation.
If you want to use the analytical expression for the Fourier transform of the grating stimulus, you have to make sure that the chosen angular frequency and spatial frequencies exists in the temporal and spatial frequencies determined by the number of points and resolutions.
In this example we just take some values from the existing values:


.. testcode::

          k_g = integrator.spatial_angular_freqs[3]
          w_g = integrator.temporal_angular_freqs[1]
          stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=w_g,
                                                                wavenumber=k_g,
                                                                orient=0.0)
          network.set_stimulus(stimulus)

.. note::
    If you wish to use frequencies that does not exist in the grid, numerical integration can be used. In such cases the inverse Fourier transform of the stimulus must be given. Then :code:`network.set_stimulus(stimulus, compute_fft=True)` method can be used to set the stimulus.



Compute response
''''''''''''''''
The lines below computes the response of the relay cells and animate their activity over time:

.. code-block:: python

          network.compute_response(relay)
          pylgn.plot.animate_cube(relay.response)
