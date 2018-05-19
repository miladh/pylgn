[![Build Status](https://travis-ci.org/miladh/pylgn.svg?branch=dev)](https://travis-ci.org/miladh/pylgn)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Anaconda-Server Badge](https://anaconda.org/cinpla/pylgn/badges/installer/conda.svg)](https://anaconda.org/cinpla/pylgn)
[![Documentation Status](https://readthedocs.org/projects/pylgn/badge/?version=latest)](http://pylgn.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/miladh/pylgn/branch/dev/graph/badge.svg)](https://codecov.io/gh/miladh/pylgn)



## pyLGN: a simulator of neural activity in the early part of the visual system 

pyLGN is a visual stimulus-driven simulator of spatiotemporal cell responses in the early part of the visual system consisting of the retina, lateral geniculate nucleus (LGN) and primary visual cortex. The simulator is based on a mechanistic, firing rate model that incorporates the influence of thalamocortical loops, in addition to the feedforward responses. The advantage of the simulator lies in its computational and conceptual ease, allowing for fast and comprehensive exploration of various scenarios for the organization of the LGN circuit.

### Table of contents

- [Example](#example)
- [Documentation](#documentation)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Citation](#citation)

### Example
In this example a natural image is used as stimulus to stimulate a layer of (ON-center) ganglion cells. The static image is shown in 80 ms after a 40 ms delay, and the response of the ganglion cells is shown as a heatmap from blue to red (low to high response).

```python
import pylgn
import pylgn.kernels as kernel
import quantities as pq

# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=8, nr=9, dt=1*pq.ms, dr=0.1*pq.deg)

# create kernels
Wg_r = kernel.spatial.create_dog_ft()
Wg_t = kernel.temporal.create_biphasic_ft()

# create neurons
ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

# create stimulus
stimulus = pylgn.stimulus.create_natural_image(filenames="natural_scene.png",
                                               delay=40*pq.ms,
                                               duration=80*pq.ms)
network.set_stimulus(stimulus, compute_fft=True)

# compute
network.compute_response(ganglion)

# visulize
pylgn.plot.animate_cube(ganglion.response,
                        title="Ganglion cell responses",
                        dt=integrator.dt.rescale("ms"))
```

The resulting response is shown below:
<p align="center">
  <img width="384" height="288" src="https://github.com/miladh/pylgn/blob/dev/docs/images/natural_scene.gif">
</p>


### Documentation 
- The documentation for pyLGN can be found at http://pylgn.rtfd.io/.

- Scientific paper: [Mobarhan MH, Halnes G, Martínez-Cañada P, Hafting T, Fyhn M, Einevoll G. (2018). PLOS Computational Biology 14(5): e1006156. https://doi.org/10.1371/journal.pcbi.1006156](https://doi.org/10.1371/journal.pcbi.1006156)


### Installation

pyLGN can easily be installed using [conda](https://www.anaconda.com/download/):

    conda install -c defaults -c conda-forge -c cinpla pylgn

To install a specific version use:

    conda install -c defaults -c conda-forge -c cinpla pylgn=0.91

pyLGN can also be installed by cloning the Github repository:

    $ git clone https://github.com/miladh/pylgn
    $ cd /path/to/pylgn
    $ python setup.py install

Note that [dependencies](#dependencies) must be installed independently. 

### Dependencies

pyLGN has the following dependencies:

- `python >=3.5`
- `matplotlib`
- `numpy`
- `scipy`
- `setuptools`
- `pillow`
- `quantities 0.12.1`

### Citation
If you use pyLGN in your work, please cite:

[Mobarhan MH, Halnes G, Martínez-Cañada P, Hafting T, Fyhn M, Einevoll G. (2018). PLOS Computational Biology 14(5): e1006156. https://doi.org/10.1371/journal.pcbi.1006156](https://doi.org/10.1371/journal.pcbi.1006156)
