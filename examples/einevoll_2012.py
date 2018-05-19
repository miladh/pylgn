import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels as kernel

# fb weights:
fb_weights = [0, -1.5]

# diameters
patch_diameter = np.linspace(0, 6, 50) * pq.deg
response = np.zeros([len(patch_diameter), len(fb_weights)]) / pq.s

for j, w_c in enumerate(fb_weights):
    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

    # create kernels
    delta_t = kernel.temporal.create_delta_ft()
    delta_s = kernel.spatial.create_delta_ft()
    Wg_r = kernel.spatial.create_dog_ft(A=1, a=0.25*pq.deg, B=0.85, b=0.83*pq.deg)
    Krc_r = kernel.spatial.create_gauss_ft(A=1, a=0.83*pq.deg)

    # create neurons
    ganglion = network.create_ganglion_cell(kernel=(Wg_r, delta_t))
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()

    # connect neurons
    network.connect(ganglion, relay, (delta_s, delta_t), 1.0)
    network.connect(cortical, relay, (Krc_r, delta_t), w_c)
    network.connect(relay, cortical, (delta_s, delta_t), 1.0)

    for i, d in enumerate(patch_diameter):
        # create stimulus
        stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=0,
                                                          patch_diameter=d)
        network.set_stimulus(stimulus)

        # compute
        network.compute_response(relay, recompute_ft=True)

        response[i, j] = relay.center_response[0]

    # clear network
    network.clear()

# visualize
plt.plot(patch_diameter, response[:, 0], '-o', label="FB weight={}".format(fb_weights[0]))
plt.plot(patch_diameter, response[:, 1], '-o', label="FB weight={}".format(fb_weights[1]))
plt.xlabel("Diameter (deg)")
plt.ylabel("Response (1/s)")
plt.legend()
plt.show()
