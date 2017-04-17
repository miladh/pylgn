import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

k_max_id = 40 
patch_diameter = np.array([3, 1.5, 0.85, 0.3]) * pq.deg
response = np.zeros([k_max_id, len(patch_diameter)]) / pq.s
 
# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
spatial_freqs = integrator.spatial_freqs[:k_max_id]

# create kernels
Wg_t = tpl.create_delta_ft()
Wg_r = spl.create_dog_ft(A=1, a=0.3*pq.deg, B=0.9, b=0.6*pq.deg)

# create neuron
ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

for j, d in enumerate(patch_diameter):
    for i, k_d in enumerate(spatial_freqs):
        # create stimulus
        stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=k_d, patch_diameter=d)
        network.set_stimulus(stimulus)

        # compute
        network.compute_response(ganglion, recompute_ft=True)
        response[i, j] = ganglion.center_response[0]
        
# visualize
for d, R in zip(patch_diameter, response.T):
    plt.plot(spatial_freqs, R, '-o', label="Diameter={}".format(d))
    
plt.xlabel("Wavenumber (1/deg)")
plt.ylabel("Response")
plt.legend()
plt.show()
