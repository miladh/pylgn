import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

    
mask_size = np.linspace(0, 14, 50) * pq.deg 
R_g = np.zeros(len(mask_size)) / pq.s
R_r = np.zeros(len(mask_size)) / pq.s

# create network
network = pylgn.Network()

# create integrator
integrator = network.create_integrator(nt=1, nr=8, dt=1*pq.ms, dr=0.1*pq.deg)

# create neurons
ganglion = network.create_ganglion_cell(background_response=36.8/pq.s)
relay = network.create_relay_cell(background_response=9.1/pq.s)

# create kernels
Wg_r = spl.create_dog_ft(A=-1, a=0.62*pq.deg, B=-0.85, b=1.26*pq.deg)
Krig_r = spl.create_gauss_ft(A=1, a=0.88*pq.deg)
Krg_r = spl.create_delta_ft()

# connect neurons    
ganglion.set_kernel((Wg_r, tpl.create_delta_ft()))
network.connect(ganglion, relay, (Krg_r, tpl.create_delta_ft()), weight=0.81)
network.connect(ganglion, relay, (Krig_r, tpl.create_delta_ft()), weight=-0.56)

for i, d in enumerate(mask_size):
    # create stimulus
    stimulus = pylgn.stimulus.create_patch_grating_ft(mask_size=d, contrast=-131.3)
    network.set_stimulus(stimulus)

    # compute
    network.compute_response(ganglion, recompute_ft=True)
    network.compute_response(relay, recompute_ft=True)

    R_g[i] = ganglion.center_response[0] 
    R_r[i] = relay.center_response[0] 

# visualize
plt.plot(mask_size, R_g, '-o', label="Ganglion")
plt.plot(mask_size, R_r, '-o', label="Relay")
plt.xlabel("Spot diameter (deg)")
plt.ylabel("Response (spikes/s)")
plt.legend()
plt.show()
