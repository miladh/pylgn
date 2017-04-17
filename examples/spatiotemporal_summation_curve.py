import quantities as pq
import numpy as np
import matplotlib.pyplot as plt

import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

# fb weights:
fb_weights = [0, -1.5]

# diameters
patch_diameter = np.linspace(0, 6, 50) * pq.deg

# list to store spatiotemporal summation curves for each weight
responses = []

for w_c in fb_weights:

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=5, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

    # create kernels
    delta_t = tpl.create_delta_ft()
    delta_s = spl.create_delta_ft()
    Wg_r = spl.create_dog_ft(A=1, a=0.25*pq.deg, B=0.85, b=0.83*pq.deg)
    Krc_r = spl.create_gauss_ft(A=1, a=0.83*pq.deg)

    # create neurons
    ganglion = network.create_ganglion_cell(kernel=(Wg_r, delta_t))
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()

    # connect neurons
    network.connect(ganglion, relay, (delta_s, delta_t), 1.0)
    network.connect(cortical, relay, (Krc_r, delta_t), w_c)
    network.connect(relay, cortical, (delta_s, delta_t), 1.0)

    st_summation_curve = np.zeros([len(patch_diameter), integrator.Nt]) / pq.s
    for i, d in enumerate(patch_diameter):
        # create stimulus
        k_pg = integrator.spatial_freqs[3]
        w_pg = integrator.temporal_freqs[1]
        stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=k_pg,
                                                          angular_freq=w_pg,
                                                          patch_diameter=d)
        network.set_stimulus(stimulus)

        # compute
        network.compute_response(relay, recompute_ft=True)

        st_summation_curve[i, :] = relay.center_response

    responses.append(st_summation_curve)
    # clear network
    network.clear()

# visualize
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey="row")

# xmin, xmax, ymin, ymax:
extent = [integrator.times.min(), integrator.times.max(), patch_diameter.min(), patch_diameter.max()]
vmin = -0.6
vmax = 0.6

im1 = ax1.imshow(responses[0], extent=extent, origin="lower", aspect="auto",
                 vmin=vmin, vmax=vmax)
ax1.set_title("FB weight={}".format(fb_weights[0]))
ax1.set_ylabel("Patch size (deg)")
ax1.set_xlabel("Time (ms)")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(responses[1], extent=extent, origin="lower", aspect="auto",
                 vmin=vmin, vmax=vmax)
ax2.set_title("FB weight={}".format(fb_weights[1]))
ax2.set_xlabel("Time (ms)")
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
