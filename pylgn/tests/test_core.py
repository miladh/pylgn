import quantities as pq
import numpy as np
import pytest

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl


@pytest.mark.core
def test_core():
    dt = 5*pq.ms
    dr = 0.1*pq.deg

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=10, nr=7, dt=dt, dr=dr)

    # create neurons
    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell(2./pq.s)
    cortical = network.create_cortical_cell(0/pq.s)

    # connect neurons
    Wg_r = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.83, b=1.26*pq.deg)
    Wg_t = tpl.create_biphasic_ft(phase=43*pq.ms,
                                  damping=0.38,
                                  delay=0*pq.ms)

    Krg_r = spl.create_delta_ft(shift_x=0*pq.deg, shift_y=0*pq.deg)
    Krg_t = tpl.create_delta_ft(delay=0*pq.ms)

    Krc_r = spl.create_dog_ft(A=1*0.9, a=0.1*pq.deg, B=2*0.9, b=0.9*pq.deg)
    Krc_t = tpl.create_delta_ft(delay=20*pq.ms)

    Kcr_r = spl.create_delta_ft(shift_x=0*pq.deg, shift_y=0*pq.deg)
    Kcr_t = tpl.create_delta_ft(delay=0*pq.ms)

    ganglion.set_kernel((Wg_r, Wg_t))
    network.connect(ganglion, relay, (Krg_r, Krg_t))
    network.connect(cortical, relay, (Krc_r, Krc_t))
    network.connect(relay, cortical, (Kcr_r, Kcr_t))

    print(ganglion.annotations["kernel"], "\n")

    # create stimulus
    k_g = integrator.spatial_freqs[2]
    w_g = -integrator.temporal_freqs[40]
    # stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=w_g,
    #                                                   wavenumber=k_g,
    #                                                   orient=0.0,
    #                                                   patch_diameter=3,
    #                                                   contrast=4)
    stimulus = pylgn.stimulus.create_flashing_spot_ft(patch_diameter=4*pq.deg,
                                                      contrast=4,
                                                      delay=4*pq.ms,
                                                      duration=20*pq.ms)
    network.set_stimulus(stimulus, compute_fft=False)
    # network.set_stimulus(stimulus)

    # print(pylgn.closure_params(stimulus))
    #
    # # compute
    # network.compute_irf(relay)
    network.compute_response(relay)
    # network.compute_response(ganglion)
    #
    # # write
    # print("shape cube: ", relay.irf_ft.shape)
    # print("Unit irf:", relay.irf.units)
    # # print("Units irf_ft: ", relay.irf_ft.units)
    # print("Units resp: ", relay.response.units)

    # visulize
    # import matplotlib.pyplot as plt
    # # import matplotlib.animation as animation
    # # plt.imshow(relay.irf[0, :, :].real)
    #
    # plt.figure()
    # # plt.plot(integrator.times, relay.response[:, 64, 64])
    # w_g = w_g.rescale(pq.Hz) * integrator.times/integrator.times.max()
    # plt.plot(w_g/2./np.pi, relay.response[:, 64, 64])
    # plt.show()
    # pylgn.plot.animate_cube(relay.response)

    # clear network
    network.clear()
