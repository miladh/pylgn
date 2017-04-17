import quantities as pq
import numpy as np
import pytest

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl


@pytest.mark.core
def test_core():
    dt = 1*pq.ms
    dr = 0.1*pq.deg

    # create network
    network = pylgn.Network()

    # create integrator
    integrator = network.create_integrator(nt=7, nr=7, dt=dt, dr=dr)

    # create neurons
    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell(2/pq.ms)
    cortical = network.create_cortical_cell(0/pq.ms)

    # connect neurons
    Wg_r = spl.create_dog_ft(A=1, a=0.62*pq.deg, B=0.83, b=1.26*pq.deg)
    Wg_t = tpl.create_delta_ft(delay=0*pq.ms)
    # Wg_t = tpl.create_biphasic_ft(phase_duration=43*pq.ms,
    #                               damping_factor=0.38,
    #                               delay=0*pq.ms)

    Krg_r = spl.create_delta_ft(shift_x=0*pq.deg, shift_y=0*pq.deg)
    Krg_t = tpl.create_delta_ft(delay=0*pq.ms)

    Krc_r = spl.create_dog_ft(A=0.3, a=0.1*pq.deg, B=0.6, b=0.9*pq.deg)
    Krc_t = tpl.create_delta_ft(delay=0*pq.ms)

    Kcr_r = spl.create_delta_ft(shift_x=0*pq.deg, shift_y=0*pq.deg)
    Kcr_t = tpl.create_delta_ft(delay=0*pq.ms)

    ganglion.set_kernel((Wg_r, Wg_t))
    network.connect(ganglion, relay, (Krg_r, Krg_t))
    network.connect(cortical, relay, (Krc_r, Krc_t))
    network.connect(relay, cortical, (Kcr_r, Kcr_t))

    print(ganglion.annotations["kernel"], "\n")

    # create stimulus
    k_g = integrator.spatial_freqs[6]
    w_g = -integrator.temporal_freqs[40]
    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=w_g,
                                                      wavenumber=k_g,
                                                      orient=0.0,
                                                      patch_diameter=3,
                                                      contrast=4)
    network.set_stimulus(stimulus)

    print(pylgn.closure_params(stimulus))

    # compute
    network.compute_irf(relay)
    network.compute_response(relay)
    network.compute_response(ganglion)

    # write
    print("shape cube: ", relay.irf_ft.shape)
    print("Unit irf:", relay.irf.units)
    # print("Units irf_ft: ", relay.irf_ft.units)
    print("Units resp: ", relay.response.units)

    # visulize
    # import matplotlib.pyplot as plt
    # import matplotlib.animation as animation
    # plt.imshow(relay.irf[0, :, :].real)
    #
    # plt.figure()
    # plt.plot(np.real(ganglion.response[:, 3, 3]))
    # pylgn.plot.animate_cube(ganglion.response)

    # clear network
    network.clear()


@pytest.mark.core
def test_minimal():
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
    k_g = integrator.spatial_freqs[3]
    w_g = -integrator.temporal_freqs[1]
    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=w_g,
                                                          wavenumber=k_g,
                                                          orient=0.0)
    network.set_stimulus(stimulus)

    # compute
    network.compute_irf(relay)
    network.compute_response(relay)

    # visulize
    # pylgn.plot.animate_cube(relay.response, title="Relay cell response")

    # write
    filename = "/tmp/pylgn_test"
    import shutil
    import os

    try:
        shutil.rmtree(filename+".exdir")
    except OSError:
        pass

    try:
        os.remove(filename+".hdf5")
        print("removing file....")
    except OSError:
        pass

    io = pylgn.io.ExdirIO(filename=filename+".exdir")
    io.write_network(network)

    io = pylgn.io.Hdf5IO(filename=filename+".hdf5")
    io.write_network(network)


# if __name__ == "__main__":
#     path = "/home/milad/Dropbox/projects/lgn/code/stimuli/tarzan.avi"
#     pylgn.stimulus.create_natural_movie(path)
