import numpy as np
import quantities as pq
import pytest

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl


@pytest.mark.network
def test_edog_patch_grating_response():
    response_e = np.array([[0.0000000000, 0.0000000000],
                           [0.5694814680, 0.3933539871],
                           [0.4933473520, 0.3005140715],
                           [0.5038172723, 0.2842527074],
                           [0.5039615741, 0.2843698815]]) / pq.s
    fb_weights = [0, -1.5]

    patch_diameter = np.linspace(0, 6, 5) * pq.deg
    response = np.zeros([len(patch_diameter), len(fb_weights)]) / pq.s

    for j, w_c in enumerate(fb_weights):
        network = pylgn.Network()
        integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

        delta_t = tpl.create_delta_ft()
        delta_s = spl.create_delta_ft()
        Wg_r = spl.create_dog_ft(A=1, a=0.25*pq.deg, B=0.85, b=0.83*pq.deg)
        Krc_r = spl.create_gauss_ft(A=1, a=0.83*pq.deg)

        ganglion = network.create_ganglion_cell(kernel=(Wg_r, delta_t))
        relay = network.create_relay_cell()
        cortical = network.create_cortical_cell()

        network.connect(ganglion, relay, (delta_s, delta_t), 1.0)
        network.connect(cortical, relay, (Krc_r, delta_t), w_c)
        network.connect(relay, cortical, (delta_s, delta_t), 1.0)

        for i, d in enumerate(patch_diameter):
            stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=integrator.spatial_angular_freqs[4], patch_diameter=d)
            network.set_stimulus(stimulus)
            network.compute_response(relay, recompute_ft=True)
            response[i, j] = relay.center_response[0]

        network.clear()

    assert (abs(response - response_e) < 1e-10).all()


@pytest.mark.network
def test_edog_spot_response():
    response_e = np.array([[0.0000000000, 0.0000000000],
                           [0.5255489103, 0.3077428042],
                           [0.1824324695, 0.0174985410],
                           [0.1505469275, 0.0632031066],
                           [0.1500017999, 0.0601939846]]) / pq.s
    fb_weights = [0, -1.5]

    patch_diameter = np.linspace(0, 6, 5) * pq.deg
    response = np.zeros([len(patch_diameter), len(fb_weights)]) / pq.s

    for j, w_c in enumerate(fb_weights):
        network = pylgn.Network()
        integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)

        delta_t = tpl.create_delta_ft()
        delta_s = spl.create_delta_ft()
        Wg_r = spl.create_dog_ft(A=1, a=0.25*pq.deg, B=0.85, b=0.83*pq.deg)
        Krc_r = spl.create_gauss_ft(A=1, a=0.83*pq.deg)

        ganglion = network.create_ganglion_cell(kernel=(Wg_r, delta_t))
        relay = network.create_relay_cell()
        cortical = network.create_cortical_cell()

        network.connect(ganglion, relay, (delta_s, delta_t), 1.0)
        network.connect(cortical, relay, (Krc_r, delta_t), w_c)
        network.connect(relay, cortical, (delta_s, delta_t), 1.0)

        for i, d in enumerate(patch_diameter):
            stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=0./pq.deg,
                                                              patch_diameter=d)
            network.set_stimulus(stimulus)
            network.compute_response(relay, recompute_ft=True)
            response[i, j] = relay.center_response[0]

        network.clear()

    assert (abs(response - response_e) < 1e-10).all()


@pytest.mark.network
def test_dog_patch_grating_response():
    response_e = np.array([[0.1017374087, 0.2867197943, 0.5105320011, 0.1666709735],
                          [0.3492268988, 0.3616052645, 0.4750891407, 0.1640082272],
                          [0.4789040160, 0.4328530438, 0.3805594923, 0.1561941696],
                          [0.2884185736, 0.3175258969, 0.2569232750, 0.1437369651],
                          [0.1140384287, 0.1163952049, 0.1390511885, 0.1274369599],
                          [0.0339383730, 0.0179229458, 0.0526124870, 0.1083213337],
                          [0.0074631404, 0.0145706277, 0.0061438229, 0.0875612286],
                          [0.0013737748, 0.0074470223, -0.0077840845, 0.0663789388]]) / pq.s

    k_max_id = 40
    step = 5
    patch_diameter = np.array([3, 1.5, 0.85, 0.3]) * pq.deg
    response = np.zeros([int(k_max_id/step), len(patch_diameter)]) / pq.s

    network = pylgn.Network()

    integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
    spatial_angular_freqs = integrator.spatial_angular_freqs[:k_max_id][::step]

    Wg_t = tpl.create_delta_ft()
    Wg_r = spl.create_dog_ft(A=1, a=0.3*pq.deg, B=0.9, b=0.6*pq.deg)

    ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

    for j, d in enumerate(patch_diameter):
        for i, k_d in enumerate(spatial_angular_freqs):
            stimulus = pylgn.stimulus.create_patch_grating_ft(wavenumber=k_d,
                                                              patch_diameter=d)
            network.set_stimulus(stimulus)

            network.compute_response(ganglion, recompute_ft=True)
            response[i, j] = ganglion.center_response[0]

    assert (abs(response - response_e) < 1e-10).all()


@pytest.mark.network
def test_nonlagged_x_cells():
    R_g_e = np.array([36.8000000000, 105.5231755992, 80.5607155005, 60.1117829282, 56.7461598311, 56.5031395446, 56.4951231072, 56.4950008690, 56.4950000029, 56.4950000000])/pq.s

    R_r_e = np.array([9.1000000000, 48.9839667670, 20.1915848052, 11.3884937007, 13.2064761119, 13.9269478889, 14.0176607852, 14.0235290468, 14.0237452461, 14.0237499389])/pq.s

    patch_diameter = np.linspace(0, 14, 10) * pq.deg
    R_g = np.zeros(len(patch_diameter)) / pq.s
    R_r = np.zeros(len(patch_diameter)) / pq.s

    network = pylgn.Network()
    integrator = network.create_integrator(nt=1, nr=7, dt=1*pq.ms, dr=0.2*pq.deg)
    ganglion = network.create_ganglion_cell(background_response=36.8/pq.s)
    relay = network.create_relay_cell(background_response=9.1/pq.s)

    Wg_r = spl.create_dog_ft(A=-1, a=0.62*pq.deg, B=-0.85, b=1.26*pq.deg)
    Krig_r = spl.create_gauss_ft(A=1, a=0.88*pq.deg)
    Krg_r = spl.create_delta_ft()

    ganglion.set_kernel((Wg_r, tpl.create_delta_ft()))
    network.connect(ganglion, relay, (Krg_r, tpl.create_delta_ft()), weight=0.81)
    network.connect(ganglion, relay, (Krig_r, tpl.create_delta_ft()), weight=-0.56)

    for i, d in enumerate(patch_diameter):
        stimulus = pylgn.stimulus.create_patch_grating_ft(patch_diameter=d, contrast=-131.3)
        network.set_stimulus(stimulus)

        network.compute_response(ganglion, recompute_ft=True)
        network.compute_response(relay, recompute_ft=True)

        R_g[i] = ganglion.center_response[0]
        R_r[i] = relay.center_response[0]

    assert (abs(R_g - R_g_e) < 1e-9).all()
    assert (abs(R_r - R_r_e) < 1e-9).all()
