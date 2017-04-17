import quantities as pq
import numpy as np
import pytest

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

from pylgn.helper import confluent_hypergeometric


def X(y, z, n):
    import math
    res = 0
    exp_term = np.exp(-z**2/4) / (4*y**2)

    for i in range(int(n)):
        n_fac = 1. / math.factorial(i)
        term = (z.magnitude/2)**(2*i)
        hyp_geom = confluent_hypergeometric(i+1, 2, -1./(4*y**2))
        res += n_fac * term * hyp_geom

    return res * exp_term


@pytest.mark.system
@pytest.mark.parametrize("nt", [1])
@pytest.mark.parametrize("nr", [8])
@pytest.mark.parametrize("dt", [1]*pq.ms)
@pytest.mark.parametrize("dr", [0.1]*pq.deg)
@pytest.mark.parametrize("A_g", [1])
@pytest.mark.parametrize("a_g", [0.62]*pq.deg)
@pytest.mark.parametrize("B_g", [0.85])
@pytest.mark.parametrize("b_g", [1.26]*pq.deg)
@pytest.mark.parametrize("delay_g", [1.34]*pq.ms)
@pytest.mark.parametrize("w_id", [0])
@pytest.mark.parametrize("k_id", [0])
@pytest.mark.parametrize("orient", [0, 90]*pq.deg)
@pytest.mark.parametrize("C", [-56.3])
@pytest.mark.parametrize("patch_diameter", [1, 5, 10]*pq.deg)
@pytest.mark.parametrize("n", [4])
def test_G_patch_grating_response(nt, nr, dt, dr,
                                  A_g, a_g, B_g, b_g, delay_g,
                                  w_id, k_id, orient, C, patch_diameter,
                                  n):
    network = pylgn.Network()
    integrator = network.create_integrator(nt=nt, nr=nr, dt=dt, dr=dr)

    w_g = integrator.temporal_freqs[w_id].rescale(1./pq.ms)
    k_g = integrator.spatial_freqs[k_id]

    Wg_r = spl.create_dog_ft(A=A_g, a=a_g, B=B_g, b=b_g)
    Wg_t = tpl.create_delta_ft(delay=delay_g)

    ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))

    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=w_g,
                                                      wavenumber=k_g,
                                                      orient=orient,
                                                      contrast=C,
                                                      patch_diameter=patch_diameter)
    network.set_stimulus(stimulus)
    network.compute_response(ganglion)

    Rg = C * (X(a_g/patch_diameter, a_g*k_g, n) - B_g/A_g * X(b_g/patch_diameter, b_g*k_g, n)) / pq.s
    assert abs(Rg - ganglion.center_response[0]) < complex(1e-10, 1e-10)


@pytest.mark.system
@pytest.mark.parametrize("nt", [5])
@pytest.mark.parametrize("nr", [5])
@pytest.mark.parametrize("dt", [0.1]*pq.ms)
@pytest.mark.parametrize("dr", [0.1]*pq.deg)
@pytest.mark.parametrize("A_g", [1])
@pytest.mark.parametrize("a_g", [0.62]*pq.deg)
@pytest.mark.parametrize("B_g", [0.83])
@pytest.mark.parametrize("b_g", [1.26]*pq.deg)
@pytest.mark.parametrize("delay_g", [0.0, 2.4]*pq.ms)
@pytest.mark.parametrize("A_rc", [0.3])
@pytest.mark.parametrize("a_rc", [0.1]*pq.deg)
@pytest.mark.parametrize("B_rc", [0.6])
@pytest.mark.parametrize("b_rc", [0.9]*pq.deg)
@pytest.mark.parametrize("delay_rc", [1.7]*pq.ms)
@pytest.mark.parametrize("w_rc", [1.234, -201.])
@pytest.mark.parametrize("a_rg", [0.25]*pq.deg)
@pytest.mark.parametrize("delay_rg", [3.2]*pq.ms)
@pytest.mark.parametrize("w_rg", [-1.2])
@pytest.mark.parametrize("w_id", [0, 24])
@pytest.mark.parametrize("k_id", [0, 8])
@pytest.mark.parametrize("orient", [0, 90, 180]*pq.deg)
@pytest.mark.parametrize("C", [1.23, -1.45])
def test_G_R_C_grating_response(nt, nr, dt, dr,
                                A_g, a_g, B_g, b_g, delay_g,
                                A_rc, a_rc, B_rc, b_rc, delay_rc, w_rc,
                                a_rg, delay_rg, w_rg,
                                w_id, k_id, orient, C):

    network = pylgn.Network()
    integrator = network.create_integrator(nt=nt, nr=nr, dt=dt, dr=dr)

    w_g = integrator.temporal_freqs[w_id].rescale(1./pq.ms)
    k_g = integrator.spatial_freqs[k_id]
    kx_g = k_g * np.cos(orient.rescale(pq.rad))
    ky_g = k_g * np.sin(orient.rescale(pq.rad))

    t, x, y = integrator.meshgrid()

    Wg_r = spl.create_dog_ft(A=A_g, a=a_g, B=B_g, b=b_g)
    Wg_t = tpl.create_delta_ft(delay=delay_g)

    Krg_r = spl.create_gauss_ft(a=a_rg)
    Krg_t = tpl.create_delta_ft(delay=delay_rg)

    Krc_r = spl.create_dog_ft(A=A_rc, a=a_rc, B=B_rc, b=b_rc)
    Krc_t = tpl.create_delta_ft(delay=delay_rc)

    Kcr_r = spl.create_delta_ft()
    Kcr_t = tpl.create_delta_ft()

    ganglion = network.create_ganglion_cell(kernel=(Wg_r, Wg_t))
    relay = network.create_relay_cell()
    cortical = network.create_cortical_cell()

    network.connect(ganglion, relay, (Krg_r, Krg_t), weight=w_rg)
    network.connect(cortical, relay, (Krc_r, Krc_t), weight=w_rc)
    network.connect(relay, cortical, (Kcr_r, Kcr_t), weight=1)

    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=w_g,
                                                          wavenumber=k_g,
                                                          orient=orient,
                                                          contrast=C)
    network.set_stimulus(stimulus)
    network.compute_response(ganglion)
    network.compute_response(relay)
    network.compute_response(cortical)

    Wg = ganglion.evaluate_irf_ft(w=w_g, kx=kx_g, ky=ky_g)
    Wr = relay.evaluate_irf_ft(w=w_g, kx=kx_g, ky=ky_g)
    Wc = cortical.evaluate_irf_ft(w=w_g, kx=kx_g, ky=ky_g)

    Rg = C * abs(Wg) * np.cos(kx_g*x + ky_g*y - w_g*t + np.angle(Wg)) / pq.s
    Rr = C * abs(Wr) * np.cos(kx_g*x + ky_g*y - w_g*t + np.angle(Wr)) / pq.s
    Rc = C * abs(Wc) * np.cos(kx_g*x + ky_g*y - w_g*t + np.angle(Wc)) / pq.s

    assert (abs(Rg - ganglion.response) < complex(1e-12, 1e-12)).all()
    assert (abs(Rr - relay.response) < complex(1e-12, 1e-12)).all()
    assert (abs(Rc.clip(min=0/pq.s) - cortical.response) < complex(1e-12, 1e-12)).all()
