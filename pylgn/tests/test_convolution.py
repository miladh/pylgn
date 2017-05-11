import quantities as pq
import numpy as np

import pylgn
import pylgn.kernels as kernel
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl


def test_delta_delta():
    integrator = pylgn.Integrator(nt=5, nr=5, dt=1, dr=1)

    t_vec, x_vec, y_vec = integrator.meshgrid()
    w_vec, kx_vec, ky_vec = integrator.freq_meshgrid()

    delay_W = t_vec.flatten()[6]
    delay_K = t_vec.flatten()[10]

    shift_x = x_vec.flatten()[-3]
    shift_y = y_vec.flatten()[8]

    W_ft = tpl.create_delta_ft(delay_W)(w_vec) * spl.create_delta_ft(shift_x=shift_x)(kx_vec, ky_vec)
    K_ft = tpl.create_delta_ft(delay_K)(w_vec) * spl.create_delta_ft(shift_y=shift_y)(kx_vec, ky_vec)
    G = integrator.compute_inverse_fft(W_ft * K_ft)

    F = tpl.create_delta(1, delay_W)(t_vec-delay_K) * spl.create_delta(1, shift_x=shift_x)(x_vec, y_vec-shift_y)

    assert (abs(G - F) < complex(1e-12, 1e-12)).all()


def test_biphasic_delta():
    integrator = pylgn.Integrator(nt=10, nr=5, dt=0.1, dr=1)

    t_vec, x_vec, y_vec = integrator.meshgrid()
    w_vec, kx_vec, ky_vec = integrator.freq_meshgrid()

    delay_W = t_vec.flatten()[6]
    delay_K = t_vec.flatten()[10]

    shift_x = x_vec.flatten()[-3]
    shift_y = y_vec.flatten()[8]

    phase, damping = 42.5*pq.ms, 0.38

    W_ft = tpl.create_biphasic_ft(phase=phase,
                                  damping=damping,
                                  delay=delay_W)(w_vec) * spl.create_delta_ft(shift_x=shift_x)(kx_vec, ky_vec)
    K_ft = tpl.create_delta_ft(delay_K)(w_vec) * spl.create_delta_ft(shift_y=shift_y)(kx_vec, ky_vec)
    G = integrator.compute_inverse_fft(W_ft * K_ft)

    F = tpl.create_biphasic(phase=phase,
                            damping=damping,
                            delay=delay_W)(t_vec-delay_K) * spl.create_delta(1, shift_x=shift_x)(x_vec, y_vec-shift_y)

    assert (abs(G - F) < complex(1e-3, 1e-12)).all()


def test_gauss_delta():
    integrator = pylgn.Integrator(nt=3, nr=8, dt=0.1, dr=0.1)

    t_vec, x_vec, y_vec = integrator.meshgrid()
    w_vec, kx_vec, ky_vec = integrator.freq_meshgrid()

    delay_W = t_vec.flatten()[0]
    delay_K = t_vec.flatten()[0]

    shift_x = x_vec.flatten()[2**5]
    shift_y = y_vec.flatten()[-37]

    W_ft = tpl.create_delta_ft(delay_W)(w_vec) * spl.create_gauss_ft(a=0.62*pq.deg)(kx_vec,
                                                                                    ky_vec)
    K_ft = tpl.create_delta_ft(delay_K)(w_vec) * spl.create_delta_ft(shift_x,
                                                                     shift_y)(kx_vec,
                                                                              ky_vec)
    G = integrator.compute_inverse_fft(W_ft * K_ft)

    F = tpl.create_delta(1./integrator.dt, delay_W)(t_vec-delay_K) * spl.create_gauss(a=0.62*pq.deg)(x_vec-shift_x, y_vec-shift_y)

    assert (abs(G - F.magnitude) < complex(1e-10, 1e-12)).all()
