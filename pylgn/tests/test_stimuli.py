import quantities as pq
import numpy as np
import pytest

import pylgn
from pylgn.stimulus import (_check_valid_orient, _convert_to_cartesian,
                            _check_valid_spatial_freq, _check_valid_temporal_freq)


def test_check_valid_spatial_freq():
    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    assert _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec,
                                     kx_g=2.67, ky_g=0) is None

    with pytest.raises(ValueError):
        _check_valid_spatial_freq(kx_vec=k_vec[:-1], ky_vec=k_vec[:-1],
                                  kx_g=2.67, ky_g=0)

    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0.0]) / pq.deg
    assert _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec,
                                     kx_g=0, ky_g=5.2) is None

    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0.0]) / pq.deg
    assert _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec,
                                     kx_g=0, ky_g=8.2) is None

    k_vec = np.array([22.530299232133, 6.3202544656492]) / pq.deg
    assert _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec,
                                     kx_g=22.530299232133, ky_g=6.3202544656492) is None

    with pytest.raises(AttributeError):
        k_vec = np.array([22.530299232133, 6.3202544656492])
        _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec,
                                  kx_g=22.530299232133, ky_g=6.3202544656492)

    kx_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    ky_vec = np.array([0]) / pq.deg
    assert _check_valid_spatial_freq(kx_vec=kx_vec, ky_vec=ky_vec,
                                     kx_g=2.67, ky_g=0) is None

    kx_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    ky_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2]) / pq.deg

    with pytest.raises(ValueError):
        _check_valid_spatial_freq(kx_vec=kx_vec, ky_vec=ky_vec,
                                  kx_g=2.67, ky_g=0)


def test_check_valid_temporal_freq():
    A = np.array([0, 1, 2, 3, 4, 5])
    v_A = 2.3
    with pytest.raises(AttributeError):
        assert _check_valid_temporal_freq(A, v_A) is None

    B = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2]) * pq.Hz
    v_B = 5.2 * pq.Hz
    assert _check_valid_temporal_freq(B, v_B) is None

    C = np.array([9.66407446, 7.40204369, 8.47934683, 8.7268378, 9.537069,
                  8.94007828, 6.37876932, 7.84503963, 8.70901142]) * pq.Hz
    v_C = 8.479 * pq.Hz
    with pytest.raises(ValueError):
        assert _check_valid_temporal_freq(C, v_C) is None


def test_check_valid_orient():
    orient = 14.3
    assert _check_valid_orient(orient) is None

    orient = 352 * pq.deg
    assert _check_valid_orient(orient) is None

    orient = 4.567 * pq.rad
    assert _check_valid_orient(orient) is None

    orient = -182.456 * pq.deg
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient)

    orient = 7.4 * pq.rad
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient)

    orient = 7.4 * pq.deg
    assert _check_valid_orient(orient) is None

    orient = 7.4
    assert _check_valid_orient(orient) is None

    orient = 360.0000001
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient)


def test_convert_to_cartesian():
    angular_freq, wavenumber, orient = 0, 1.0, 0.0
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 1/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0*pq.Hz, 2.3/pq.deg, 0.0*pq.rad
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 2.3/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0*pq.Hz, -2.3/pq.deg, 0.0*pq.rad
    with pytest.warns(UserWarning):
        w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 2.3/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0*pq.Hz, 10.345/pq.deg, 90.0*pq.deg
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 10.345/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 10.6*pq.Hz, 10.345/pq.deg, np.pi*pq.rad
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - -10.345/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    assert abs(w - 10.6*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0*pq.Hz, 3.4/pq.deg, 45.
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 2.4041630560343/pq.deg) < 1e-12
    assert abs(ky - 2.4041630560343/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0*pq.Hz, 23.4, 15.67
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 22.530299232133/pq.deg) < 1e-12
    assert abs(ky - 6.3202544656492/pq.deg) < 1e-12
    assert abs(w - 0*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = 0.0, 23.4, -15.67
    with pytest.raises(ValueError):
        w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)

    angular_freq, wavenumber, orient = 2.45*pq.Hz, 3.4/pq.deg, 225.
    w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 2.4041630560343/pq.deg) < 1e-12
    assert abs(ky - 2.4041630560343/pq.deg) < 1e-12
    assert abs(w - -2.45*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = -34*pq.Hz, 10.345/pq.deg, 90.*pq.deg
    with pytest.warns(UserWarning):
        w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 10.345/pq.deg) < 1e-12
    assert abs(w - 34*pq.Hz) < 1e-12

    angular_freq, wavenumber, orient = -34*pq.Hz, 10.345/pq.deg, 270.*pq.deg
    with pytest.warns(UserWarning):
        w, kx, ky = _convert_to_cartesian(angular_freq, wavenumber, orient)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 10.345/pq.deg) < 1e-12
    assert abs(w - -34*pq.Hz) < 1e-12


def test_fullfield_grating_ft():
    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=0,
                                                          wavenumber=0,
                                                          orient=0,
                                                          contrast=23.4)
    assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == 8*np.pi**3 * 23.4

    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=2,
                                                          wavenumber=0,
                                                          orient=0,
                                                          contrast=1)
    with pytest.raises(ValueError):
        assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == 0

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=0,
                                                              wavenumber=-2.345,
                                                              orient=0,
                                                              contrast=1)
    with pytest.raises(ValueError):
        assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == 0

    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=0,
                                                          wavenumber=4.3,
                                                          orient=0.,
                                                          contrast=1)
    assert stimulus(w=0*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) == 4*np.pi**3

    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=3.1,
                                                          wavenumber=4.3,
                                                          orient=0.,
                                                          contrast=1)
    assert (stimulus(w=[-3.1, 3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) == np.array([0, 4*np.pi**3/6.2])).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=-3.1,
                                                              wavenumber=4.3,
                                                              orient=0,
                                                              contrast=1)
    assert (stimulus(w=[-3.1, 3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) == np.array([0, 4*np.pi**3/6.2])).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=-3.1,
                                                              wavenumber=4.3,
                                                              orient=0,
                                                              contrast=1)
    assert (stimulus(w=[3.1, -3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) == np.array([4*np.pi**3/6.2, 0])).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=-3.1,
                                                              wavenumber=4.3,
                                                              orient=0,
                                                              contrast=1)

    assert (stimulus(w=[3.1, -3.1]*pq.Hz, kx=[-4.3, 4.3]/pq.deg, ky=0/pq.deg) == np.array([0, 0])).all()

    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=3.1,
                                                          wavenumber=4.3,
                                                          orient=180,
                                                          contrast=1)

    assert (stimulus(w=[-3.1, 3.1]*pq.Hz, kx=-4.3/pq.deg, ky=0/pq.deg) == np.array([0, 4*np.pi**3/6.2])).all()


def test_fullfield_grating():
    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=0,
                                                       wavenumber=0,
                                                       orient=0,
                                                       contrast=1)
    assert stimulus(t=0*pq.s, x=0*pq.deg, y=0*pq.deg) == 1

    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=0,
                                                       wavenumber=0,
                                                       orient=0,
                                                       contrast=1)

    assert stimulus(t=1.2*pq.s, x=2.30*pq.deg, y=-2.34*pq.deg) == 1

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                           wavenumber=0.785398163397448,
                                                           orient=180,
                                                           contrast=1.)

    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - 0.587785252292) < 1e-12

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                           wavenumber=0.785398163397448,
                                                           orient=180,
                                                           contrast=1.)
        assert abs(stimulus(t=2.1*pq.s, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.522498564716) < 1e-12

    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=2.356194490192345*pq.kHz,
                                                       wavenumber=-0.39269908169872414,
                                                       orient=0,
                                                       contrast=2.1)
    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.726845819863) < 1e-12


def test_fullfield_grating_with_fft():
    network = pylgn.Network()
    integrator = network.create_integrator(nt=2, nr=7, dt=1*pq.ms, dr=0.1*pq.deg)
    t, x, y = integrator.meshgrid()

    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=0,
                                                       wavenumber=2,
                                                       orient=33,
                                                       contrast=1)
    network.set_stimulus(stimulus, compute_fft=True)
    assert(abs(integrator.compute_inverse_fft(network.stimulus.ft) - stimulus(t, x, y)) < 1e-12).all()


def test_patch_grating():
    stimulus = pylgn.stimulus.create_patch_grating(angular_freq=0,
                                                   wavenumber=0,
                                                   orient=0,
                                                   contrast=1,
                                                   patch_diameter=1.4*pq.deg)
    assert stimulus(t=0*pq.s, x=0*pq.deg, y=0*pq.deg) == 1
    assert stimulus(t=3.5*pq.s, x=0.4*pq.deg, y=-0.3*pq.deg) == 1
    assert stimulus(t=1.2*pq.s, x=2.30*pq.deg, y=-2.34*pq.deg) == 0

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                       wavenumber=0.785398163397448,
                                                       orient=180,
                                                       contrast=1.,
                                                       patch_diameter=8*pq.deg)

    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - 0.587785252292) < 1e-12
    assert stimulus(t=1.2*pq.s, x=9.30*pq.deg, y=-2.34*pq.deg) == 0

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                       wavenumber=0.785398163397448,
                                                       orient=180,
                                                       contrast=1.,
                                                       patch_diameter=8*pq.deg)
        assert abs(stimulus(t=2.1*pq.s, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.522498564716) < 1e-12
        assert stimulus(t=1.2*pq.s, x=1.30*pq.deg, y=-7.34*pq.deg) == 0

    stimulus = pylgn.stimulus.create_patch_grating(angular_freq=2.356194490192345*pq.kHz,
                                                   wavenumber=-0.39269908169872414,
                                                   orient=0,
                                                   contrast=2.1,
                                                   patch_diameter=8*pq.deg)

    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.726845819863) < 1e-12
    assert stimulus(t=1.2*pq.s, x=1.30*pq.deg, y=-7.34*pq.deg) == 0


def test_patch_grating_ft():
    C = 23.4
    size = 1*pq.deg
    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=0,
                                                      wavenumber=0,
                                                      orient=0,
                                                      contrast=C,
                                                      patch_diameter=size)

    assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == C * np.pi**2 * size**2 / 2

    C = 1
    size = 1*pq.deg
    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=2,
                                                      wavenumber=0,
                                                      orient=0,
                                                      contrast=C,
                                                      patch_diameter=size)
    with pytest.raises(ValueError):
        assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == 0

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=0,
                                                          wavenumber=-2.345,
                                                          orient=0,
                                                          contrast=C,
                                                          patch_diameter=size)
    with pytest.raises(ValueError):
        assert stimulus(w=0*pq.Hz, kx=0/pq.deg, ky=0/pq.deg) == 0

    C = 1
    size = 1*pq.deg
    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=0,
                                                      wavenumber=4.3,
                                                      orient=0,
                                                      contrast=C,
                                                      patch_diameter=size)

    assert abs(stimulus(w=0*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) - 2.2701277226799) < 1e-12

    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=3.1,
                                                      wavenumber=4.3,
                                                      orient=0,
                                                      contrast=C,
                                                      patch_diameter=size)

    assert (abs(stimulus(w=[-3.1, 3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) - [-0.0318182867085, 0.3979679193988]) < 1e-12).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=-3.1,
                                                          wavenumber=4.3,
                                                          orient=0,
                                                          contrast=C,
                                                          patch_diameter=size)
    assert (abs(stimulus(w=[-3.1, 3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) - [-0.0318182867085, 0.3979679193988]) < 1e-12).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=-3.1,
                                                          wavenumber=4.3,
                                                          orient=0,
                                                          contrast=C,
                                                          patch_diameter=size)
    assert (abs(stimulus(w=[3.1, -3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) - [0.3979679193988, -0.0318182867085]) < 1e-12).all()

    with pytest.warns(UserWarning):
        stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=-3.1,
                                                          wavenumber=4.3,
                                                          orient=0,
                                                          contrast=C,
                                                          patch_diameter=size)

    assert (abs(stimulus(w=[3.1, -3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) - [0.3979679193988, -0.0318182867085]) < 1e-12).all()

    C = 2
    size = 7*pq.deg
    stimulus = pylgn.stimulus.create_patch_grating_ft(angular_freq=3.1,
                                                      wavenumber=4.3,
                                                      orient=180,
                                                      contrast=C,
                                                      patch_diameter=size)

    assert (abs(stimulus(w=[-3.1, 3.1]*pq.Hz, kx=-4.3/pq.deg, ky=0/pq.deg) - [-0.3274845723539, 39.0008561010789]) < 1e-12).all()


def test_flashing_spot():
    stimulus = pylgn.stimulus.create_flashing_spot(contrast=1,
                                                   patch_diameter=2*pq.deg,
                                                   delay=0*pq.ms,
                                                   duration=2*pq.ms)

    assert stimulus(t=0*pq.s, x=0*pq.deg, y=0*pq.deg) == 1
    assert stimulus(t=3.5*pq.s, x=0.4*pq.deg, y=-0.3*pq.deg) == 0
    assert stimulus(t=1.2*pq.s, x=2.30*pq.deg, y=-2.34*pq.deg) == 0

    stimulus = pylgn.stimulus.create_flashing_spot(contrast=1.2,
                                                   patch_diameter=10.5*pq.deg,
                                                   delay=5,
                                                   duration=2.4)

    assert stimulus(t=0*pq.s, x=0*pq.deg, y=0*pq.deg) == 0
    assert stimulus(t=3.5*pq.s, x=0.4*pq.deg, y=-0.3*pq.deg) == 0
    assert stimulus(t=7.2*pq.ms, x=1.30*pq.deg, y=-0.34*pq.deg) == 1.2

    with pytest.raises(ValueError):
        stimulus = pylgn.stimulus.create_flashing_spot(contrast=1.2,
                                                       patch_diameter=10.5*pq.deg,
                                                       delay=-5*pq.ms,
                                                       duration=2.4*pq.ms)

    with pytest.raises(ValueError):
        stimulus = pylgn.stimulus.create_flashing_spot(contrast=1.2,
                                                       patch_diameter=10.5*pq.deg,
                                                       delay=5*pq.ms,
                                                       duration=-2.4*pq.ms)


def test_flashing_spot_ft():
    with pytest.raises(ValueError):
        stimulus = pylgn.stimulus.create_flashing_spot_ft(contrast=1.2,
                                                          patch_diameter=10.5*pq.deg,
                                                          delay=-5*pq.ms,
                                                          duration=2.4*pq.ms)

    with pytest.raises(ValueError):
        stimulus = pylgn.stimulus.create_flashing_spot_ft(contrast=1.2,
                                                          patch_diameter=10.5*pq.deg,
                                                          delay=5*pq.ms,
                                                          duration=-2.4*pq.ms)


def test_natural_image():
    with pytest.raises(ValueError):
            stimulus = pylgn.stimulus.create_natural_image("",
                                                           delay=0*pq.ms,
                                                           duration=-20*pq.ms)

            stimulus = pylgn.stimulus.create_natural_image("",
                                                           delay=-1*pq.ms,
                                                           duration=-20*pq.ms)


def test_natural_movie():
    stimulus = pylgn.stimulus.create_natural_movie("test.gif")

    with pytest.raises(NameError):
            stimulus = pylgn.stimulus.create_natural_movie("")
