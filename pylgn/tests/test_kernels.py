import numpy as np
import pytest
import pylgn
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
from .helper import (create_dog, create_gauss, create_spatial_delta,
                     create_temporal_delta, create_exp_decay, create_biphasic)


##############################################################
# spatiotemporal
##############################################################
def test_non_separable_kernel():
    non_sep = pylgn.kernels.create_non_separable_kernel()

    with pytest.raises(NotImplementedError):
        non_sep(0, 0, 0)


##############################################################
# spatial
##############################################################
def test_spl_dog():
    dog = create_dog(A=1.0, a=0.25, B=0.85, b=0.83, dx=0, dy=0)
    assert abs(dog(x=0.5, y=0.1)--0.189791527743) < 1e-12
    assert abs(dog(x=1.2, y=1.9)--0.00025733892027) < 1e-12


def test_spl_dog_ft():
    dog_ft = spl.create_dog_ft(A=1.0, a=0.25, B=0.85, b=0.83, dx=0, dy=0)
    assert abs(dog_ft(kx=0.5, ky=1.1)-0.316423256919) < complex(1e-12, 0)
    assert abs(dog_ft(kx=1.5, ky=0.1)-0.389361200098) < complex(1e-12, 0)


def test_spl_gauss():
    gauss = create_gauss(A=1, a=0.25, dx=0, dy=0)
    assert abs(gauss(x=0.5, y=0.1)-0.079488639761866486) < 1e-12
    assert(gauss(x=1.2, y=1.9) < 1e-12)


def test_spl_gauss_ft():
    gauss_ft = spl.create_gauss_ft(A=1, a=0.25, dx=0, dy=0)
    assert abs(gauss_ft(kx=0.5, ky=1.1)-0.9774457376685004) < complex(1e-12, 0)
    assert abs(gauss_ft(kx=1.5, ky=0.1)-0.96530371170877705) < complex(1e-12, 0)


def test_spl_delta():
    delta = create_spatial_delta(1., 0, 0)
    assert(delta(x=0, y=0) == 1)

    delta = create_spatial_delta(2.3, 2.1, 3.4)
    assert(delta(x=2.1, y=3.4) == 2.3)

    delta = create_spatial_delta(-4.3, 1.456, 0)
    assert(delta(x=0, y=10.345) == 0)

    delta = create_spatial_delta(-4.3, 0, 1.456)
    assert(delta(x=0, y=1.456) == -4.3)


def test_spl_delta_ft():
    delta_ft = spl.create_delta_ft(0.5, 0.1)

    assert abs(delta_ft(kx=2.5, ky=-3.1) - complex(0.5897880250310923,
                                                   -0.8075581004051077)) < complex(1e-12, 1e-12)


##############################################################
# temporal
##############################################################
def test_create_exp_decay():
    tau = 0.25
    delay = 0.0

    exp_decay = create_exp_decay(tau, delay)
    assert abs(exp_decay(-10.24) - 0) < 1e-12
    assert abs(exp_decay(0.0) - 4.0) < 1e-12
    assert abs(exp_decay(1.1) - 0.049109359612273744) < 1e-12
    assert abs(exp_decay(1.8) - 0.0029863432335067168) < 1e-12
    assert abs(exp_decay(2.3) - 0.00040415760734837367) < 1e-12
    assert abs(exp_decay(100.4) - 0.000000000000000) < 1e-12

    tau = 2.3
    delay = 1.0056

    exp_decay = create_exp_decay(tau, delay)

    assert abs(exp_decay(-10.24) - 0.) < 1e-12
    assert abs(exp_decay(0.0) - 0) < 1e-12
    assert abs(exp_decay(1.1) - 0.41729886919726) < 1e-12
    assert abs(exp_decay(1.8) - 0.30780142520792) < 1e-12
    assert abs(exp_decay(2.3) - 0.24766166169630) < 1e-12
    assert abs(exp_decay(100.4) - 0) < 1e-12


def test_create_exp_decay_ft():
    tau = 0.25
    delay = 0.0

    exp_decay_ft = tpl.create_exp_decay_ft(tau, delay)

    assert abs(exp_decay_ft(0.0) - complex(1, 0)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(-157.079632679) - complex(0.00064803535317723543, -0.025448288810021549)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(314.159265359) - complex(0.00016208761717295794, 0.012730331683711824)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(0.0383495196971) - complex(0.99990809059430297, 0.009586498753883968)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(1005.18693069) - complex(1.5835049995458247e-05, 0.0039792963255643439)) < complex(1e-12, 1e-12)

    tau = 2.3
    delay = 1.0056

    exp_decay_ft = tpl.create_exp_decay_ft(tau, delay)
    assert abs(exp_decay_ft(0.0) - complex(1, 0)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(-157.079632679) - complex(-0.00212781306943, -0.00177022314509)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(314.159265359) - complex(-0.00135979865416, -0.00025744559813)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(0.0383495196971) - complex(0.98816793945543, 0.12571498037097)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(1005.18693069) - complex(0.00030336119799, 0.00030832123791)) < complex(1e-12, 1e-12)
    assert abs(exp_decay_ft(2.45) - complex(-0.13167644011040, -0.11485930412104)) < complex(1e-12, 1e-12)


def test_tpl_biphasic():
    duration = 42.5
    damping = 0.38
    delay = 0.0

    biphasic = create_biphasic(duration, damping, delay)

    assert abs(biphasic(0.5) - 0.0369514993891) < 1e-12
    assert abs(biphasic(6.0) - 0.429120608773) < 1e-12
    assert abs(biphasic(11.5) - 0.751331889557) < 1e-12
    assert abs(biphasic(17.0) - 0.951056516295) < 1e-12
    assert abs(biphasic(22.5) - 0.995734176295) < 1e-12
    assert abs(biphasic(28.0) - 0.878081248084) < 1e-12
    assert abs(biphasic(33.5) - 0.61727822129) < 1e-12
    assert abs(biphasic(39.0) - 0.255842777594) < 1e-12

    duration = 20.6
    damping = 0.88
    delay = 44.5

    biphasic = create_biphasic(duration, damping, delay)
    assert abs(biphasic(45) - 0.0761783767709) < 1e-12
    assert abs(biphasic(50.5) - 0.792579042689) < 1e-12
    assert abs(biphasic(56.0) - 0.983301195364) < 1e-12
    assert abs(biphasic(61.5) - 0.521848255578) < 1e-12


def test_tpl_biphasic_ft():
    duration = 42.5
    damping = 0.38
    delay = 0.0

    biphasic_ft = tpl.create_biphasic_ft(duration,
                                         damping,
                                         delay)

    assert abs(biphasic_ft(-0.0245436926062) - complex(22.7988820989, -3.1173542118)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-0.294524311274) - complex(-1.1286928585, 0.0062065364)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-0.564504929942) - complex(-0.3555288674, -0.0651266984)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-0.83448554861) - complex(-0.0760582312, -0.0917320759)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-1.10446616728) - complex(-0.0021875183, 0.0152324929)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-1.37444678595) - complex(-0.0445794299, 0.0315679060)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-1.64442740461) - complex(-0.0392905913, 0.0014546807)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-1.91440802328) - complex(-0.0259257859, 0.0006440208)) < complex(1e-10, 1e-10)

    duration = 20.6
    damping = 0.88
    delay = 44.5

    biphasic_ft = tpl.create_biphasic_ft(duration,
                                         damping,
                                         delay)

    assert abs(biphasic_ft(-2.20893233456) - complex(0.0355236623, -0.0472345014)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-2.47891295322) - complex(0.0327956407, 0.0088992464)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-2.74889357189) - complex(0.0044665680, 0.0035356123)) < complex(1e-10, 1e-10)
    assert abs(biphasic_ft(-3.01887419056) - complex(0.0192555610, 0.0001958484)) < complex(1e-10, 1e-10)


def test_tpl_delta():
    delta = create_temporal_delta(peak=1.3, delay=0)
    assert(delta(t=0) == 1.3)

    delta = create_temporal_delta(peak=-21.3, delay=2.123)
    assert(delta(t=0) == 0)

    delta = create_temporal_delta(peak=-21.3, delay=-10.02)
    assert(delta(t=-10.02) == -21.3)


def test_tpl_delta_ft():
    delta_ft = tpl.create_delta_ft(delay=1.3)
    assert abs(delta_ft(w=0.5) - complex(0.796083798549055, 0.605186405736039)) < complex(1e-12, 1e-12)

    delta_ft = tpl.create_delta_ft(delay=0.0)
    assert(delta_ft(w=0) == complex(1.0, 0))
