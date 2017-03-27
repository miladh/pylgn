import quantities as pq
import numpy as np
import pytest

import pylgn
from pylgn.stimulus import _check_valid_orient, _convert_to_cartesian, _check_valid_spatial_freq, _check_valid_temporal_freq, create_fullfield_grating, create_fullfield_grating_ft, create_patch_grating, create_patch_grating_ft


def test_check_valid_spatial_freq():
    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    kx, ky = _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec, 
                                       wavenumber=2.67, orient=0)
    assert abs(kx - 2.67/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    
    with pytest.raises(ValueError):
        _check_valid_spatial_freq(kx_vec=k_vec[:-1], ky_vec=k_vec[:-1],
                                  wavenumber=2.67, orient=0)

    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0.0]) / pq.deg
    kx, ky = _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec, wavenumber=5.2, orient=90.)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 5.2/pq.deg) < 1e-12
    
    k_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0.0]) / pq.deg
    kx, ky = _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec, wavenumber=8.2, orient=np.pi/2*pq.rad)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 8.2/pq.deg) < 1e-12
    
    k_vec = np.array([22.530299232133, 6.3202544656492]) / pq.deg
    kx, ky = _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec, wavenumber=23.4, orient=15.67)
    assert abs(kx - 22.530299232133/pq.deg) < 1e-12
    assert abs(ky - 6.3202544656492/pq.deg) < 1e-12
    
    with pytest.raises(AttributeError):
        k_vec = np.array([22.530299232133, 6.3202544656492])
        _check_valid_spatial_freq(kx_vec=k_vec, ky_vec=k_vec, wavenumber=23.4, orient=15.67)
        
    kx_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    ky_vec = np.array([0]) / pq.deg
    kx, ky = _check_valid_spatial_freq(kx_vec=kx_vec, ky_vec=ky_vec, 
                                       wavenumber=2.67, orient=0)
    assert abs(kx - 2.67/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    
    kx_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2, 0]) / pq.deg
    ky_vec = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2]) / pq.deg
    
    with pytest.raises(ValueError):
        kx, ky = _check_valid_spatial_freq(kx_vec=kx_vec, ky_vec=ky_vec, 
                                           wavenumber=2.67, orient=0)
    

def test_check_valid_temporal_freq():
    A = np.array([0, 1, 2, 3, 4, 5])
    v_A = 2.3
    with pytest.raises(AttributeError):
        assert _check_valid_temporal_freq(A, v_A) == v_A

    B = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2]) * pq.Hz
    v_B = 5.2 * pq.Hz
    assert _check_valid_temporal_freq(B, v_B) == v_B

    C = np.array([9.66407446, 7.40204369, 8.47934683, 8.7268378, 9.537069,
                  8.94007828, 6.37876932, 7.84503963, 8.70901142]) * pq.Hz
    v_C = 8.479 * pq.Hz
    with pytest.raises(ValueError):
        assert _check_valid_temporal_freq(C, v_C) == v_C
    
    
def test_check_valid_orient():
    orient = 14.3
    assert _check_valid_orient(orient) == orient

    orient = 352 * pq.deg
    assert _check_valid_orient(orient) == orient
    
    orient = 4.567 * pq.rad
    assert _check_valid_orient(orient) == orient
    
    orient = -182.456 * pq.deg
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient)
        
    orient = 7.4 * pq.rad
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient)
        
    orient = 7.4 * pq.deg
    assert _check_valid_orient(orient) == orient
    
    orient = 7.4 
    assert _check_valid_orient(orient) == orient
    
    orient = 360.0000001
    with pytest.raises(ValueError):
        assert _check_valid_orient(orient) 

    
def test_convert_to_cartesian():
    wavenumber, orient = 1, 0.0  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - 1/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    
    wavenumber, orient = 2.3/pq.deg, 0.0*pq.rad  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - 2.3/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12

    wavenumber, orient = -2.3/pq.deg, 0.0*pq.rad  
    with pytest.warns(UserWarning):
        kx, ky = _convert_to_cartesian(wavenumber, orient) 
    assert abs(kx - 2.3/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    
    wavenumber, orient = 10.345/pq.deg, 90.0*pq.deg  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - 0/pq.deg) < 1e-12
    assert abs(ky - 10.345/pq.deg) < 1e-12
    
    wavenumber, orient = 10.345/pq.deg, np.pi*pq.rad  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - -10.345/pq.deg) < 1e-12
    assert abs(ky - 0/pq.deg) < 1e-12
    
    wavenumber, orient = 3.4/pq.deg, 45.  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - 2.4041630560343/pq.deg) < 1e-12
    assert abs(ky - 2.4041630560343/pq.deg) < 1e-12
    
    wavenumber, orient = 23.4, 15.67  
    kx, ky = _convert_to_cartesian(wavenumber, orient)
    assert abs(kx - 22.530299232133/pq.deg) < 1e-12
    assert abs(ky - 6.3202544656492/pq.deg) < 1e-12
    
    wavenumber, orient = 23.4, -15.67  
    with pytest.raises(ValueError):
        _convert_to_cartesian(wavenumber, orient)


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
            
    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=0,
                                                          wavenumber=-2.345,
                                                          orient=0,
                                                          contrast=1)
    with pytest.warns(UserWarning):
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
    
    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=-3.1,
                                                          wavenumber=4.3,
                                                          orient=0.,
                                                          contrast=1)
    assert (stimulus(w=[-3.1, 3.1]*pq.Hz, kx=4.3/pq.deg, ky=0/pq.deg) == np.array([4*np.pi**3/6.2, 0])).all()
    
    stimulus = pylgn.stimulus.create_fullfield_grating_ft(angular_freq=-3.1,
                                                          wavenumber=4.3,
                                                          orient=0.,
                                                          contrast=1)
    assert (stimulus(w=[-3.1, 3.1]*pq.Hz, kx=[-4.3, 4.3]/pq.deg, ky=0/pq.deg) == np.array([0, 0])).all()
    
    
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
    
    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                       wavenumber=0.7853981633974483,
                                                       orient=0,
                                                       contrast=1.)
    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - 0.587785252292) < 1e-12

    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=-2.945243112740431*pq.kHz,
                                                       wavenumber=0.7853981633974483,
                                                       orient=0,
                                                       contrast=1.)
    assert abs(stimulus(t=2.1*pq.s, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.522498564716) < 1e-12       
    
    stimulus = pylgn.stimulus.create_fullfield_grating(angular_freq=2.356194490192345*pq.kHz,
                                                       wavenumber=-0.39269908169872414,
                                                       orient=0,
                                                       contrast=2.1)
    assert abs(stimulus(t=1.2*pq.ms, x=2.30*pq.deg, y=-2.2*pq.deg) - -0.726845819863) < 1e-12
