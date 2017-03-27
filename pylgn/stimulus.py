"""
stimulus class

"""

from abc import ABC, abstractmethod
import numpy as np
import quantities as pq
import warnings

from .helper import epsilon, heaviside, kronecker_delta, first_kind_bessel, find_nearst


def _check_valid_orient(orient):
    orient = orient.rescale(pq.deg) if isinstance(orient, pq.Quantity) else orient * pq.deg
    if orient < 0*pq.deg or orient > 360*pq.deg:
        raise ValueError("Orientation must be in range [0, 360]", orient)
        
    return orient


def _convert_to_cartesian(wavenumber, orient):
    orient = _check_valid_orient(orient)
    
    if wavenumber < 0:
        warnings.warn("Warning: wavenumber sign is ignored. Use orientation to specify desired direction.")
    wavenumber = abs(wavenumber) if isinstance(wavenumber, pq.Quantity) else abs(wavenumber) / pq.deg
    
    kx = wavenumber * np.cos(orient.rescale("rad"))
    ky = wavenumber * np.sin(orient.rescale("rad"))
    return kx, ky


def _check_valid_spatial_freq(kx_vec, ky_vec, wavenumber, orient):
    kx_vec = kx_vec.rescale(1/pq.deg)
    ky_vec = ky_vec.rescale(1/pq.deg)
    kx_g, ky_g = _convert_to_cartesian(wavenumber, orient)
    
    for k, k_vec in zip([kx_g, ky_g], [kx_vec, ky_vec]):
        if not np.any(abs(k_vec - k) < epsilon):
            raise ValueError("required freq ({}) doesn't exist in array. Use numerical integration instead".format(k))

    return kx_g, ky_g


def _check_valid_temporal_freq(w_vec, w_g):
    w_g = w_g if isinstance(w_g, pq.Quantity) else w_g * w_vec.units
    
    if not np.any(abs(w_vec - w_g) < epsilon):
        raise ValueError("required freq ({}) doesn't exist in array. Use numerical integration instead".format(w_g))
        
    return w_g
    

def create_fullfield_grating(angular_freq=0*pq.Hz, wavenumber=0*pq.deg, 
                             orient=0*pq.deg, contrast=1):
    """
    Full-field grating
    """
    def evaluate(t, x, y,):
        w_g = angular_freq if isinstance(angular_freq, pq.Quantity) else angular_freq * pq.Hz
        kx_g, ky_g = _convert_to_cartesian(wavenumber, orient)
        return contrast * np.cos(kx_g * x + ky_g * y - w_g * t)
    return evaluate

    
def create_fullfield_grating_ft(angular_freq=0*pq.Hz, wavenumber=0*pq.deg, 
                                orient=0*pq.deg, contrast=1):
    def evaluate(w, kx, ky):
        w_g = _check_valid_temporal_freq(w, angular_freq)
        kx_g, ky_g = _check_valid_spatial_freq(kx, ky, wavenumber, orient)

        dw = abs(w.flatten()[1] - w.flatten()[0]) if isinstance(w, np.ndarray) and w.ndim > 0 else 1*pq.Hz
        dk = abs(kx.flatten()[1] - kx.flatten()[0]) if isinstance(kx, np.ndarray) and kx.ndim > 0 else 1/pq.deg

        g_1 = kronecker_delta(kx, kx_g) * kronecker_delta(ky, ky_g) * kronecker_delta(w, w_g)
        g_2 = kronecker_delta(kx, -kx_g) * kronecker_delta(ky, -ky_g) * kronecker_delta(w, -w_g)

        return 4 * np.pi**3 * contrast * (g_1 + g_2) / dw / dk**2
        
    return evaluate


def create_patch_grating(angular_freq=0*pq.Hz, wavenumber=0*pq.deg, 
                         orient=0*pq.deg, contrast=1, mask_size=1*pq.deg):
    mask_size = mask_size if isinstance(mask_size, pq.Quantity) else mask_size * pq.deg
    
    def evaluate(t, x, y):
        # TODO: write test
        w_g = angular_freq if isinstance(angular_freq, pq.Quantity) else angular_freq * pq.Hz
        kx_g, ky_g = _convert_to_cartesian(wavenumber, orient)
        r = np.sqrt(x**2 + y**2)
        return contrast * np.cos(kx_g*x + ky_g*y - w_g*t) * (1 - heaviside(r - mask_size*0.5))
        
    return evaluate
    

def create_patch_grating_ft(angular_freq=0*pq.Hz, wavenumber=0*pq.deg, 
                            orient=0*pq.deg, contrast=1, mask_size=1*pq.deg):
    # TODO: write test
    def evaluate(w, kx, ky):
        w_g = _check_valid_temporal_freq(w, angular_freq)
        kx_g, ky_g = _check_valid_spatial_freq(kx, ky, wavenumber, orient)
        
        dw = abs(w.flatten()[1] - w.flatten()[0]) if isinstance(w, np.ndarray) and w.ndim > 0 else 1*pq.Hz
      
        factor = contrast * np.pi**2 * mask_size**2 / dw / 4
        dk_1 = np.sqrt((kx - kx_g)**2 + (ky - ky_g)**2)
        dk_2 = np.sqrt((kx + kx_g)**2 + (ky + ky_g)**2)
        arg_1 = dk_1 * mask_size * 0.5
        arg_2 = dk_2 * mask_size * 0.5
        
        term_1 = np.where(arg_1 == 0, 1, 2 * first_kind_bessel(arg_1) / arg_1) 
        term_2 = np.where(arg_2 == 0, 1, 2 * first_kind_bessel(arg_2) / arg_2) 

        return factor * (term_1*kronecker_delta(w, w_g) + term_2*kronecker_delta(w, -w_g))
        
    return evaluate
