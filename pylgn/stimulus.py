from abc import ABC, abstractmethod
import numpy as np
import quantities as pq
import warnings

from .helper import epsilon, heaviside, kronecker_delta, first_kind_bessel, find_nearst

# TODO write doc


def _check_valid_orient(orient):
    orient = orient.rescale(pq.deg) if isinstance(orient, pq.Quantity) else orient * pq.deg
    if orient < 0*pq.deg or orient > 360*pq.deg:
        raise ValueError("Orientation must be in range [0, 360]", orient)


def _check_valid_spatial_freq(kx_vec, ky_vec, kx_g, ky_g):
    kx_g = kx_g if isinstance(kx_g, pq.Quantity) else kx_g * kx_vec.units
    ky_g = ky_g if isinstance(ky_g, pq.Quantity) else ky_g * ky_vec.units

    for k, k_vec in zip([kx_g, ky_g], [kx_vec, ky_vec]):
        if not np.any(abs(k_vec - k) < epsilon):
            raise ValueError("required freq ({}) doesn't exist in array. Use numerical integration instead".format(k))


def _check_valid_temporal_freq(w_vec, w_g):
    w_g = w_g if isinstance(w_g, pq.Quantity) else w_g * w_vec.units

    if not np.any(abs(w_vec - w_g) < epsilon):
        raise ValueError("required freq ({}) doesn't exist in array. Use numerical integration instead".format(w_g))


def _convert_to_cartesian(angular_freq, wavenumber, orient):
    _check_valid_orient(orient)

    if angular_freq < 0:
        warnings.warn("Warning: angular freq sign is ignored. Use orientation to specify desired direction.")
    if wavenumber < 0:
        warnings.warn("Warning: wavenumber sign is ignored. Use orientation to specify desired direction.")

    orient = orient.rescale(pq.deg) if isinstance(orient, pq.Quantity) else orient * pq.deg
    angular_freq = abs(angular_freq) if isinstance(angular_freq, pq.Quantity) else abs(angular_freq) * pq.Hz
    wavenumber = abs(wavenumber) if isinstance(wavenumber, pq.Quantity) else abs(wavenumber) / pq.deg

    if orient.rescale("deg") > 180*pq.deg:
        angular_freq *= -1
        orient -= 180*pq.deg

    kx = wavenumber * np.cos(orient.rescale("rad"))
    ky = wavenumber * np.sin(orient.rescale("rad"))
    return angular_freq, kx, ky


def create_fullfield_grating(angular_freq=0*pq.Hz, wavenumber=0*pq.deg,
                             orient=0*pq.deg, contrast=1):
    """
    Create full-field grating


    Parameters
    ----------
    angular_freq : float
        Angular frequency (positive number)
    wavenumber : float/quantity scalar
        Wavenumber (positive number)
    orient : float/quantity scalar
        Orientation
    contrast : float
        Contrast value


    Returns
    -------
    out : function
        Evaluate function


    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    """
    w_g, kx_g, ky_g = _convert_to_cartesian(angular_freq, wavenumber, orient)

    def evaluate(t, x, y):
        """
        Evaluates full-field grating

        Parameters
        ----------
        t : float/quantity scalar
        x : float/quantity scalar
        y : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return contrast * np.cos(kx_g * x + ky_g * y - w_g * t)
    return evaluate


def create_fullfield_grating_ft(angular_freq=0*pq.Hz, wavenumber=0*pq.deg,
                                orient=0*pq.deg, contrast=1):
    """
    Create Fourier transformed full-field grating

    Parameters
    ----------
    angular_freq : float
        Angular frequency (positive number)
    wavenumber : float/quantity scalar
        Wavenumber (positive number)
    orient : float/quantity scalar
        Orientation
    contrast : float
        Contrast value


    Returns
    -------
    out : function
        Evaluate function

    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    The combination of angular_freq, wavenumber, and orient should
    give w, kx, and ky that exist in function arguments in evaluate function.
    """
    w_g, kx_g, ky_g = _convert_to_cartesian(angular_freq, wavenumber, orient)

    def evaluate(w, kx, ky):
        """
        Evaluates Fourier transformed full-field grating

        Parameters
        ----------
        w : float/quantity scalar
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        _check_valid_spatial_freq(kx, ky, kx_g, ky_g)
        _check_valid_temporal_freq(w, w_g)

        dw = abs(w.flatten()[1] - w.flatten()[0]) if isinstance(w, np.ndarray) and w.ndim > 0 else 1*pq.Hz
        dkx = abs(kx.flatten()[1] - kx.flatten()[0]) if isinstance(kx, np.ndarray) and kx.ndim > 0 else 1/pq.deg
        dky = abs(ky.flatten()[1] - ky.flatten()[0]) if isinstance(ky, np.ndarray) and ky.ndim > 0 else 1/pq.deg

        g_1 = kronecker_delta(kx, kx_g) * kronecker_delta(ky, ky_g) * kronecker_delta(w, w_g)
        g_2 = kronecker_delta(kx, -kx_g) * kronecker_delta(ky, -ky_g) * kronecker_delta(w, -w_g)

        return 4 * np.pi**3 * contrast * (g_1 + g_2) / dw.magnitude / dkx.magnitude / dky.magnitude

    return evaluate


def create_patch_grating(angular_freq=0*pq.Hz, wavenumber=0*pq.deg,
                         orient=0*pq.deg, contrast=1, patch_diameter=1*pq.deg):
    """
    Create patch grating

    Parameters
    ----------
    angular_freq : float
        Angular frequency (positive number)
    wavenumber : float/quantity scalar
        Wavenumber (positive number)
    orient : float/quantity scalar
        Orientation
    contrast : float
        Contrast value
    patch_diameter : float/quantity scalar
        Patch size


    Returns
    -------
    out : function
        Evaluate function


    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    """
    # TODO: write test
    w_g, kx_g, ky_g = _convert_to_cartesian(angular_freq, wavenumber, orient)
    patch_diameter = patch_diameter if isinstance(patch_diameter, pq.Quantity) else patch_diameter * pq.deg

    def evaluate(t, x, y):
        """
        Evaluates patch grating function

        Parameters
        ----------
        w : float/quantity scalar
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        r = np.sqrt(x**2 + y**2)
        return contrast * np.cos(kx_g*x + ky_g*y - w_g*t) * (1 - heaviside(r - patch_diameter*0.5))

    return evaluate


def create_patch_grating_ft(angular_freq=0*pq.Hz, wavenumber=0*pq.deg,
                            orient=0*pq.deg, contrast=1, patch_diameter=1*pq.deg):
    """
    Create Fourier transformed patch grating

    Parameters
    ----------
    angular_freq : float
        Angular frequency (positive number)
    wavenumber : float/quantity scalar
        Wavenumber (positive number)
    orient : float/quantity scalar
        Orientation
    contrast : float
        Contrast value
    patch_diameter : float/quantity scalar
        Patch size


    Returns
    -------
    out : function
        Evaluate function

    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    The combination of angular_freq, wavenumber, and orient should
    give w, kx, and ky that exist in function arguments in evaluate function.
    """
    # TODO: write test
    w_g, kx_g, ky_g = _convert_to_cartesian(angular_freq, wavenumber, orient)
    patch_diameter = patch_diameter if isinstance(patch_diameter, pq.Quantity) else patch_diameter * pq.deg

    def evaluate(w, kx, ky):
        """
        Evaluates Fourier transformed patch grating function

        Parameters
        ----------
        w : float/quantity scalar
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        _check_valid_spatial_freq(kx, ky, kx_g, ky_g)
        _check_valid_temporal_freq(w, w_g)

        dw = abs(w.flatten()[1] - w.flatten()[0]) if isinstance(w, np.ndarray) and w.ndim > 0 else 1*pq.Hz

        factor = contrast * np.pi**2 * patch_diameter**2 / dw / 4
        dk_1 = np.sqrt((kx - kx_g)**2 + (ky - ky_g)**2)
        dk_2 = np.sqrt((kx + kx_g)**2 + (ky + ky_g)**2)
        arg_1 = dk_1 * patch_diameter * 0.5
        arg_2 = dk_2 * patch_diameter * 0.5

        term_1 = np.where(arg_1 == 0, 1, 2 * first_kind_bessel(arg_1) / arg_1)
        term_2 = np.where(arg_2 == 0, 1, 2 * first_kind_bessel(arg_2) / arg_2)

        return factor.magnitude * (term_1*kronecker_delta(w, w_g) + term_2*kronecker_delta(w, -w_g))

    return evaluate
