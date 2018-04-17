from abc import ABC, abstractmethod
import numpy as np
import quantities as pq
import warnings

from .helper import epsilon, heaviside, kronecker_delta, first_kind_bessel


def _check_valid_orient(orient):
    orient = orient.rescale(pq.deg) if isinstance(orient, pq.Quantity) else orient * pq.deg
    if orient < 0*pq.deg or orient > 360*pq.deg:
        raise ValueError("Orientation must be in range [0, 360]", orient)


def _check_valid_spatial_freq(kx_vec, ky_vec, kx_g, ky_g):
    kx_g = kx_g if isinstance(kx_g, pq.Quantity) else kx_g * kx_vec.units
    ky_g = ky_g if isinstance(ky_g, pq.Quantity) else ky_g * ky_vec.units

    for k, k_vec in zip([kx_g, ky_g], [kx_vec, ky_vec]):
        if not np.any(abs(k_vec - k) < epsilon) and not np.any(abs(k_vec + k) < epsilon):
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


def create_fullfield_grating(angular_freq=0*pq.Hz, wavenumber=0/pq.deg,
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
    out : callable
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


def create_fullfield_grating_ft(angular_freq=0*pq.Hz, wavenumber=0/pq.deg,
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
    out : callable
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


def create_patch_grating(angular_freq=0*pq.Hz, wavenumber=0/pq.deg,
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
    out : callable
        Evaluate function


    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    """
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


def create_patch_grating_ft(angular_freq=0*pq.Hz, wavenumber=0/pq.deg,
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
    out : callable
        Evaluate function

    Notes
    -----
    Both angular_freq and wavenumber are positive numbers.
    Use orientation to specify desired direction.
    The combination of angular_freq, wavenumber, and orient should
    give w, kx, and ky that exist in function arguments in evaluate function.
    """
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
        _check_valid_spatial_freq(kx, ky, kx_g, ky_g)  # TODO: is this necessary?
        _check_valid_temporal_freq(w, w_g)

        dw = abs(w.flatten()[1] - w.flatten()[0]) if isinstance(w, np.ndarray) and w.ndim > 0 else 1*pq.Hz

        factor = contrast * np.pi**2 * patch_diameter**2 / dw / 4
        dk_1 = np.sqrt((kx - kx_g)**2 + (ky - ky_g)**2)
        dk_2 = np.sqrt((kx + kx_g)**2 + (ky + ky_g)**2)
        arg_1 = dk_1 * patch_diameter * 0.5
        arg_2 = dk_2 * patch_diameter * 0.5

        # NOTE: a runtime warning will arise since np.where evaluates the function
        # in all points, including 0, and then proper final results are selected
        with np.errstate(invalid='ignore'):
            term_1 = np.where(arg_1 == 0, 1, 2 * first_kind_bessel(arg_1) / arg_1)
            term_2 = np.where(arg_2 == 0, 1, 2 * first_kind_bessel(arg_2) / arg_2)

        return factor.magnitude * (term_1*kronecker_delta(w, w_g) + term_2*kronecker_delta(w, -w_g))

    return evaluate


def create_flashing_spot(contrast=1, patch_diameter=1*pq.deg,
                         delay=0*pq.ms, duration=0*pq.ms):
    """
    Create flashing spot


    Parameters
    ----------
    contrast : float
        Contrast value
    patch_diameter : float/quantity scalar
        Patch size
    delay : float/quantity scalar
        onset time
    duration : float/quantity scalar
        duration of flashing spot

    Returns
    -------
    out : callable
        Evaluate function

    """
    if delay < 0:
        raise ValueError("delay must be a postive number: ".format(delay))

    if duration < 0:
        raise ValueError("duration must be a postive number: ".format(duration))

    patch_diameter = patch_diameter if isinstance(patch_diameter, pq.Quantity) else patch_diameter * pq.deg

    delay = delay if isinstance(delay, pq.Quantity) else delay * pq.ms
    duration = duration if isinstance(duration, pq.Quantity) else duration * pq.ms

    def evaluate(t, x, y):
        """
        Evaluates flashing spot

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
        r = np.sqrt(x**2 + y**2)

        return contrast * (1 - heaviside(r - patch_diameter*0.5)) * (heaviside(t - delay) - heaviside(t - delay - duration))

    return evaluate


def create_flashing_spot_ft(contrast=1, patch_diameter=1*pq.deg,
                            delay=0*pq.ms, duration=0*pq.ms):
    # TODO write tests
    """
    Create Fourier transformed flashing spot


    Parameters
    ----------
    contrast : float
        Contrast value
    patch_diameter : float/quantity scalar
        Patch size
    delay : float/quantity scalar
        onset time
    duration : float/quantity scalar
        duration of flashing spot

    Returns
    -------
    out : callable
        Evaluate function

    """
    if delay < 0:
        raise ValueError("delay must be a postive number: ".format(delay))

    if duration < 0:
        raise ValueError("duration must be a postive number: ".format(duration))

    delay = delay if isinstance(delay, pq.Quantity) else delay * pq.ms
    duration = duration if isinstance(duration, pq.Quantity) else duration * pq.ms

    def evaluate(w, kx, ky):
        """
        Evaluates Fourier transformed flashing spot function

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
        factor = contrast * patch_diameter.magnitude**2 * np.pi / 4
        arg = np.sqrt(kx**2 + ky**2) * patch_diameter * 0.5

        # NOTE: a runtime warning will arise since np.where evaluates the function
        # in all points, including 0, and then proper final results are selected
        with np.errstate(invalid='ignore'):
            spatial = np.where(arg == 0, 1, 2 * first_kind_bessel(arg) / arg)

        half_duration = duration.rescale(1/w.units) / 2
        temporal = np.sinc(w * half_duration / np.pi) * np.exp(1j * w * (delay+half_duration).rescale(1/w.units))
        return factor * duration.magnitude * temporal * spatial

    return evaluate


def create_natural_image(filenames, delay=0*pq.ms, duration=0*pq.ms):
    """
    Creates natural image stimulus

    Parameters
    ----------
    filenames : list/string
        path to image(s)
    delay : quantity scalar
        Onset time
    duration : quantity scalar

    Returns
    -------
    out : callable
        Evaluate function
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    if delay < 0:
        raise ValueError("delay must be a postive number: ".format(delay))

    if duration < 0:
        raise ValueError("duration must be a postive number: ".format(duration))

    delay = delay if isinstance(delay, pq.Quantity) else delay * pq.ms
    duration = duration if isinstance(duration, pq.Quantity) else duration * pq.ms

    def evaluate(t, x, y):
        # TODO: fix normalization
        """
        converts image to numpy array

        Parameters
        ----------
        t : quantity scalar
        x : quantity scalar
        y : quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        from PIL import Image

        Nt = t.shape[0]
        Nx = x.shape[2]
        Ny = y.shape[1]
        stim = np.zeros([Nt, Nx, Ny])

        for i, filename in enumerate(filenames):
            im = Image.open(filename).convert("L").transpose(Image.FLIP_TOP_BOTTOM)
            t_start = delay + i * (delay + duration)
            t_stop = (i+1) * (duration + delay)
            stim += np.array(im.resize((Ny, Nx))) * (heaviside(t - t_start) - heaviside(t - t_stop))

        if stim.max() - stim.min() != 0:
            stim = 2 * ((stim - stim.min()) / (stim.max() - stim.min())) - 1
        return stim

    return evaluate


def create_natural_movie(filename):
    """
    Creates natural movie stimulus

    Parameters
    ----------
    filename : string
        path to gif

    Returns
    -------
    out : callable
        Evaluate function
    """
    if not filename.endswith('.gif'):
        raise NameError("unsupported fomat. use '.gif'", filename)

    def evaluate(t, x, y):
        """
        converts movie to numpy array

        Parameters
        ----------
        t : quantity scalar
        x : quantity scalar
        y : quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        from PIL import Image
        im = Image.open(filename)
        duration = im.info["duration"]*pq.ms if im.info["duration"] is not 0 else 30*pq.ms

        Nt = t.shape[0]
        Nx = x.shape[2]
        Ny = y.shape[1]

        stim = np.zeros([Nt, Ny, Nx])
        t_map = (t.flatten().rescale("ms") / duration).astype(int)
        t_map = t_map[1:] - t_map[:-1]
        for i, ti in enumerate(t_map):
            try:
                im.seek(im.tell()+ti)
            except EOFError:
                break
            frame = im.convert("L").transpose(Image.FLIP_TOP_BOTTOM).resize((Ny, Nx))
            stim[i, :, :] = np.array(frame)
            stim[i, :, :] = 2 * ((stim[i, :, :] - stim[i, :, :].min()) / (stim[i, :, :].max() - stim[i, :, :].min())) - 1

        return stim

    return evaluate
