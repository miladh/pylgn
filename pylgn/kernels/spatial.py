import numpy as np
import quantities as pq


def create_delta_ft(shift_x=0*pq.deg, shift_y=0*pq.deg):
    """
    Create delta_ft closure

    Parameters
    ----------
    shift_x : float/quantity scalar
        Shift in x-direction
    shift_y : float/quantity scalar
        Shift in y-direction

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(kx, ky):
        """
        Evaluates the delta_ft function

        Parameters
        ----------
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return np.exp(-1j * (kx * shift_x + ky * shift_y))

    return evaluate


def create_gauss_ft(A=1, a=0.62*pq.deg,
                    dx=0.*pq.deg, dy=0.*pq.deg):
    """
    Create Fourier transformed
    Gaussian function closure.

    Parameters
    ----------
    A : float
        peak value
    a : float/quantity scalar
        Width
    dx: float/quantity scalar
        shift in x-direction
    dy: float/quantity scalar
        shift in y-direction

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(kx, ky):
        """
        Evaluates the Fourier transform of gauss function

        Parameters
        ----------
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        k2 = kx**2 + ky**2
        return A * np.exp(-a**2 * k2/4.) * np.exp(-1j * (kx*dx + ky*dy))

    return evaluate


def create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg,
                  dx=0.*pq.deg, dy=0.*pq.deg):
    """
    Create Fourier transformed
    difference of Gaussian function closure

    Parameters
    ----------
    A : float
        Center peak value
    a : float/quantity scalar
        Center width
    B : float
        Surround peak value
    b : float/quantity scalar
        Surround width
    dx: float/quantity scalar
        shift in x-direction
    dy: float/quantity scalar
        shift in y-direction

    Returns
    -------
    out : function
        Evaluate function
    """
    center = create_gauss_ft(A, a, dx, dy)
    surround = create_gauss_ft(B, b, dx, dy)

    def evaluate(kx, ky):
        """
        Evaluates the Fourier transform of
        difference of Gaussian function

        Parameters
        ----------
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return center(kx, ky) - surround(kx, ky)

    return evaluate
