import numpy as np
import quantities as pq
from pylgn.helper import kronecker_delta, heaviside


def create_temporal_delta(peak, delay=0*pq.ms):
    """
    Create delta closure

    Parameters
    ----------
    peak : float
        Peak value
    delay : float/quantity scalar
        Delay

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(t):
        """
        Evaluates the delta function

        Parameters
        ----------
        t : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return peak * kronecker_delta(t, delay)

    return evaluate


def create_biphasic(phase=42.5*pq.ms, damping=0.38, delay=0*pq.ms):
    """
    Create Biphasic closure

    Parameters
    ----------
    phase : float/quantity scalar
        Delay
    damping : float
        Damping factor
    delay : float/quantity scalar
        Delay

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(t):
        """
        Evaluates the Biphasic function

        Parameters
        ----------
        t : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        delta_t = t - delay
        sin_term = np.sin(np.pi/phase * delta_t)

        condition1 = delta_t < 0
        condition2 = np.logical_and(0 <= delta_t, delta_t <= phase)
        condition3 = np.logical_and(phase < delta_t,
                                    delta_t <= 2*phase)
        condition4 = delta_t > 2*phase

        r = np.where(condition1, 0.0, 0.0)
        r = np.where(condition2, sin_term, r)
        r = np.where(condition3, damping*sin_term, r)
        r = np.where(condition4, 0.0, r)

        return r

    return evaluate


def create_exp_decay(tau, delay):
    """
    Create exponential decay closure

    Parameters
    ----------
    tau : float/quantity scalar
        Time constant
    delay : float/quantity scalar
        Delay

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(t):
        """
        Evaluates the exponential decay function

        Parameters
        ----------
        t : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return np.exp(-(t - delay) / tau) / tau * heaviside(t - delay)

    return evaluate


def create_spatial_delta(peak, shift_x=0*pq.deg, shift_y=0*pq.deg):
    """
    Create delta closure

    Parameters
    ----------
    peak : float
        Peak value
    shift_x : float/quantity scalar
        Shift in x-direction
    shift_y : float/quantity scalar
        Shift in y-direction

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(x, y):
        """
        Evaluates the delta function

        Parameters
        ----------
        x : float/quantity scalar
        y : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return peak * kronecker_delta(x, shift_x) * kronecker_delta(y, shift_y)

    return evaluate


def create_gauss(A=1, a=0.62*pq.deg,
                 dx=0.*pq.deg, dy=0.*pq.deg):
    """
    Create Gaussian function closure

    Parameters
    ----------
    A : float
        Peak value
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
    def evaluate(x, y):
        """
        Evaluates the gauss function

        Parameters
        ----------
        x : float/quantity scalar
        y : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        r2 = (x-dx)**2 + (y-dy)**2
        return A / a**2 / np.pi * np.exp(-r2/a**2)

    return evaluate


def create_dog(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg,
               dx=0*pq.deg, dy=0*pq.deg):
    """
    Create difference of Gaussian function closure

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
    center = create_gauss(A, a, dx, dy)
    surround = create_gauss(B, b, dx, dy)

    def evaluate(x, y):
        """
        Evaluates the difference of Gaussian function

        Parameters
        ----------
        x : float/quantity scalar
        y : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return center(x, y) - surround(x, y)

    return evaluate
