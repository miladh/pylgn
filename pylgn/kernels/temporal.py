import numpy as np
import quantities as pq

from ..helper import kronecker_delta, heaviside


def create_delta(peak, delay=0*pq.ms):
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


def create_delta_ft(delay=0*pq.ms):
    """
    Create Fourier transform delta closure

    Parameters
    ----------
    delay : float/quantity scalar
        Delay

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(w):
        """
        Evaluates the Fourier transformed
        delta function

        Parameters
        ----------
        w : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return np.exp(1j * delay * w)

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


def create_biphasic_ft(phase=43*pq.ms, damping=0.38, delay=0*pq.ms):
    """
    Create Fourier transformed Biphasic closure

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
    def evaluate(w):
        """
        Evaluates the Fourier transformed Biphasic function

        Parameters
        ----------
        w : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        factor = np.pi * phase / (np.pi**2 - phase**2 * w**2)
        exp_term = np.exp(1j * phase * w)
        term1 = 1. + (1. - damping) * exp_term
        term2 = damping * np.exp(1j * phase * 2.*w)

        factor = factor.magnitude if isinstance(factor, pq.Quantity) else factor  # TODO: hack
        return factor * np.exp(1j * delay * w) * (term1 - term2)

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


def create_exp_decay_ft(tau, delay):
    """
    Create Fourier transformed
    exponential decay closure

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
    def evaluate(w):
        """
        Evaluates the Fourier transformed
        exponential decay function

        Parameters
        ----------
        w : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        return np.exp(1j * w * delay) / (1 - 1j * w * tau)

    return evaluate


def create_doe_ft(tau_cen, tau_sur, delay):
    def evaluate(w):
        exp_factor = np.exp(1j * w * delay)
        center = exp_factor / (1. - tau_cen * w * 1j)**2
        surround = exp_factor / (1. - tau_sur * w * 1j)**2

        return center - surround

    return evaluate


def create_poly_exp_decay_ft(tau, delay):
    def evaluate(w):
        return np.exp(1j * w * delay) * -1j * w.magnitude / (1 - 1j * w * tau)**3

    return evaluate
