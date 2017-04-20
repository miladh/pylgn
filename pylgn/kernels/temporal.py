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


def create_biphasic(phase_duration=43*pq.ms, damping_factor=0.38, delay=0*pq.ms):
    """
    Create Biphasic closure

    Parameters
    ----------
    phase_duration : float/quantity scalar
        Delay
    damping_factor : float
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
        sin_term = np.sin(np.pi/phase_duration * delta_t)

        condition1 = delta_t < 0
        condition2 = np.logical_and(0 <= delta_t, delta_t <= phase_duration)
        condition3 = np.logical_and(phase_duration < delta_t,
                                    delta_t <= 2*phase_duration)
        condition4 = delta_t > 2*phase_duration

        r = np.where(condition1, 0.0, 0.0)
        r = np.where(condition2, sin_term, r)
        r = np.where(condition3, damping_factor*sin_term, r)
        r = np.where(condition4, 0.0, r)

        return r

    return evaluate


def create_biphasic_ft(phase_duration, damping_factor, delay=0*pq.ms):
    """
    Create Fourier transformed Biphasic closure

    Parameters
    ----------
    phase_duration : float/quantity scalar
        Delay
    damping_factor : float
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
        factor = np.pi * phase_duration / (np.pi**2 - phase_duration**2 * w**2)
        exp_term = np.exp(1j * phase_duration * w)
        term1 = 1. + (1. - damping_factor) * exp_term
        term2 = damping_factor * np.exp(1j * phase_duration * 2.*w)

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
