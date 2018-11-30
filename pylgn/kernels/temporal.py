import numpy as np
import quantities as pq


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
