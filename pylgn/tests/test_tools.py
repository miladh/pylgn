from pylgn.tools import*
import quantities as pq
import pytest


def test_heaviside_nonlinearity():
    rates = np.array([0, 1, 2, 3, 4.5])*pq.Hz
    assert (heaviside_nonlinearity(rates) == rates).all()

    rates = np.array([0, -1, -22, -3.1, 1])*pq.Hz
    assert (heaviside_nonlinearity(rates) == np.array([0, 0, 0, 0, 1])*pq.Hz).all()
    
    rates = np.array([0, 1, 2, 3, 4.5])
    with pytest.raises(AttributeError):
        heaviside_nonlinearity(rates)
    
    rates = np.array([0, 1, 2, 3, 4.5])*pq.s
    rates_nonlinear = heaviside_nonlinearity(rates)
    assert (rates_nonlinear == rates).all()
    assert rates_nonlinear.units == pq.s


def test_scale_rates():
    rates = np.array([0, 1, 2, 3, 4])*pq.Hz
    scaled_rates = scale_rates(rates, 10*pq.Hz)
    assert (5.*np.array([0, 1, 2, 3, 4])*pq.Hz == scaled_rates).all()

    scaled_rates = scale_rates(rates, 1*pq.Hz)
    assert (0.5*np.array([0, 1, 2, 3, 4])*pq.Hz == scaled_rates).all()

    with pytest.raises(AttributeError):
        scale_rates(rates, 15*pq.s)

    scaled_rates = scale_rates(rates, 15/pq.s)
    assert (7.5*np.array([0, 1, 2, 3, 4])*pq.Hz == scaled_rates).all()
    
    scaled_rates = scale_rates(rates)
    assert (30*np.array([0, 1, 2, 3, 4])*pq.Hz == scaled_rates).all()
    
    with pytest.raises(ValueError):
        scale_rates(rates, -15*pq.Hz)
    
    with pytest.raises(ValueError):
        scale_rates(rates, -15)
