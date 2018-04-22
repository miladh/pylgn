import numpy as np
import quantities as pq


def heaviside_nonlinearity(rates):
    """
    Heaviside static nonlinearity

    Parameters
    ----------
    rates : quantity array
    
    Returns
    -------
    out : quantity array
    """
    return rates.clip(min=0*rates.units)
    

def scale_rates(rates, target_population_rate=60.*pq.Hz):
    """
    Scales the rates to match a target 
    mean population rate

    Parameters
    ----------
    rates : quantity array
    
    target_population_rate : quantity array
    
    Returns
    -------
    out : quantity array
        scaled population activity
    """
    if target_population_rate < 0.0:
        raise ValueError("target population rate should be positive")
    if not rates.units == target_population_rate.units:
        raise AttributeError("rates (unit: {}) and target rate (unit: {}) must have same unit".format(rates.units, target_population_rate.units))
        
    rates *= target_population_rate / np.mean(rates)
    return rates


def nonstationary_poisson(times, rate):
    """
    Non-stationary Poisson process
    """
    n_exp = (rate.max() * (times.max()-times.min())).simplified
    t_events = np.sort(np.random.uniform(times.min(), 
                                         times.max(), 
                                         np.random.poisson(n_exp)))
    mask = np.digitize(t_events, times)
    ratio = rate[mask] / rate.max()
    mask = np.random.uniform(0., 1., len(ratio)) < ratio
    return t_events[mask]


def generate_spike_train(data, times):
    """
    Generates spike trains based on rates
    for each neuron

    Parameters
    ----------
    data : quantity array
        rates (N_spikes X Nx X Ny)
    
    times : quantity array
    
    Returns
    -------
    spike_trains : array_like
        spike arrays (Nx x Ny x N_spikes)
    """
    Nx, Ny = data.shape[1:]
    spike_trains = np.zeros([Nx, Ny], dtype=object)
    
    for i in range(Nx):
        for j in range(Ny):
            spike_trains[i, j] = nonstationary_poisson(times, data[:, i, j]) * times.units
            
    return spike_trains
