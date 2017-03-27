import quantities as pq
import numpy as np

epsilon = 1e-12


def find_nearst(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def heaviside(x):
    return np.where(x < 0, 0.0, 1.0)
    

def kronecker_delta(x, y):
    return np.where(abs(x - y) < epsilon, 1.0, 0.0)


def first_kind_bessel(x):
    from scipy.special import jn 
    x = x.magnitude if isinstance(x, pq.Quantity) else x
    return jn(1, x)


def confluent_hypergeometric(a, b, x):
    from scipy.special import hyp1f1 
    x = x.magnitude if isinstance(x, pq.Quantity) else x
    return hyp1f1(a, b, x)
