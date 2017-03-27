from pylgn.helper import*
import quantities as pq


def test_heaviside():
    assert heaviside(-1) == 0
    assert heaviside(0.0) == 1
    assert heaviside(2.2) == 1
    assert heaviside(-100.2) == 0
    assert heaviside(1e-6) == 1
    
    assert heaviside(-1.3 * pq.s) == 0
    assert heaviside(0.0 * pq.m) == 1
    assert heaviside(1e-6 * pq.m) == 1
    
    assert (heaviside(np.array([-2.3, 5.6, 0.0, 100.2])) == np.array([0, 1, 1, 1])).all()
    assert (heaviside(np.array([-2.3, 5.6, 0.0, 100.2])*pq.s) == np.array([0, 1, 1, 1])).all()
    
    
def test_kronecker_delta():
    
    assert(kronecker_delta(0, 0) == 1)
    assert(kronecker_delta(-1e6, -1e6) == 1)
    assert(kronecker_delta(3.456, 3.456) == 1)
    assert(kronecker_delta(2.1, -2.1) == 0)
    assert(kronecker_delta(1., 3.) == 0)
    assert(kronecker_delta(-1230., -50.) == 0)
    
    assert(kronecker_delta(10*pq.s, 0*pq.s) == 0)
    assert(kronecker_delta(2.1*pq.s, 2.1*pq.s) == 1)
    
    A = np.array([-2.3, 5.6, 0.0, 100.2])
    B = np.array([2.3, 5.6, 10.0, 10.2])
    assert (kronecker_delta(A, B) == np.array([0, 1, 0, 0])).all()
    
    A = np.array([-2.3, 5.6, 0.0, 100.2]) * pq.s
    B = np.array([2.3, 5.6, 10.0, 10.2]) * pq.s
    assert (kronecker_delta(A, B) == np.array([0, 1, 0, 0])).all()
    
        
def test_first_kind_bessel():
    assert(first_kind_bessel(0.0) == 0)
    assert(first_kind_bessel(0.1) - 0.049937526036241998 < 1e-12)
    assert(first_kind_bessel(5.0) + 0.32757913759146522 < 1e-12)
    assert(first_kind_bessel(7.01558666981562) < 1e-12)
    assert(first_kind_bessel(-7.01558666981562) < 1e-12)
    assert(first_kind_bessel(3.83170597020751) < 1e-12)


def test_find_nearst():
    A = np.array([0, 1, 2, 3, 4, 5])
    v_A = 2.3
    assert find_nearst(A, v_A) == 2

    B = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2])
    v_B = -2.4
    assert find_nearst(B, v_B) == -2.3

    C = np.array([9.66407446, 7.40204369, 8.47934683, 8.7268378, 9.537069,
                  8.94007828, 6.37876932, 7.84503963, 8.70901142])
    v_C = 7.5
    assert find_nearst(C, v_C) == 7.40204369

    D = np.array([4.20844744, 5.44088512, -1.44998235, 1.8764609, 
                  -2.22633141, 0.33623971, 7.23507673])
    v_D = 0.0
    assert find_nearst(D, v_D) == 0.33623971
