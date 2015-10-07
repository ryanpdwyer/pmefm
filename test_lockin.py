from __future__ import division, absolute_import, print_function
from numpy.testing import assert_array_almost_equal
import numpy as np

from lockin import FIRStateLock

def test_FIRStateLock():
    t1 = np.arange(128, dtype=np.float64)
    t2 = np.arange(128, 256, dtype=np.float64)
    x1 = np.sin(np.pi/2*t1) * np.exp(-t1/200000.)
    x2 = np.sin(np.pi/2*t2) * np.exp(-t2/200000.)
    f0 = 0.25
    phi0 = np.pi/2
    fir = np.array([ -2.24576405e-21,   7.38375885e-07,   4.47985274e-06,
         1.04499910e-05,   9.53493909e-06,  -2.11536301e-05,
        -1.21463857e-04,  -3.45121941e-04,  -7.45930068e-04,
        -1.35036096e-03,  -2.11774624e-03,  -2.89530921e-03,
        -3.38156003e-03,  -3.11543482e-03,  -1.50742755e-03,
         2.07858743e-03,   8.19742309e-03,   1.71690331e-02,
         2.89335254e-02,   4.29551745e-02,   5.82110859e-02,
         7.32838205e-02,   8.65526725e-02,   9.64518343e-02,
         1.01743149e-01,   1.01743149e-01,   9.64518343e-02,
         8.65526725e-02,   7.32838205e-02,   5.82110859e-02,
         4.29551745e-02,   2.89335254e-02,   1.71690331e-02,
         8.19742309e-03,   2.07858743e-03,  -1.50742755e-03,
        -3.11543482e-03,  -3.38156003e-03,  -2.89530921e-03,
        -2.11774624e-03,  -1.35036096e-03,  -7.45930068e-04,
        -3.45121941e-04,  -1.21463857e-04,  -2.11536301e-05,
         9.53493909e-06,   1.04499910e-05,   4.47985274e-06,
         7.38375885e-07,  -2.24576405e-21])
    firstate = FIRStateLock(fir, 2, f0, phi0)
    firstate.filt(x1)
    firstate.filt(x2)
    t_out = firstate.get_t()
    A_out = abs(firstate.z_out)
    A_expected = np.exp(-t_out / 200000.)
    assert_array_almost_equal(A_out, A_expected, decimal=5)


def test_FIRStateLock_indices():
    x = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    h = np.array([0.25, 0.25, 0.25, 0.25])/2.
    dec = 2
    f0 = 0.
    fs = 1.
    firstate = FIRStateLock(h, 2, f0, 0., fs=1.)
    firstate.filt(x)
    exp_z = np.array([1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5])
    assert_array_almost_equal(firstate.z_out, exp_z)

