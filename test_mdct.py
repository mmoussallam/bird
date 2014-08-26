# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_almost_equal
from mdct_tools import mdct, imdct


def test_mdct():
    "Test mdct and imdct tight frame property"
    sfreq = 1000.  # Hz
    f = 7.  # Hz
    for T in [128, 512]:
        t = np.arange(T).astype(np.float)
        x = np.sin(2 * np.pi * f * t / sfreq)
        wsize = 32
        X = mdct(x, wsize)
        xp = imdct(X, wsize)

        assert_array_almost_equal(x, xp, decimal=6)
