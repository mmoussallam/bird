# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_almost_equal
from bird.mdct_tools import mdct, imdct


def test_mdct():
    "Test mdct and imdct tight frame property"
    sfreq = 1000.  # Hz
    f = 7.  # Hz
    x1 = np.sin(2. * np.pi * f * np.arange(128, dtype=float) / sfreq)
    x2 = np.sin(2. * np.pi * f * np.arange(512, dtype=float) / sfreq)

    rng = np.random.RandomState(42)
    x3 = rng.standard_normal(x1.shape)

    wsize = 32

    for x in [x1, x2, x3]:
        X = mdct(x, wsize)
        xp = imdct(X, wsize)

        assert_array_almost_equal(x, xp, decimal=12)
