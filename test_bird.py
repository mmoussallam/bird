# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_array_almost_equal
from bird import bird, s_bird
from scipy import linalg


def _make_doppler(N):
    x = np.linspace(0, 1, N)
    doppler = np.sqrt(x * (1 - x)) * np.sin((2.1 * np.pi) / (x + .05))
    return doppler.reshape((1, N))


def test_bird():
    """ test bird calls """
    N = 1024
    scales = [32, 64, 128, 256]
    # Size of the Shift-Invariant dictionary
    M = np.sum(np.array(scales) / 2) * N
    n_runs = 30
    verbose = False

    # tolerated probability of appearance under noise hypothesis
    # A good first choice for p_above is arguably 1/M
    p_above = 1.0 / M
    # creating noisy mix using classic doppler signal
    rng = np.random.RandomState(42)
    target_snr = 5
    X = _make_doppler(N)
    X = X / linalg.norm(X)
    truth = X.copy()
    noise = rng.randn(*truth.shape)
    noise = 0.3 * np.exp(-float(target_snr) / 10.0) * noise / linalg.norm(noise)
    data = X + noise

    X_denoised = bird(data, scales, n_runs, p_above=p_above, random_state=42,
                      n_jobs=1, verbose=verbose)
    # test denoised estimate is close to original
    assert_array_almost_equal(X_denoised, truth, decimal=2)

    # test second call produce same result
    X_denoised_again = bird(data, scales, n_runs, p_above=p_above,
                            random_state=42, n_jobs=1, verbose=verbose)
    assert_array_almost_equal(X_denoised, X_denoised_again, decimal=8)


def test_sbird():
    """ test s-bird calls """
    N = 1024
    scales = [32, 64, 128, 256]
    # Size of the Shift-Invariant dictionary
    M = np.sum(np.array(scales) / 2) * N
    n_runs = 10
    n_channels = 5
    verbose = False

    # tolerated probability of appearance under noise hypothesis
    # A good first choice for p_above is arguably 1/M
    p_above = 1.0 / M
    # creating noisy mix using classic doppler signal
    # noise different in each channel
    rng = np.random.RandomState(42)
    target_snr = 5
    X = _make_doppler(N)
    X = X / linalg.norm(X)
    X = np.tile(X, [n_channels, 1])
    truth = X.copy()
    data = np.zeros_like(X)
    for chan in range(X.shape[0]):
        noise = rng.randn(*truth[chan, :].shape)
        noise = 0.3 * np.exp(-float(target_snr) / 10.0) * noise / linalg.norm(noise)
        data[chan, :] = X[chan, :] + noise

    X_denoised = s_bird(data, scales, n_runs, p_above=p_above, random_state=42,
                      n_jobs=1, verbose=verbose)
    # test denoised estimate is close to original
    assert_array_almost_equal(X_denoised, truth, decimal=2)

