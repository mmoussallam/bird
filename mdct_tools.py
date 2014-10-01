# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

import math
import numpy as np
from scipy import linalg
from scipy.fftpack import fft, ifft
import six


def _framing(a, L):
    shape = a.shape[:-1] + (a.shape[-1] - L + 1, L)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape,
                                           strides=strides)[::L // 2].T.copy()


def mdct_waveform(scale, freq_bin):
    L = float(scale)
    K = L / 2.0
    fact = math.sqrt(2.0 / K)
    const_fact = (np.pi / K) * (float(freq_bin) + 0.5)
    const_offset = (L + 1.0) / 2.0
    f = np.pi / L
    i = np.arange(scale, dtype=np.float)
    wf = (fact * np.sin(f * (i + 0.5)) *
          np.cos(const_fact * ((i - K / 2.0) + const_offset)))
    return wf / linalg.norm(wf)


def mdct(x, L):
    """Modified Discrete Cosine Transform (MDCT)

    Returns the Modified Discrete Cosine Transform with fixed
    window size L of the signal x.

    The window is based on a sine window.

    Parameters
    ----------
    x : ndarray, shape (N,)
        The signal
    L : int
        The window length

    Returns
    -------
    y : ndarray, shape (L/2, 2 * N / L)
        The MDCT coefficients

    See also
    --------
    imdct
    """
    x = np.asarray(x, dtype=np.float)

    N = x.size

    # Number of frequency channels
    K = L // 2

    # Test length
    if N % K != 0:
        raise RuntimeError('Input length must be a multiple of the half of '
                           'the window size')

    # Pad edges with zeros
    xx = np.zeros(L // 4 + N + L // 4)
    xx[L // 4:-L // 4] = x
    x = xx
    del xx

    # Number of frames
    P = N // K
    if P < 2:
        raise ValueError('Signal too short')

    # Framing
    x = _framing(x, L)

    # Windowing
    aL = np.arange(L, dtype=np.float)
    w_long = np.sin((np.pi / L) * (aL + 0.5))
    w_edge_L = w_long.copy()
    w_edge_L[:L // 4] = 0.
    w_edge_L[L // 4:L // 2] = 1.
    w_edge_R = w_long.copy()
    w_edge_R[L // 2:L // 2 + L // 4] = 1.
    w_edge_R[L // 2 + L // 4:] = 0.
    x[:, 0] *= w_edge_L
    x[:, 1:-1] *= w_long[:, None]
    x[:, -1] *= w_edge_R

    # Pre-twiddle
    x = x.astype(np.complex)
    x *= np.exp((-1j * np.pi / L) * aL)[:, None]

    # FFT
    y = fft(x, axis=0)

    # Post-twiddle
    y = y[:L // 2, :]
    y *= np.exp((-1j * np.pi * (L // 2 + 1.) / L)
                * (0.5 + aL[:L // 2]))[:, None]

    # Real part and scaling
    y = math.sqrt(2. / K) * np.real(y)
    return y


def imdct(y, L):
    """Inverse Modified Discrete Cosine Transform (MDCT)

    Returns the Inverse Modified Discrete Cosine Transform
    with fixed window size L of the vector of coefficients y.

    The window is based on a sine window.

    Parameters
    ----------
    y : ndarray, shape (L/2, 2 * N / L)
        The MDCT coefficients
    L : int
        The window length

    Returns
    -------
    x : ndarray, shape (N,)
        The reconstructed signal

    See also
    --------
    mdct
    """
    # Signal length
    N = y.size

    # Number of frequency channels
    K = L // 2

    # Test length
    if N % K != 0:
        raise ValueError('Input length must be a multiple of the half of '
                         'the window size')

    # Number of frames
    P = N // K
    if P < 2:
        raise ValueError('Signal too short')

    # Reshape
    temp = y
    y = np.zeros((L, P), dtype=np.float)
    y[:K, :] = temp
    del temp

    # Pre-twiddle
    aL = np.arange(L, dtype=np.float)
    y = y * np.exp((1j * np.pi * (L / 2. + 1.) / L) * aL)[:, None]

    # IFFT
    x = ifft(y, axis=0)

    # Post-twiddle
    x *= np.exp((1j * np.pi / L) * (aL + (L / 2. + 1.) / 2.))[:, None]

    # Windowing
    w_long = np.sin((np.pi / L) * (aL + 0.5))
    w_edge_L = w_long.copy()
    w_edge_L[:L // 4] = 0.
    w_edge_L[L // 4:L // 2] = 1.
    w_edge_R = w_long.copy()
    w_edge_R[L // 2:L // 2 + L // 4] = 1.
    w_edge_R[L // 2 + L // 4:L] = 0.
    x[:, 0] *= w_edge_L
    x[:, 1:-1] *= w_long[:, None]
    x[:, -1] *= w_edge_R

    # Real part and scaling
    x = math.sqrt(2. / K) * L * np.real(x)

    # Overlap and add
    def overlap_add(y, x):
        z = np.concatenate((y, np.zeros((K,))))
        z[-2 * K:] += x
        return z

    x = six.moves.reduce(overlap_add, [x[:, i] for i in range(x.shape[1])])

    # Cut edges
    x = x[K // 2:-K // 2].copy()
    return x


class MDCT(object):
    """Modified Discrete Cosine Transform (MDCT)

    Supports multiple MDCT dictionaries.

    Parameters
    ----------
    sizes : list of int
        The sizes of MDCT windows e.g. [256, 1024]
    """
    def __init__(self, sizes):
        self.sizes = sizes

    def _dot(self, y):
        cnt = 0
        N = y.size / len(self.sizes)
        x = np.zeros(N)
        for L in self.sizes:
            this_y = y[cnt:cnt + N]
            if (np.count_nonzero(this_y) > 0):
                this_x = imdct(np.reshape(this_y, (L // 2, -1)), L)
                x += this_x
            cnt += N

        return x

    def dot(self, y):
        if y.ndim == 1:
            return self._dot(y)
        else:
            return np.array([self._dot(this_y) for this_y in y])

    def _doth(self, x):
        return np.concatenate([mdct(x, L).ravel() for L in self.sizes])

    def doth(self, x):
        if x.ndim == 1:
            return self._doth(x)
        else:
            return np.array([self._doth(this_x) for this_x in x])
