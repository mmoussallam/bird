# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# Algorithm presented here are described in:
# Blind Denoising with Random Greedy Pursuits.
# Moussallam, M., Gramfort, A., Daudet, L., & Richard, G. (2014).
# IEEE Signal Processing Letters, 21(11), 1341�1345
#
# License: BSD (3-clause)

from math import sqrt
import multiprocessing
from functools import partial

import numpy as np
from scipy.special import erfinv
from scipy import linalg

from joblib import Parallel, delayed
from mdct_tools import mdct_waveform, mdct, MDCT


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _single_mp_run(x, Phi, bound, max_iter, verbose=False, pad=0,
                   random_state=None):
    """ run of the RSSMP algorithm """

    rng = check_random_state(random_state)
    pad = int(pad)
    x = np.concatenate((np.zeros(pad), x, np.zeros(pad)))

    n = x.size
    m = Phi.doth(x).size
    err_mse = []

    # Initialisation
    residual = np.concatenate((x.copy(), np.zeros(max(Phi.sizes) / 2)))

    s = np.zeros(m)
    x_est = np.zeros(n)
    # Main algorithm
    coeffs = np.zeros(m)
    it_number = 0
    current_lambda = 1
    err_mse.append(linalg.norm(residual))

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (current_lambda > bound) & (it_number < max_iter):

        # pick a shift at random : in each size
        rndshifts = []
        for scale_idx, size in enumerate(Phi.sizes):
            shift = rng.randint(low=0, high=size / 4)
            coeffs[scale_idx * n:(scale_idx + 1) * n] = mdct(
                residual[shift:shift + n], size).ravel()
            rndshifts.append(shift)

        # Select a new element
        idx = np.argmax(np.abs(coeffs))

        # Update coefficients
        s[idx] += coeffs[idx]

        # Only one method now : local update via a cached waveform
        # find scale and frequency bin of selected atom
        scale_idx = idx / n
        size = Phi.sizes[scale_idx]
        F = n / (size / 2)
        frame = (idx - (scale_idx * n)) % F
        freq_bin = ((idx - (scale_idx * n))) // F
        pos = (frame * size / 2) - size / 4 + rndshifts[scale_idx]
        residual[pos:pos + size] -= coeffs[idx] * mdct_waveform(size, freq_bin)

        # also add it to the reconstruction
        x_est[pos:pos + size] += coeffs[idx] * mdct_waveform(size, freq_bin)

        # error computation (err_mse)
        err_mse.append(linalg.norm(residual))

        current_lambda = np.sqrt(1 - err_mse[-1] / err_mse[-2])
        if current_lambda <= bound:
            x_est[pos:pos + size] -= coeffs[idx] * mdct_waveform(size,
                                                                 freq_bin)
        if verbose:
            print("Iteration %d : Current lambda of %1.4f" % (
                  it_number, current_lambda))
        it_number += 1

    return x_est, err_mse


def _single_multichannel_mp_run(X, Phi, bound, selection_rule, stop_crit,
                                max_iter, verbose=False, pad=0,
                                random_state=None):
    """ run of the structured variant of the RSSMP algorithm """
    rng = check_random_state(random_state)

    # padding as v stak
    pad = int(pad)
    n_channels = X.shape[0]
    X = np.hstack((np.zeros((n_channels, pad)), X,
                   np.zeros((n_channels, pad))))
    n_samples = X.shape[1]
    n_projs = Phi.doth(X).shape[1]
    err_mse = {}

    # Initialisation
    residual = np.hstack((X.copy(), np.zeros((n_channels,
                                              max(Phi.sizes) / 2))))

    s_rep = np.zeros((n_channels, n_projs))
    X_est = np.zeros((n_channels, n_samples))
    # Main algorithm
    coeffs = np.zeros((n_channels, n_projs))

    it_number = 0
    current_lambda = 1
    for c_idx in range(n_channels):
        err_mse[c_idx] = []
        err_mse[c_idx].append(linalg.norm(residual[c_idx, :]))

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (current_lambda > bound) & (it_number < max_iter):

        # pick a shift at random : in each size
        rndshifts = {}
        for c_idx in range(n_channels):
            rndshifts[c_idx] = []
        for s_idx, L in enumerate(Phi.sizes):
            shift = rng.randint(low=0, high=L / 4)
            for c_idx in range(n_channels):
                coeffs[c_idx, s_idx * n_samples:(s_idx + 1) * n_samples] = mdct(
                    residual[c_idx, shift:shift + n_samples], L).ravel()
                rndshifts[c_idx].append(shift)

        # Multichannel mode : we combine projections
        combined = selection_rule(coeffs ** 2)

        # Select a new element
        idx = np.argmax(np.abs(combined))
        # find scale and frequency bin of selected atom
        s_idx = idx / n_samples
        L = Phi.sizes[s_idx]
        F = n_samples / (L / 2)
        frame = (idx - (s_idx * n_samples)) % F
        freq_bin = ((idx - (s_idx * n_samples))) // F

        # Update coefficients and residual
        current_lambda_array = np.zeros(n_channels)
        for c_idx in range(n_channels):
            s_rep[c_idx, idx] += coeffs[c_idx, idx]

            # Only one method now : local update via a cached waveform
            pos = (frame * L / 2) - L / 4 + rndshifts[c_idx][s_idx]
            residual[c_idx, pos:pos + L] -= coeffs[c_idx, idx] * \
                mdct_waveform(L, freq_bin)

            # also add it to the reconstruction
            X_est[c_idx, pos:pos + L] += coeffs[c_idx, idx] * \
                mdct_waveform(L, freq_bin)

            # error computation (err_mse)
            err_mse[c_idx].append(linalg.norm(residual[c_idx, :]))

            current_lambda_array[c_idx] = np.sqrt(
                1. - err_mse[c_idx][-1] / err_mse[c_idx][-2])

        current_lambda = stop_crit(current_lambda_array)

        if verbose:
            print("Iteration %d : Current lambda of %1.4f" % (
                  it_number, current_lambda))
        it_number += 1

    return X_est[:, pad: -pad], err_mse


def _pad(X):
    """ add zeroes on the border to make sure the signal length is a
    power of two """
    p_above = int(np.floor(np.log2(X.shape[1])))
    M = 2 ** (p_above + 1) - X.shape[1]
    X = np.hstack((np.zeros((X.shape[0], M)), X))

    return X, M


def _denoise(seeds, x, dico, sup_bound, n_atoms, verbose=False, indep=True,
             stop_crit=None, selection_rule=None, pad=0):
    """ multiple rssmp runs with a smart stopping criterion using
    the convergence decay monitoring
    """
    approx = []
    for seed in seeds:
        if verbose > 0:
            print("Run seed %d" % seed)
        if indep:
            approx.append(_single_mp_run(x, dico, sup_bound, n_atoms,
                                         verbose=verbose, pad=pad,
                                         random_state=seed)[0])
        else:
            approx.append(_single_multichannel_mp_run(x, dico, sup_bound,
                                                      selection_rule,
                                                      stop_crit,
                                                      n_atoms, verbose=verbose,
                                                      pad=pad,
                                                      random_state=seed)[0])
    return approx


def _bird_core(X, scales, n_runs, Lambda_W, max_iter=100,
               stop_crit=np.mean,
               selection_rule=np.sum,
               n_jobs=1, indep=True,
               random_state=None, verbose=False):
    """Automatically detect when noise zone has been reached and stop
    MP at this point

    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        The numpy n_channels-vy-N array to be denoised where n_channels is number of sensors
        and N the dimension
    scales : lists
        The list of MDCT scales that will be used to built the
        dictionary Phi
    n_runs : int
        the number of runs (n_runs in the paper)
    Lambda_W : float
        bound for lambda under which a run will be stopped
    max_iter : int
        Maximum number of iterations (serves as alternate stopping criterion)
    stop_crit : function
        controls the calculation of Lambda
    selection_rule : function
        controls the way multiple channel projections are combined for atom selection
        only used if indep=False
    n_jobs : int
        number of jobs to run in parallel
    indep : bool
        True for BIRD (independent processing of each channel,
        False for S-BIRD (structured sparsity seeked)
    random_state : None | int | np.random.RandomState
        To specify the random generator state (seed).
    verbose : bool
        verbose mode

    Returns
    -------
    X_denoise : array, shape (n_channels, n_times)
        denoised array of same shape as X
    """
    Phi = MDCT(scales)
    pad = int(1.5 * max(scales))
    X_denoise = np.zeros_like(X)
    approx = []
    rng = check_random_state(random_state)
    seeds = rng.randint(np.iinfo(np.int).max, size=n_runs)

    if n_jobs <= 0:
        n_cores = multiprocessing.cpu_count()
        n_jobs = min(n_cores + n_jobs + 1, n_cores)

    if indep:
        # Independent treat of each channel (plain BIRD)
        for r, x in zip(X_denoise, X):
            this_approx = Parallel(n_jobs=n_jobs)(
                        delayed(_denoise)(this_seeds, x, Phi, Lambda_W, max_iter,
                                      pad=pad, verbose=verbose) for this_seeds in
                                      np.array_split(seeds, n_jobs))
            this_approx = sum(this_approx[1:], this_approx[0])
            r[:] = sum([a[pad:-pad] for a in this_approx])
            approx.append(this_approx)
    else:
        # data need to be processed jointly
        this_approx = Parallel(n_jobs=n_jobs)(
                        delayed(_denoise)(this_seeds, X, Phi, Lambda_W, max_iter,
                                    pad=pad, verbose=verbose,
                                    selection_rule=selection_rule,
                                    indep=False, stop_crit=stop_crit)
                                    for this_seeds in
                                    np.array_split(seeds, n_jobs))

        # reconstruction by averaging
        for jidx in range(len(this_approx)):
            for ridx in range(len(this_approx[jidx])):
                X_denoise += this_approx[jidx][ridx]

    X_denoise /= float(n_runs)
    return X_denoise


def bird(X, scales, n_runs, p_above, random_state=None, n_jobs=1,
         verbose=False):
    """ The BIRD algorithm as described in the paper

    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        The numpy n_channels-vy-N array to be X_denoised where n_channels
        is number of sensors and n_times the dimension
    scales : lists
        The list of MDCT scales that will be used to built the
        dictionary Phi
    n_runs : int
        the number of runs (n_runs in the paper)
    p_above : float
        probability of appearance of the max above which the noise hypothesis
        is considered false
    random_state : None | int | np.random.RandomState
        To specify the random generator state (seed).
    n_jobs : int
        The number of jobs to run in parallel.
    verbose : bool
        verbose mode

    Returns
    -------
    X_denoise : array, shape (n_channels, n_times)
        The X_denoised data.
    """
    X, prepad = _pad(X)

    # Computing Lambda_W(Phi, p_above)
    N = float(X.shape[1])
    # size of the full shift-invariant dictionary
    M = np.sum(np.array(scales) / 2) * N
    sigma = sqrt((1.0 - (2.0 / np.pi)) / float(N))
    Lambda_W = sigma * sqrt(2.0) * erfinv((1.0 - p_above) ** (1.0 / long(M)))
    print("Starting BIRD with MDCT dictionary of %d Atoms. "
          "Lambda_W=%1.3f, n_runs=%d,\n (can take a "
          "while)" % (M, Lambda_W, n_runs))
    X_denoised = _bird_core(X, scales, n_runs, Lambda_W, verbose=verbose,
                            max_iter=100, indep=True, n_jobs=n_jobs,
                            random_state=random_state)
    return X_denoised[:, prepad:]


# the stopping criterion is determined by the p_active parameter
def stop_crit(lambda_array, lint):
    lambda_array.sort()
    return np.mean(lambda_array[-int(lint):])


def selection_rule(projections_matrix, lint):
    sorted_projs = np.sort(projections_matrix ** 2, axis=0)
    return np.mean(sorted_projs[-lint:, :], axis=0)


def s_bird(X, scales, n_runs, p_above, p_active=1, random_state=None,
           n_jobs=1, verbose=False):
    """ Multichannel version of BIRD (S-BIRD) seeking Structured Sparsity

    Parameters
    ----------
     X : array, shape (n_channels, n_times)
        The numpy n_channels-vy-n_samples array to be denoised where n_channels
         is the number of sensors and n_samples the dimension
    scales : list of int
        The list of MDCT scales that will be used to built the
        dictionary Phi
    n_runs : int
        the number of runs (n_runs in the paper)
    p_above : float
        probability of appearance of the max above which the noise hypothesis
        is considered false
    p_active : float
        proportion of active channels (l in the paper)
    random_state : None | int | np.random.RandomState
        To specify the random generator state (seed).
    n_jobs : int
        The number of jobs to run in parallel.
    verbose : bool
        verbose mode

    Returns
    -------
    X_denoise : array, shape (n_channels, n_times)
        The denoised data.
    """
    X, prepad = _pad(X)
    # Computing Lambda_W(Phi, p_above)
    n_channels = X.shape[0]
    n_samples = float(X.shape[1])
    # size of the full shift-invariant dictionary
    M = np.sum(np.array(scales) / 2) * n_samples
    sigma = sqrt((1.0 - (2.0 / np.pi)) / float(n_samples))
    Lambda_W = sigma * sqrt(2.0) * erfinv((1.0 - p_above) ** (1.0 / long(M)))

    lint = int(n_channels * p_active)

    this_stop_crit = partial(stop_crit, lint=lint)  # XXX : check lint here
    this_selection_rule = partial(selection_rule, lint=lint)

    print("Starting S-BIRD with MDCT dictionary of %d Atoms."
          " Lambda_W=%1.3f, n_runs=%d, p_active=%1.1f" % (M, Lambda_W,
                                                            n_runs, p_active))
    denoised = _bird_core(X, scales, n_runs, Lambda_W, verbose=verbose,
                          stop_crit=this_stop_crit, n_jobs=n_jobs,
                          selection_rule=this_selection_rule,
                          indep=False)

    return denoised[:, prepad:]

if __name__ == '__main__':
    """ run a demo here"""
    print("Decomposing a signal + noise at 0dB of SNR")
    # will stop whenever the components can't be tell appart from noise anymore
    # We could increase the number of runs to improve results
    N = 2048
    scales = [32, 64, 128, 256, 512, 1024]
    # Size of the Shift-Invariant dictionary
    M = np.sum(np.array(scales) / 2) * N
    n_runs = 30
    verbose = False

    # tolerated probability of appearance under noise hypothesis
    # A good first choice for p_above is arguably 1/M
    p_above = 1.0 / M
    # creating noisy mix using classic doppler signal
    rng = np.random.RandomState(42)
    target_snr = 0
    x = np.linspace(0, 1, N)
    doppler = np.sqrt(x * (1 - x)) * np.sin((2.1 * np.pi) / (x + .05))
    X = doppler.reshape((1, N))
    X = X / linalg.norm(X)
    truth = X.copy()
    noise = rng.randn(*truth.shape)
    noise = 0.3 * np.exp(-float(target_snr) / 10.0) * noise / linalg.norm(noise)
    snr = 20 * np.log10(linalg.norm(X) / linalg.norm(noise))
    data = X + noise

    print("SNR = %s." % snr)
    print("Dictionary of {0} atoms with {1} runs: chose "
          "p_above={2}".format(M, n_runs, p_above))
    X_denoised = bird(data, scales, n_runs, p_above=p_above, random_state=42,
                      n_jobs=-1, verbose=verbose)

    residual = X_denoised - truth
    print("Noisy Signal at %1.3f dB gave a RMSE of "
          "%1.3f" % (snr, linalg.norm(residual)))

    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.plot(data.T, 'k', alpha=0.5)
    plt.plot(truth.T, 'r:', linewidth=2.0)
    plt.plot(X_denoised.T, 'b', linewidth=2.0)
    plt.legend(('Noisy', 'Clean', 'BIRD Estimate'))
    plt.title('Noisy at %d dB' % snr)
    plt.show()
