# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

from meeg_tools import simu_meg
from bird import bird, s_bird
from joblib import Memory


if __name__ == '__main__':
    SNR = 6.
    white = False  # change to True/False for white/pink noise

    scales = [8, 16, 32, 64, 128, 256]
    n_runs = 20

    # Structured sparsity parameters
    n_channels = 20
    l = 1.0

    random_state = 42

    # Reference true data
    seed = 101
    evoked_no_noise = simu_meg(snr=200, white=True, seed=seed)
    single_no_noise = evoked_no_noise.data[:n_channels, :]  # * 1e12

    # noisy simulation
    evoked_noise = simu_meg(snr=SNR, white=white, seed=seed)
    single_noise = evoked_noise.data[:n_channels, :]  # * 1e12

    n_jobs = 1
    memory = Memory(None)
    sbird_estimate = s_bird(single_noise, scales, n_runs, p_above=1e-8,
                            p_active=l, random_state=random_state,
                            n_jobs=1, memory=memory)
    bird_estimate = bird(single_noise, scales, n_runs, p_above=1e-8,
                         random_state=random_state, n_jobs=1,
                         memory=memory)

    subset = range(1, n_channels, 2)
    start = 100  # make time start at 0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    p1 = plt.plot(1e3 * evoked_no_noise.times[start:],
                 single_noise[subset, start:].T, 'k:', alpha=0.3)
    p2 = plt.plot(1e3 * evoked_no_noise.times[start:],
                 single_no_noise[subset, start:].T, 'r:', linewidth=1.5)
    p3 = plt.plot(1e3 * evoked_no_noise.times[start:],
                 bird_estimate[subset, start:].T, 'k-', linewidth=1.5)
    p4 = plt.plot(1e3 * evoked_no_noise.times[start:],
                 sbird_estimate[subset, start:].T, 'm-', linewidth=1.5)

    plt.legend((p1[0], p2[0], p3[0], p4[0]),
              ('Noisy', 'Clean', 'BIRD Estimates', 'S-BIRD Estimates'),
              loc='upper right')

    plt.xlabel('Time (ms)', fontsize=16.0)
    plt.ylabel('MEG')
    plt.ylim([-1.5e-12, 2.0e-12])
    plt.show()