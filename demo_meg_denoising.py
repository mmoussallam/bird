# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
#
# License: BSD (3-clause)

from scipy import linalg
from meeg_tools import simu_meg
from bird import bird, s_bird
from joblib import Memory


if __name__ == '__main__':
    white = False  # change to True/False for white/pink noise

    scales = [8, 16, 32, 64, 128]
    n_runs = 30

    # Structured sparsity parameters
    n_channels = 20  # Set this value to 20 to reproduce figures from the paper
    p_active = 1.

    random_state = 42

    # Reference true data
    # Note : due to some changes in MNE, simulated data is no longer 
    # parameterized using explicit SNR values, but rather using the NAVE parameter
    # Look up some documentation there: https://mne.tools/dev/generated/mne.simulation.simulate_evoked.html#mne.simulation.simulate_evoked
    seed = 42
    evoked_no_noise = simu_meg(nave=10000, white=True, seed=seed)
    single_no_noise = evoked_no_noise.data[:n_channels, :]

    # noisy simulation : to simulate a SNR of approximately 10
    # we use 10 times less averaged epochs (nave parameter set to 2000)
    evoked_noise = simu_meg(nave=2000, white=white, seed=seed)
    single_noise = evoked_noise.data[:n_channels, :]

    n_jobs = 1  # set to -1 to run in parellel
    memory = Memory(None)
    p_above = 1e-8
    bird_estimate = bird(single_noise, scales, n_runs, p_above=p_above,
                         random_state=random_state, n_jobs=n_jobs,
                         memory=memory)
    sbird_estimate = s_bird(single_noise, scales, n_runs, p_above=p_above,
                            p_active=p_active, random_state=random_state,
                            n_jobs=n_jobs, memory=memory)

    print("RMSE BIRD : %s" % linalg.norm(bird_estimate - single_noise))
    print("RMSE S-BIRD : %s" % linalg.norm(sbird_estimate - single_noise))

    subset = range(1, n_channels, 2)
    start = 100  # make time start at 0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    p1 = plt.plot(1e3 * evoked_no_noise.times[start:],
                  single_noise[subset, start:].T, 'k:', alpha=0.5)
    p2 = plt.plot(1e3 * evoked_no_noise.times[start:],
                  single_no_noise[subset, start:].T, 'r:', linewidth=1.5)
    p3 = plt.plot(1e3 * evoked_no_noise.times[start:],
                  bird_estimate[subset, start:].T, 'k-', linewidth=1.5)
    p4 = plt.plot(1e3 * evoked_no_noise.times[start:],
                  sbird_estimate[subset, start:].T, 'm-', linewidth=1.5)

    plt.legend((p1[0], p2[0], p3[0], p4[0]),
               ('Noisy', 'Clean', 'BIRD Estimates', 'S-BIRD Estimates'),
               loc='upper right')

    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.ylim([-1.5e-12, 2.5e-12])
    plt.show()
