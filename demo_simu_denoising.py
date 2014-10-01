"""
===============================================
Demo of denoising of synthetic signal iwth BIRD
===============================================

Reference:

Blind Denoising with Random Greedy Pursuits.
Moussallam, M., Gramfort, A., Daudet, L., & Richard, G. (2014).
IEEE Signal Processing Letters, 21(11), 1341-1345

"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Manuel Moussallam <manuel.moussallam@gmail.com>
# License: BSD 3 clause

print(__doc__)

import numpy as np
from scipy import linalg
from joblib import Memory

from bird import bird

print("Decomposing a signal + noise at 0dB of SNR")
# will stop whenever the components can't be tell appart from noise anymore
# We could increase the number of runs to improve results
N = 2048
scales = [32, 64, 128, 256, 512, 1024]
# Size of the Shift-Invariant dictionary
M = np.sum(np.array(scales) // 2) * N
n_runs = 30
verbose = False

# tolerated probability of appearance under noise hypothesis
# A good first choice for p_above is arguably 1/M
p_above = 1.0 / M
# creating noisy mix using classic doppler signal
rng = np.random.RandomState(42)
x = np.linspace(0, 1, N)
doppler = np.sqrt(x * (1 - x)) * np.sin((2.1 * np.pi) / (x + .05))
X = doppler.reshape((1, N))
X /= linalg.norm(X)
truth = X.copy()
noise = rng.randn(*truth.shape)
noise *= .3 / linalg.norm(noise)
snr = 20. * np.log10(linalg.norm(X) / linalg.norm(noise))
data = X + noise

memory = Memory(None)

print("SNR = %s." % snr)
print("Dictionary of {0} atoms with {1} runs: chose "
      "p_above={2}".format(M, n_runs, p_above))
X_denoised = bird(data, scales, n_runs, p_above=p_above, random_state=42,
                  n_jobs=-1, verbose=verbose, memory=memory)

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
