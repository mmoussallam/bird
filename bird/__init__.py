"""
BIRD
====

Pure Python implementation of the BIRD algorithm. BIRD is a denoising
algorithm using randomized greedy pursuits. It works for single
signal and supports (structured)-sparsity for multivariate signals, e.g.
multichannel array data.

Reference
---------
Algorithm presented here are described in:

Blind Denoising with Random Greedy Pursuits.
Moussallam, M., Gramfort, A., Daudet, L., & Richard, G. (2014).
IEEE Signal Processing Letters, 21(11), 1341-1345
Preprint available at: http://arxiv.org/abs/1312.5444
"""
__version__ = '0.1'

from ._bird import bird, s_bird
