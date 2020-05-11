from ase.optimize.activelearning.gp.kernel import SquaredExponential
from ase.optimize.activelearning.gp.kernel import BondExponential

import numpy as np

X = np.random.rand(4,9)
#print(X)

mask = np.asarray([True, False, False] *3)

sq = SquaredExponential()
sq.l = 0.4
sq.weight = 1.
sq.mask = mask

print(sq.kernel_matrix(X).shape)
print(sq.gradient(X)[0].shape)

be = BondExponential()
be.l = 0.2
be.weight = 1.
be.init_metric(['C']*3, lambda x, y: 1.)
be.mask = mask

print(be.kernel_matrix(X).shape)
print(be.gradient(X)[0].shape)
