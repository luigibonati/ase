import numpy as np
import scipy.optimize


def hungarian_assignment(cost):

    n = len(cost)
    answers = hungarian.lap(np.copy(cost))
    return (np.arange(n), answers[0])

# Import the fast module if present, otherwise fall back to scipy.
try:
    import hungarian
    linear_sum_assignment = hungarian_assignment
except ImportError:
    linear_sum_assignment = scipy.optimize.linear_sum_assignment
