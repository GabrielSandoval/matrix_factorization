from lib.mf import MF
from lib.logger import Logger

import numpy as np

"""
Arguments
- R (ndarray)     : user-item rating full matrix
- K (int)         : number of latent dimensions
- alpha (float)   : learning rate
- beta (float)    : regularization parameter
- target_accuracy : training stops when target accuracy is reached
- max_iterations  : training stops when max iteration is reached
- tol             : training stops when error difference between two
                    consecutive iteration is less than epsilon
"""

R = np.array([
    [0, 0, 1, 3, 2],
    [4, 0, 1, 5, 3],
    [3, 0, 1, 0, 0],
    [0, 2, 0, 2, 0],
])

print("\n------------------- INPUT --------------------")
print(R)

mf = MF(
    R,
    K=2,
    alpha=0.0001,
    beta=0.001,
    target_accuracy=0.9999,
    max_iterations=50000,
    tol=0.000001,
    logger=Logger()
)

print("\n--------------- HYPERPARAMETERS ----------------")
mf.log_hyperparameters()

print("\n------------------ TRAINING --------------------")
mf.train()

print("\n------------------ RESULTS ---------------------")

print("\nFull matrix:")
print(mf.full_matrix())

print("\n-------------- TRAINED PARAMETERS --------------")

print("\nBias:")
print(mf.b)
print("\nItem bias:")
print(mf.b_i)
print("\nUser bias:")
print(mf.b_u)
print("\nUser latent matrix:")
print(mf.P)
print("\nItem latent matrix:")
print(mf.Q)
