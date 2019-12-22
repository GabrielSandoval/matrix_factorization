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
- tol         : training stops when error difference between two consecutive iteration is less than epsilon
"""

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

print("\n------------------- INPUT --------------------")
print(R)


mf = MF(R, K=2, alpha=0.05, beta=0.01, target_accuracy=0.99, max_iterations=1000, tol=0.0001, logger=Logger())
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
print("\nItem latent matrix:")
print(mf.P)
print("\nUser latent matrix:")
print(mf.Q)
