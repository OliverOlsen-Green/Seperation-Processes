import numpy as np
from UNIQUAC import UNIQUAC

u = np.array([
    [0.0, 907.180, 295.280],
    [-165.93, 0.0, 356.300],
    [268.16, -78.297, 0.0]
])

R = 8.314
T = 303.15
r = np.array([3.1878, 2.5735, 0.92])
q = np.array([2.4, 2.336, 1.4])

eps = 1e-8

# Interaction between Benzene(1) and Acetone(2)
gamma_1_in_2 = UNIQUAC.gamma(np.array([eps, 1.0 - eps, 0.0]), r, q, R, T, u)[0]
gamma_2_in_1 = UNIQUAC.gamma(np.array([1.0 - eps, eps, 0.0]), r, q, R, T, u)[1]

# Interaction Benzene(1) and Water(3)
gamma_1_in_3 = UNIQUAC.gamma(np.array([eps, 0.0, 1.0 - eps]), r, q, R, T, u)[0]
gamma_3_in_1 = UNIQUAC.gamma(np.array([1.0 - eps, 0.0, eps]), r, q, R, T, u)[2]

# Interaction Acetone(2) and Water(3)
gamma_2_in_3 = UNIQUAC.gamma(np.array([0.0, eps, 1.0 - eps]), r, q, R, T, u)[1]
gamma_3_in_2 = UNIQUAC.gamma(np.array([0.0, 1.0 - eps, eps]), r, q, R, T, u)[2]

print(f"1 in 2 (Benzene in Acetone): {gamma_1_in_2:.2f}")
print(f"2 in 1 (Acetone in Benzene): {gamma_2_in_1:.2f}")
print(f"1 in 3 (Benzene in Water)  : {gamma_1_in_3:.2f}")
print(f"3 in 1 (Water in Benzene)  : {gamma_3_in_1:.2f}")
print(f"2 in 3 (Acetone in Water)  : {gamma_2_in_3:.2f}")
print(f"3 in 2 (Water in Acetone)  : {gamma_3_in_2:.2f}")