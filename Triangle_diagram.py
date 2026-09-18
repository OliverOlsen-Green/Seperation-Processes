import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad_vec
import math
import mpltern
import pandas as pd

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()
data_E = pd.read_csv("LLE_phase_data.csv")

A_l = data_I[:,0]
A_r = data_II[:,0]
B_l = data_I[:,1]
B_r = data_II[:,1]
C_l = data_I[:,2]
C_r = data_II[:,2]

data_1_pd = data_E.loc[:,['1','2','3']]
data_2_pd = data_E.loc[:,['4','5','6']]
data_1 = data_1_pd.to_numpy()
data_2 = data_2_pd.to_numpy()
A_L = data_1[:,0]
A_R = data_2[:,0]
B_L = data_1[:,1]
B_R = data_2[:,1]
C_L = data_1[:,2]
C_R = data_2[:,2]


#A = data_left[:,0]

#print(data_left[:,0])

ax = plt.subplot(projection="ternary", ternary_sum=100.0)

ax.set_tlabel("A")
ax.set_llabel("B")
ax.set_rlabel("C")

ax.plot(B_l,A_l,C_l)
ax.plot(B_r,A_r,C_r)

ax.plot(B_L,A_L,C_L)
ax.plot(B_R,A_R,C_R)

ax.grid()

plt.show()
