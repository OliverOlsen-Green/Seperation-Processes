#import libraries and classes from other files
from Thomas import Thomas
from UNIQUAC import UNIQUAC
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad_vec
import math

class Isotheral_Sum_Rates:

    def Mass_Molar(x,Mr):
        #molar conc
        x_mol = np.zeros(len(Mr))
        for i in range(0,len(Mr)):
            x_mol[i] = x[i] / Mr[i]
        return x_mol

    def Molar_mass_to_mol_frac(X):
        Components = X.shape[1]
        # y inital
        Stages = X.shape[0]
        x_frac = np.zeros((Stages,Components))
        X_total = np.zeros(Stages)
        for j in range(0,Stages):
            X_total[j] = np.sum(X[j,:])
            for i in range(0,Components):
                x_frac[j,i] = X[j,i] / X_total[j]
        return x_frac



    def y_guess(x_I,V,L,F,z):
        Components = x_I.shape[1]
        #y inital
        Stages = x_I.shape[0]
        y = np.zeros((Stages,Components))
        error = 1
        epsillon = 1e-12

        while error > epsillon:
            y_new = y.copy()
            for j in range(0, Stages):
                for i in range(0, Components):
                    if j == 0:
                        y_new[j, i] = ((V[j + 1] * y[j + 1, i]) + (F[j]* z[j, i]) - (L[j] * x_I[j, i])) / V[j]
                    elif j == Stages - 1:
                        y_new[j, i] = ((L[j - 1] * x_I[j - 1, i]) + (F[j]* z[j, i]) - (L[j] * x_I[j, i])) / V[j]
                    else:
                        y_new[j, i] = ((L[j - 1] * x_I[j - 1, i]) + (V[j + 1] * y[j + 1, i]) + (F[j]* z[j, i]) - (
                                L[j] * x_I[j, i])) / V[j]

            error = np.max(y_new - y)
            y = y_new.copy()
        return y

    def K_factors(x_I,x_II,r,q,R,T,u):
        Components = x_I.shape[1]
        Stages = x_I.shape[0]
        gamma_I = np.zeros((Stages,Components))
        gamma_II = np.zeros((Stages,Components))
        K = np.zeros((Stages,Components))
        for j in range(0,Stages):
            gamma_I[j,:] = UNIQUAC.gamma(x_I[j, :], r, q, R, T, u)
            gamma_II[j,:] = UNIQUAC.gamma(x_II[j, :], r, q, R, T, u)

        for j in range(0,Stages):
            for i in range(0,Components):
                K[j,i] = gamma_I[j,i] / gamma_II[j,i]

        return K

    def Stripping_Factor(x_I,x_II,r,q,R,T,u,L_I,L_II):
        Components = x_I.shape[1]
        # y inital
        Stages = x_I.shape[0]
        K = Isotheral_Sum_Rates.K_factors(x_I,x_II,r,q,R,T,u)
        Components = x_I.shape[1]

        S = np.zeros((Stages,Components))

        for i in range(0,Components):
            for j in range(0,Stages):
                S[j,i] = L_II[j] * (K[j,i] / L_I[j])

        return S

    def Inner_Loop(x_I, x_ii, r, q, R, T, u, L_I, L_II, F):

        # copies of the x and y phase for the while loop
        x_i = x_I.copy()
        x_II = x_ii.copy()

        # shape of x_ii for stages and components
        Stages = x_i.shape[0]
        Components = x_i.shape[1]

        # initalise error and tau for convergence check
        tau = 1
        error = 0.001

        while tau > error:
            # find the Stripping factor from the inputs
            S_factor = Isotheral_Sum_Rates.Stripping_Factor(x_i, x_II, r, q, R, T, u, L_I, L_II)

            # find the liquid flow l_ji using thomas

            l_ji = Thomas.Thomas(S_factor, F)

            # from the liquid flow find the v_ji and then x and y compositions
            v_ji = np.zeros((Stages, Components))
            for j in range(0, Stages):
                for i in range(0, Components):
                    v_ji[j, i] = S_factor[j, i] * l_ji[j, i]

            # find the sum of liquid and vapour flows for each stage
            L_J = np.zeros(Stages)
            V_J = np.zeros(Stages)
            # initalise for the x and y compositions
            x_new = np.zeros((Stages, Components))
            y_new = np.zeros((Stages, Components))

            for j in range(0, Stages):
                L_J[j] = np.sum(l_ji[j, :])
                V_J[j] = np.sum(v_ji[j, :])
                for i in range(0, Components):
                    x_new[j, i] = l_ji[j, i] / L_J[j]
                    y_new[j, i] = v_ji[j, i] / V_J[j]

            # convergence check

            tau = np.sum(abs(x_new - x_i))
            # print(tau)
            # print(x_new)
            # print(y_new)
            # Normalise the values, initalise matrix of new vals and sums of each stages
            x_new_norm = np.zeros((Stages, Components))
            y_new_norm = np.zeros((Stages, Components))
            x_norm_sum = np.zeros(Stages)
            y_norm_sum = np.zeros(Stages)

            for j in range(0, Stages):
                x_norm_sum[j] = np.sum(x_new[j, :])
                y_norm_sum[j] = np.sum(y_new[j, :])
                for i in range(0, Components):
                    x_new_norm[j, i] = x_new[j, i] / x_norm_sum[j]
                    y_new_norm[j, i] = y_new[j, i] / y_norm_sum[j]

            # find the new K values from normalised x and y
            K = Isotheral_Sum_Rates.K_factors(x_new_norm, y_new_norm, r, q, R, T, u)

            # Use K-values to find new y values
            y_New = K * x_new_norm

            # normalise the y values again
            y_New_Norm = np.zeros((Stages, Components))
            y_New_Norm_sum = np.zeros(Stages)
            for j in range(0, Stages):
                y_New_Norm_sum[j] = np.sum(y_New[j, :])
                for i in range(0, Components):
                    y_New_Norm[j, i] = y_New[j, i] / y_New_Norm_sum[j]
            # update inputs for loop
            x_i = x_new_norm
            x_II = y_New_Norm

        return x_i, x_II, L_J, V_J

    def Outer_Loop(x_I, x_ii, r, q, R, T, u, L_I, L_II, F):
        # copies of the x and y phase for the while loop
        x_i = x_I.copy()
        x_II = x_ii.copy()

        # copies of flows for outer while loop
        L_i = L_I.copy()
        L_ii = L_II.copy()
        # shape of x_ii for stages and components
        Stages = x_i.shape[0]
        Components = x_i.shape[1]

        # initalise error and tau for convergence check
        tau = 1
        error = 0.001

        while tau > error:
            # get x,y,L_J,V_J values from the inner loop
            x, y, L_J, V_J = Isotheral_Sum_Rates.Inner_Loop(x_I, x_ii, r, q, R, T, u, L_i, L_ii, F)

            # print(L_J, "Liquid from innner loop")
            # print(V_J, "Vapour from inner loop")
            # find the stripping factors from these
            S_factors = Isotheral_Sum_Rates.Stripping_Factor(x, y, r, q, R, T, u, L_J, V_J)
            # print(y, "y values")
            # find the new tear variable to recalc liquid flows per stage (V)
            V_New = np.zeros(Stages)
            y_sum = np.zeros(Stages)
            for j in range(0, Stages):
                y_sum[j] = np.sum(y[j, :])
                V_New[j] = V_J[j] * y_sum[j]
            # print(y_sum, "Summation of y")
            # initalise tau and use the V_New values compared to inlet L_ii for the convergfence chejck 
            tau = 0.0
            for j in range(0, Stages):
                tau += abs((((V_New[j] - L_ii[j]) / L_ii[j]) ** 2))
            print(tau)

            F_stage = np.zeros(Stages)
            F_sum = 0.0
            L_new = np.zeros(Stages)
            # find the new L from V_new and F_sum
            for j in range(0, Stages):
                F_stage[j] = np.sum(F[j, :])
                F_sum += F_stage[j]
                if j == Stages - 1:
                    L_new[j] = F_sum - V_New[0]
                else:
                    L_new[j] = V_New[j + 1] + F_sum - V_New[0]
            # print(L_new, "New liquid flows")
            # print(V_New, "New Vapour Flows")
            # update L_i, L_ii, x and y
            L_i = L_new
            L_ii = V_New
            x_i = x
            x_ii = y

        return L_i, L_ii,x_i, x_ii


V = np.array([200., 192., 184., 176., 166., 150.])

F = np.array([100., 0., 0., 0., 0., 150.])

L = np.array([90., 82., 74., 66., 58., 50.])

x = np.array([[0.02  , 0.5445, 0.4355],
              [0.016 , 0.6014, 0.3826],
              [0.012 , 0.6705, 0.3175],
              [0.008 , 0.7559, 0.2361],
              [0.008 , 0.8554, 0.1366],
              [0.02  , 0.96  , 0.02  ]])

z = np.array([[0. , 0.5, 0.5],
              [0. , 0. , 0. ],
              [0. , 0. , 0. ],
              [0. , 0. , 0. ],
              [0. , 0. , 0. ],
              [1. , 0. , 0. ]])

F_comp = np.array([[  0.,  50.,  50.],
                    [  0.,   0.,   0.],
                    [  0.,   0.,   0.],
                    [  0.,   0.,   0.],
                    [  0.,   0.,   0.],
                    [150.,   0.,   0.]])

y = Isotheral_Sum_Rates.y_guess(x,V,L,F,z)


#print(y, "Initial y guess")
Stages = 9


R = 8.314
T = 303.15

r = np.array([3.1878,2.5735,0.92])
q = np.array([2.4,2.336,1.4])
u = np.array([
    [0,295.280,907.180],
    [-165.93,0,356.3],
    [268.18,-78.297,0]
])


L_I,L_II,x_i,x_ii = Isotheral_Sum_Rates.Outer_Loop(x,y,r,q,R,T,u,L,V,F_comp)


print(L_I, "Liquid flow phase 1")
print(L_II, "Liquid flow phase 2")

print(x_i, "composition of 1st liquid")

print(x_ii, "composition of 2nd liquid")
"""""
section for debugging funcs, step by step
K = Isotheral_Sum_Rates.K_factors(x,y,r,q,R,T,u)

#print(K)

S = Isotheral_Sum_Rates.Stripping_Factor(x,y,r,q,R,T,u,L,V)
#print(S)

#liqudi flowrates for each species
l_ij= Thomas.Thomas(S,F_comp)

v_ij = S * l_ij

print(l_ij)

print(v_ij)

V_j = np.zeros(len(V))
L_j = np.zeros(len(V))

for i in range(0,len(V)):
    V_j[i] = np.sum(v_ij[i,:])
    L_j[i] = np.max(l_ij[i,:])

print(V_j, "Vapour flows per stage")
print(L_j, "Liquid flows per stage")
Stages = 6
Components = 3

y_ji = Isotheral_Sum_Rates.Molar_mass_to_mol_frac(v_ij)
x_ji = Isotheral_Sum_Rates.Molar_mass_to_mol_frac(l_ij)
print(y_ji)
print(x_ji)
K_ji = Isotheral_Sum_Rates.K_factors(x_ji,y_ji,r,q,R,T,u)
"""""
"""""
print(K_ji)
#print(x_new)

#x_frac = Isotheral_Sum_Rates.Molar_mass_to_mol_frac(x_new)

#print(x_frac)

#keep using this as a method to debug


x_i, x_ii = Isotheral_Sum_Rates.Inner_Loop(x,y,r,q,R,T,u,L,V,F_comp)

print(x_i, "x values from inner loop")

print(x_ii, "y values from inner loop")
S_new = Isotheral_Sum_Rates.Stripping_Factor(x_i,x_ii,r,q,R,T,u,L_j,V_j)



print(S_new)

print(y_new, "new y values")

L_II_new = np.zeros(Stages)
y_sum = np.zeros(Stages)
for j in range(0,Stages):
    y_sum[j] = np.sum(y_new[j,:])
    L_II_new[j] = V[j] * (y_sum[j])

print(y_sum)
print(L_II_new, "new flow of liquid 2")
tau = 0.0
for j in range(0,Stages):
    tau += ((L_II_new[j] - V[j]) /  V[j]) ** 2

            #find L_I
F_total =0.0
L_I_new = np.zeros(Stages)
F_stage = np.zeros(Stages)
for i in range(0,Stages):
    F_stage[i] = np.sum(F_comp[i,:])
    F_total += F_stage[i]
    if i == Stages - 1:
        L_I_new[i] = F_total - L_II_new[0]
    else:
        L_I_new[i] = L_II_new[i + 1] + F_total - L_II_new[0]

print(L_I_new, "new flows of liquid 1")
print(tau)
"""
