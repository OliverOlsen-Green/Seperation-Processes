import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad_vec
import math
#create class for the solving RR using UNIQUAC params
class UNIQUAC:

    def segment_fraction(x,r):
        #length of array x
        length = len(x)
        #initalise the segment fraciton and the denominator
        phi = np.zeros(length)
        denominator = 0.0
        sum = np.sum(x * r)
        for i in range(0,length):
            denominator += x[i] * r[i]
        for i in range(0,length):
            phi[i] = (x[i] * r[i]) / denominator

        return phi

    def area_fraction(x,q):
        # length of array x
        length = len(x)
        # initalise the area fraciton and the denominator
        theta = np.zeros(length)
        denominator = 0.0
        for i in range(0,length):
            denominator += (x[i] * q[i])
        for i in range(0,length):
            theta[i] = (x[i] * q[i]) / denominator

        return theta

    def binary_interaction(u,R,T):
        # rows
        rows = u.shape[0]
        cols = u.shape[-1]
        T_ji = np.zeros((rows,cols))
        for i in range(0,cols):
            for j in range(0,rows):
                T_ji[j][i] = np.exp(-(u[j][i] - u[i][i]) / (T))

        return T_ji

    def gamma(x,r,q,R,T,u):
        length = len(x)
        phi = UNIQUAC.segment_fraction(x,r)
        theta = UNIQUAC.area_fraction(x,q)
        T_ji = UNIQUAC.binary_interaction(u,R,T)
        #initalise vectors of gamma
        ln_gamma = np.zeros(length)
        Z = 10.0
        l_j = np.zeros(length)
        l_i = np.zeros(length)
        #for the for loops
        rows = u.shape[0]
        cols = u.shape[-1]

        #initalise the sum and calculate vectors of 2,3 and 4
        sum_1 = 0.00
        E_i = np.zeros(len(theta))

        for i in range(0, len(theta)):
            for j in range(0, len(theta)):
                E_i[i] += theta[j] * T_ji[j][i]

        # third sum
        C_i = np.zeros(len(theta))

        for j in range(0, len(theta)):
            for k in range(0, len(theta)):
                C_i[j] += theta[k] * T_ji[k][j]

        D_i = np.zeros(len(theta))

        for j in range(0, len(theta)):
            for i in range(0, len(theta)):
                D_i[i] += (theta[j] * T_ji[i][j]) / C_i[j]

        # for loops for the l params as need to be ised in sum loops
        for j in range(0,rows):
            l_j[j] = (Z / 2) * (r[j] - q[j]) - (r[j] - 1)
        for i in range(0,length):
            l_i[i] = (Z / 2) * (r[i] - q[i]) - (r[i] - 1)

        #loops that calculate each of the sums utilised
        for j in range(0,rows):
            sum_1 += x[j] * l_j[j]
        for i in range(0,cols):
            ln_gamma[i] = math.log(phi[i] / x[i]) + (Z/2) * q[i] * math.log(theta[i] / phi[i]) + \
                          l_i[i] - ((phi[i] / x[i]) * sum_1) + \
                          q[i] * (1 - math.log(E_i[i]) - D_i[i])

        activity_co = np.exp(ln_gamma)

        return activity_co



    def RR_LLE(E,z,x_I, x_II,F,r,q,R,T,u,Model):
        #find the gamma values for each of the respective phases
        gamma_I = UNIQUAC.gamma(x_I,r,q,R,T,u)
        gamma_II = UNIQUAC.gamma(x_II, r, q, R, T, u)

        #find the K-values
        K = np.zeros(len(x_I))

        #for the inital guess later use ideal model (Psat and P)
        if Model == "Ideal":
            for i in range(0, len(x_I)):
                K[i] = (x_I[i]) / (x_II[i])

        else:
            #eq K values using the UNIQUAC
            for i in range(0, len(x_I)):
                K[i] = gamma_II[i]/ gamma_I[i]

        #unpack for use in f-solve
        E = E[0]

        #array for the new values calcualted from RR eq
        x_I_new = np.zeros(len(x_I))
        x_II_new = np.zeros(len(x_I))
        for i in range(0,len(x_I)):
            x_I_new[i] = (z[i] * K[i]) / (1 + E * (K[i]-1))
            x_II_new[i] = z[i] / (1 + (E * (K[i]-1)))

        # sums of each to find the error
        x_I_new_sum = np.sum(x_I_new)
        x_II_new_sum = np.sum(x_II_new)

        error = abs(x_I_new_sum - x_II_new_sum)

        return error

    def liquid_comps(E,z,x_i, x_ii,F,r,q,R,T,u,Model):

        #for the while loop that will iterate until found compositions
        Error = 1
        epsillon = 1e-10
        Iterations = 0
        max_ITER = 1000

        x_I = x_i.copy()
        x_II = x_ii.copy()


        while Error > epsillon and max_ITER > Iterations:


            gamma_I = UNIQUAC.gamma(x_I, r, q, R, T, u)
            gamma_II = UNIQUAC.gamma(x_II, r, q, R, T, u)
            # find the K-values
            K = np.zeros(len(x_I))

            # for the inital guess later use ideal model (Psat and P)
            if Model == "Ideal":
                for i in range(0, len(x_I)):
                    K[i] = (x_I[i]) / (x_II[i])

            else:
                # eq K values using the UNIQUAC
                for i in range(0, len(x_I)):
                    K[i] = gamma_II[i]/ gamma_I[i]

            #find the value of the extract based on inital guesses (inputs)
            E_final = fsolve(UNIQUAC.RR_LLE,E,args=(z,x_I, x_II,F,r,q,R,T,u,Model))

            E_final = E_final[0]
            #find the new composition
            x_I_new = np.zeros(len(x_I))
            x_II_new = np.zeros(len(x_I))
            for i in range(0, len(x_I)):
                x_I_new[i] = (z[i] * K[i]) / (1 + E_final * (K[i] - 1))
                x_II_new[i] = z[i] / (1 + (E_final * (K[i] - 1)))

            #initalise the absoulrte error and the error for each phase
            error = 0
            error_I = 0.0
            error_II = 0.0
            for i in range(0,len(x_I)):
                error_I = abs(x_I_new[i] - x_I[i])
                error_II = abs(x_II_new[i] - x_II[i])
                error += np.maximum(error_I, error_II)
            Error = error

            Iterations = Iterations + 1
            x_I = x_I_new.copy()
            x_II = x_II_new.copy()

        return x_I, x_II, Iterations

    def plait_point(E,z,x_i, x_ii,F,r,q,R,T,u,Model):

        X_I = x_i.copy()
        X_II = x_ii.copy()
        Z = z.copy()
        Error = 1
        epsillon = 1e-4
        steps = 0
        max_steps = 133
        #store X_I and X_II for plotting tie lines
        history_I = [X_i.copy()]
        history_II = [X_ii.copy()]
        while Error > epsillon:

            x_I_new, x_II_new, iterations = UNIQUAC.liquid_comps(E, Z, X_I, X_II, 1, r, q, R, T, u, Model)

            # increase of acetone
            a_I = 0.0015
            Z[1] = Z[1] + a_I
            Z[0] = Z[0] - (a_I / 2)
            Z[2] = Z[2] - (a_I / 2)
            error = 0
            error_I = 0.0

            for i in range(0,len(X_I)):
                error_I = abs(x_I_new[i] - x_II_new[i])
                error += error_I
            Error = error

            X_I = x_I_new.copy()
            X_II = x_II_new.copy()
            history_I.append(X_I.copy())
            history_II.append(X_II.copy())
            steps = steps + 1
        return X_I, X_II, history_I, history_II







u = np.array([
    [0,295.280,907.180],
    [-165.93,0,356.3],
    [268.18,-78.297,0]
])

#print(u)

R = 8.314
T = 303.15

UNI = UNIQUAC

T_ji = UNI.binary_interaction(u,R,T)



X_i = np.array([0.499999,0.000001,0.5])
X_ii = np.array([0.2,0.1,0.7])
r = np.array([3.1878,2.5735,0.92])
q = np.array([2.4,2.336,1.4])
z = np.array([(0.5-(0.5e-10)),1e-10,0.5-(0.5e-10)])

E = 0.5
model = "UNIQUAC"
x_I, x_II, iterations = UNI.liquid_comps(E,z,X_i,X_ii,1,r,q,R,T,u,model)

"""print(x_I, "composition of Extract")
print(x_II, "composition of Raffinate")
print(iterations, "Iterations")

print(np.sum(x_I),"sum of Liquid phase 1")

print(np.sum(x_II), "sum of liquid phase 2")
"""

X_I_plait, X_II_plait,History_I,History_II = UNI.plait_point(E,z,x_I,x_II,1,r,q,R,T,u,model)

print(X_I_plait)
print(X_II_plait)

print(np.sum(X_I_plait),"sum of X_I")
print(np.sum(X_II_plait),"sum of X_II")

dh_I = pd.DataFrame(History_I)
dh_II = pd.DataFrame(History_II)
#x = np.array([1/3,1/3,1/3])
#gamma = UNI.gamma(x,r,q,R,T,u)

#print(gamma)

dh_I.to_csv("Phase_1_data.csv", index=False)
dh_II.to_csv("Phase_2_data.csv", index=False)
