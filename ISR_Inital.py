from Thomas import Thomas
from UNIQUAC import UNIQUAC
from ISR import Isotheral_Sum_Rates
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad_vec
import math
from LLE_unit_Sims import Mass_to_Mol


class LSR_Inital_Guess:

    def Min_Stage_Feed(N,Feed_1,Feed_2,F,S,L0): #finction for building feed matrix based on multifeeds for min stage func
        #F is the array of secondary feed, S sovent, L0 main feed
        Feed_Comp = np.zeros((N,len(F)))
        Feed = np.zeros(N)
        for j in range(0,N):
            for i in range(0,len(F)):
                #if statement that makes each of the arrays their location in the Feed and Feed_Comp
                if j == (Feed_1 - 1):
                    Feed[j] = np.sum(F)
                    Feed_Comp[j,i] = F[i]
                elif j == (Feed_2 - 1):
                    Feed[j] = np.sum(L0)
                    Feed_Comp[j,i] = L0[i]
                elif j == (N-1):
                    Feed[j] = S
                    if i == 0:
                        Feed_Comp[j,i] = S
                    else:
                        Feed_Comp[j,i] = 0.0
        return Feed, Feed_Comp

    def Feed_matrix(F,N,S_F,Feed_Comp,Feed_stage_amount):
        #find the solvent Flow rate (kg/s)
        S = np.sum(F) * S_F

        # if statement for if feed stage is 1 or multiple
        if Feed_stage_amount == 1:
            Components  = Feed_Comp.shape[0]
            #initalise the matrix based on number of stages
            F_ji = np.zeros((N,Components))

            for j in range(0,N):
                for i in range(0,Components):
                    if j == 0:
                        F_ji[j,i] = (Feed_Comp[i] * F[0])
                    elif j == (N-1):
                        if i == 0:
                            F_ji[j, i] = S
                        else:
                            F_ji[j,i] = 0.0
                    else:
                        F_ji[j,i] = 0.0
        else:
            Components  = Feed_Comp.shape[0]

            F_ji = np.zeros((N,Components))

            for j in range(0,N):
                for i in range(0,Components):
                    if j == (N-1):
                        if i == 0:
                            F_ji[j,i] = S
                        else:
                            F_ji[j,i] = 0.0
                    else:
                        if F[j] == 0:
                            F_ji[j,i] = 0.0
                        else:
                            F_ji[j,i] = F[j] * Feed_Comp[j,i]

        return F_ji

    def Sum_F(F_ji):
        Stages = F_ji.shape[0]
        Components = F_ji.shape[1]

        F = np.zeros(Stages)
        for j in range(Stages):
            F[j] = np.sum(F_ji[j,:])
        return F

    def Liquid_Flow(L_0,L_N, N):
        #increase of the liquid flow based on inital and final value (linear interpolation)
        L_increase = (L_N - L_0) / N

        L = np.zeros(N)
        for i in range(0,N):
            if i == 0:
                L[i] = L_0
            elif i == (N-1):
                L[i] = L_N
            else:
                L[i] = L[i-1] + L_increase
        return L

    def x_initial(N,Desired_Purity,First_stage_purity,Components):
        #initalise the array of x
        x = np.zeros((N,Components))
        #set the increase of x for the comp as linear interpolation
        x_increase = (Desired_Purity - First_stage_purity) / N
        #for loop to use linear interpolation for x (set water as x = index[2])
        for j in range(0,N):
            for i in range(0,(Components)):
                # set the 0 index for the inlet to the first stage puriy
                if j == 0:
                    if i == (Components - 1):
                        x[j, i] = First_stage_purity
                    else:
                        x[j,i] = 0.0
                elif j == (N-1):
                    if i == (Components - 1):
                        x[j,i] = Desired_Purity
                    else:
                        x[j,i] = 0.0
                else:
                    if i == (Components - 1):
                        x[j,i] = x_increase + x[j-1,i]
        return x

    def V_Flow(F, L):
        #length of L array
        length = L.shape[0]

        V = np.zeros(length)
        for i in range(length-1,-1,-1):
            if i == (length - 1):
                V[i] = (F[i] + L[i-1]) - L[i]

            elif i == 0:
                V[i] = (F[i] + V[i+1]) - L[i]
            else:
                V[i] = (F[i] + V[i+1] + L[i-1]) - L[i]

        return V

    def y_perfect(V,F,N,Feed_stage_amount):
        #Components
        Components = F.shape[1]
        y_ji = np.zeros((N,Components))

        if Feed_stage_amount == 1:
            for j in range(0, N):
                for i in range(0, Components):
                    if i == 0:
                        y_ji[j, i] = F[N - 1, i] / V[j]
                    elif i == 1:
                        y_ji[j, i] = 1 - y_ji[j, i - 1]
                    else:
                        y_ji[j, i] = 0.0
        else:
            F_sum = np.zeros(Components)
            for i in range(0,Components):
                F_sum[i] = np.sum(F[:,i])
                for j in range(0, N):
                    for i in range(0, Components):
                        if i == 0:
                            y_ji[j,i] = F_sum[i] / V[j]
                        elif i == 1:
                            y_ji[j,i] = 1 - y_ji[j,i-1]
                        else:
                            y_ji[j,i] = 0.0

        return y_ji

    def Min_Stages(N, Desired_Purity, Phase, Component,Feed_1,Feed_2,F,L0,S_F, Feed_Stage_amount, Temp, L_0, L_N, Mr,
                   x_inital, x_final,r,q,R,u,feed,Feed_Comp):
        Comps = Mr.shape[0]
        x_desired_new = 0.0  # initlaise for the while loop

        S = (np.sum(F) + np.sum(L0)) * S_F
        while x_desired_new < Desired_Purity:
            if Feed_Stage_amount == 1:
                F_ji = LSR_Inital_Guess.Feed_matrix(feed, N, S_F, Feed_Comp, Feed_Stage_amount)
                Feed = LSR_Inital_Guess.Sum_F(F_ji)
            else:
                Feed,F_ji = LSR_Inital_Guess.Min_Stage_Feed(N,Feed_1,Feed_2,F,S,L0)

            z = Mass_to_Mol.Mass_flow_mass_frac(F_ji)

            F_mol = Mass_to_Mol.mass_flow_to_mol(z, Feed, Mr)

            F_mol_Stage = LSR_Inital_Guess.Sum_F(F_mol)
            L = LSR_Inital_Guess.Liquid_Flow(L_0, L_N, N)

            x_mol_frac = LSR_Inital_Guess.x_initial(N, x_inital, x_final, Comps)

            V = LSR_Inital_Guess.V_Flow(F_mol_Stage, L)

            y_mol_frac = LSR_Inital_Guess.y_perfect(V, F_mol, N, Feed_Stage_amount)

            L_I, L_II, x_i, x_ii = Isotheral_Sum_Rates.Outer_Loop(x_mol_frac, y_mol_frac, r, q, R, Temp, u, L, V, F_mol,
                                                                  )

            if Phase == 1:
                    L_I_Comp = np.zeros((len(L_I), Comps))
                    L_I_Mass = np.zeros((len(L_I), Comps))
                    for j in range(0, len(L_I_Comp)):
                        for i in range(0, Comps):
                            L_I_Comp[j, i] = L_I[j] * x_i[j, i]  # Component Molar Flow
                            L_I_Mass[j, i] = L_I_Comp[j, i] * Mr[i]  # mass flow of each comp for each stage

                    # composition of the mass flow
                    x_mass_new = np.zeros((len(L_I), Comps))
                    L_I_mass = np.zeros(len(L_I))  # mass flow per stage (rename)
                    for j in range(0, len(L_I)):
                        L_I_mass[j] = np.sum(L_I_Mass[j, :])
                        for i in range(0, Comps):
                            x_mass_new[j, i] = L_I_Mass[j, i] / L_I_mass[j]
                    x_desired_new = x_mass_new[(N - 1), (Component - 1)]
                    print(x_desired_new, "x per iter of Stages")
                    print(x_mass_new[(N - 1), (Comps - 2)], "Mass frac of Acetone")
            else:
                x_desired_new = x_i[0, (Component - 1)]

            N = N + 1

        return N

    def S_F_Ratio(Solvent_Feed, Desired_Purity, Phase, Component, F, Feed_Comp, N, Feed_Stage_amount, Temp, L_0, L_N, Mr,
                   x_inital, x_final,r,q,R,u):
        Comps = Mr.shape[0]

        x_desired_new = 0.0

        while x_desired_new < Desired_Purity:
            F_ji = LSR_Inital_Guess.Feed_matrix(F, N, Solvent_Feed, Feed_Comp, Feed_Stage_amount)
            Feed = LSR_Inital_Guess.Sum_F(F_ji)

            z = Mass_to_Mol.Mass_flow_mass_frac(F_ji)

            F_mol = Mass_to_Mol.mass_flow_to_mol(z, Feed, Mr)

            F_mol_Stage = LSR_Inital_Guess.Sum_F(F_mol)
            L = LSR_Inital_Guess.Liquid_Flow(L_0, L_N, N)

            x_mol_frac = LSR_Inital_Guess.x_initial(N, x_inital, x_final, Comps)

            V = LSR_Inital_Guess.V_Flow(F_mol_Stage, L)

            y_mol_frac = LSR_Inital_Guess.y_perfect(V, F_mol, N, Feed_Stage_amount)

            L_I, L_II, x_i, x_ii = Isotheral_Sum_Rates.Outer_Loop(x_mol_frac, y_mol_frac, r, q, R, Temp, u, L, V, F_mol,
                                                              )
            #find mass fraction from the molar frac of x_ii
            if Phase == 1:
                L_I_Comp = np.zeros((len(L_I),Comps))
                L_I_Mass = np.zeros((len(L_I),Comps))
                for j in range(0,len(L_I_Comp)):
                    for i in range(0,Comps):
                        L_I_Comp[j,i] = L_I[j] * x_i[j,i] #Component Molar Flow
                        L_I_Mass[j,i] = L_I_Comp[j,i] * Mr[i] #mass flow of each comp for each stage

                #composition of the mass flow
                x_mass_new = np.zeros((len(L_I),Comps))
                L_I_mass = np.zeros(len(L_I)) #mass flow per stage (rename)
                for j in range(0,len(L_I)):
                    L_I_mass[j] = np.sum(L_I_Mass[j,:])
                    for i in range(0,Comps):
                        x_mass_new[j,i] = L_I_Mass[j,i] / L_I_mass[j]
                x_desired_new = x_mass_new[(N - 1), (Component - 1)]
                print(x_desired_new,"x per iter of S_F_Ratio")
                print(x_i[(N - 1), (Component - 1)], "mol frac of per iter")
                print(x_mass_new[(N-1),(Comps-2)], "Mass frac of Acetone")
            else:
                x_desired_new  = x_i[0, (Component - 1)]

            Solvent_Feed = Solvent_Feed + 0.01
            print(Solvent_Feed, "S/F Ratio per iter")
        return Solvent_Feed, x_desired_new
