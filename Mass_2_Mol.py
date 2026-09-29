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

#molar Mass

class Mass_to_Mol:
    def mass_flow_to_mol(z,F,Mr):
        #rows
        rows = z.shape[1]
        #Cols
        cols = z.shape[0]

        #initalise mol frac and mass flowrate
        z_mol = np.zeros((cols,rows))
        F_ji = np.zeros((cols,rows))

        for j in range(0,cols):
            for i in range(0,rows):
                F_ji[j,i] = z[j,i] * F[j]
                z_mol[j,i] = F_ji[j,i] / Mr[i]

        return z_mol

    def Mass_flow_mass_frac(F):
        Stages = F.shape[0]
        Components = F.shape[1]

        #initalse flow of each stage and z
        F_stage = np.zeros(Stages)
        z = np.zeros((Stages,Components))
        for j in range(0,Stages):
            F_stage[j] = np.sum(F[j,:])
            for i in range(0,Components):
                if F[j,i] == 0:
                    z[j,i] = 0.0
                else:
                    z[j,i] = F[j,i] / F_stage[j]
        return z

    def mass_flow_mol_frac(z,F,Mr):
        #molar flow rate
        # rows
        rows = z.shape[1]
        # Cols
        cols = z.shape[0]

        mol_flow = Mass_to_Mol.mass_flow_to_mol(z,F,Mr)
        #molar flow to frac
        Stage_Flow = np.zeros(cols)
        mol_frac = np.zeros((cols,rows))

        for j in range(0,cols):
            Stage_Flow[j] = np.sum(mol_flow[j,:])
            for i in range(0,rows):
                if mol_flow[j,i] > 0:
                    mol_frac[j,i] = mol_flow[j,i] / Stage_Flow[j]
                else:
                    mol_frac[j,i] = 0.0

        return mol_frac
