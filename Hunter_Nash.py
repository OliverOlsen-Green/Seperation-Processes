from UNIQUAC import UNIQUAC
from UNIQUAC import u, R, T, r, q
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import math
class Hunter_Nash:

    def mass_to_mol(F_mass,Mr):
        #molar flow of each component converted from mass flow given
        Components = len(Mr)
        F_mol = np.zeros(Components)
        for i in range(0,Components):
            F_mol[i] = F_mass[i] / Mr[i]

        return F_mol

    def mol_frac(F_mol):
        Components = len(F_mol)
        F_total = np.sum(F_mol)
        x = np.zeros(Components)
        for i in range(0,Components):
            x[i] = F_mol[i] / F_total

        return x

    def acetone_mass_frac(x,Mr):
        x = np.atleast_2d(x)
        mass = x * Mr
        w_acetone = mass[:,1] / np.sum(mass,axis=1)

        return w_acetone

    def tie_line_curves(data_I,data_II,points):
        Tie_lines = data_I.shape[0]
        Components = data_I.shape[1]

        t = np.arange(0,Tie_lines)
        t_fine = np.linspace(0,Tie_lines - 1,points)

        Curve_I = np.zeros((points,Components))
        Curve_II = np.zeros((points,Components))
        for i in range(0,Components):
            Curve_I[:,i] = PchipInterpolator(t,data_I[:,i])(t_fine)
            Curve_II[:,i] = PchipInterpolator(t,data_II[:,i])(t_fine)

        #normalise so that every point sums to one
        Curve_I = Curve_I / np.sum(Curve_I,axis=1,keepdims=True)
        Curve_II = Curve_II / np.sum(Curve_II,axis=1,keepdims=True)

        return Curve_I, Curve_II

    def conjugate(x,Curve_from,Curve_to):
        #tie line through x, nearest point on the curve it is on and the point with the same tie line number on the other curve
        distance = np.sum((Curve_from - x)**2,axis=1)
        k = np.argmin(distance)

        return Curve_from[k], Curve_to[k]

    def line_crossings(Curve,p0,p1):
        f = (Curve[:,0] - p0[0]) * (p1[1] - p0[1]) - (Curve[:,1] - p0[1]) * (p1[0] - p0[0])
        index = np.where(f[:-1] * f[1:] < 0)[0]

        crossings = []
        for k in index:
            fraction = f[k] / (f[k] - f[k+1])
            crossings.append(Curve[k] + fraction * (Curve[k+1] - Curve[k]))

        return crossings

    def raffinate_target(Curve_II,Mr,w_target):
        #raffinate leaving stage N is on the water rich curve at the required mass fraction of acetone
        w_acetone = Hunter_Nash.acetone_mass_frac(Curve_II,Mr)
        f = w_acetone - w_target
        index = np.where(f[:-1] * f[1:] < 0)[0]
        #the mass fraction of acetone must pass the target only once
        assert len(index) == 1, "raffinate target is not unique"

        k = index[0]
        fraction = f[k] / (f[k] - f[k+1])
        x_LN = Curve_II[k] + fraction * (Curve_II[k+1] - Curve_II[k])

        return x_LN

    def extract_point(x_LN,M,Curve_I):
        #the extract V1 is on the line raffinate LN - mixing point M, beyond M, where it cuts the benzene rich curve
        crossings = Hunter_Nash.line_crossings(Curve_I,x_LN,M)
        direction = M[0:2] - x_LN[0:2]

        x_V1 = None
        t_min = 1e9
        for c in crossings:
            t_line = np.dot(c[0:2] - x_LN[0:2],direction) / np.dot(direction,direction)
            #t = 1 is the mixing point
            if t_line > 1.0 and t_line < t_min:
                t_min = t_line
                x_V1 = c
        assert x_V1 is not None, "no extract point found"

        return x_V1

    def lever_flows(M_flow,x_V1,x_LN):
        A = np.column_stack([x_V1,x_LN])
        sol, residual, rank, sv = np.linalg.lstsq(A,M_flow,rcond=None)
        error = np.max(np.abs(A @ sol - M_flow))
        assert error < 1e-6, "overall balance does not close"

        return sol[0], sol[1]

    def stage_steps(x_V1,x_LN,Dvec,Curve_I,Curve_II,N_max):
        #difference point (net flow), it is outside the triangle
        Dtot = np.sum(Dvec)
        Delta = Dvec / Dtot

        Vs = [x_V1.copy()]   #extract leaving stage j
        Ls = []              #raffinate leaving stage j
        ops = []             #operating lines (L_j, V_j+1)
        L_flow = []
        V_flow = []

        for n in range(1,N_max + 1):
            #tie line from the extract of stage n to the raffinate of stage n
            x_I, x_L = Hunter_Nash.conjugate(Vs[-1],Curve_I,Curve_II)
            Ls.append(x_L)

            #stop when the raffinate has reached the target
            if x_L[1] <= x_LN[1]:
                break

            #operating line from the raffinate of stage n through the difference point to the extract of stage n+1
            crossings = Hunter_Nash.line_crossings(Curve_I,x_L,Delta)

            found = False
            for y in crossings:
                #net flow balance Dvec = L_n x_n - V_n+1 y_n+1
                A = np.column_stack([x_L,-y])
                flows = np.linalg.lstsq(A,Dvec,rcond=None)[0]
                error = np.max(np.abs(A @ flows - Dvec))
                if flows[0] > 0 and flows[1] > 0 and error < 1e-6:
                    ops.append((x_L,y))
                    L_flow.append(flows[0])
                    V_flow.append(flows[1])
                    Vs.append(y)
                    found = True
                    break
            assert found, "no operating line point on stage " + str(n)

        return Vs, Ls, ops, L_flow, V_flow, Delta

    def fractional_stages(Ls,x_LN):
        #number of whole stages and the fraction of the last stage, interpolated on the acetone fraction of the raffinate
        n_ideal = len(Ls)
        a_prev = Ls[-2][1]
        a_last = Ls[-1][1]
        fraction = (a_prev - x_LN[1]) / (a_prev - a_last)

        return n_ideal, (n_ideal - 1) + fraction

    def check_tie_lines(data_I,data_II,r,q,R,T,u):
        #iso-activity of the tie line data, x_i gamma_i must be equal in both phases
        Tie_lines = data_I.shape[0]
        error = np.zeros(Tie_lines)
        for k in range(0,Tie_lines):
            gamma_I = UNIQUAC.gamma(data_I[k,:],r,q,R,T,u)
            gamma_II = UNIQUAC.gamma(data_II[k,:],r,q,R,T,u)
            ln_a_I = np.log(data_I[k,:] * gamma_I)
            ln_a_II = np.log(data_II[k,:] * gamma_II)
            error[k] = np.max(np.abs(ln_a_I - ln_a_II))

        return error


Mr = np.array([78.11,58.08,18.015])       #benzene, acetone, water
feed_mass = np.array([0.0,50.0,50.0])     #50 wt% acetone, 50 wt% water
S_F = 1.5                                 #solvent to feed ratio (mass)
solvent_mass = np.array([100.0 * S_F,0.0,0.0])    #pure benzene
w_target = 0.01                           #raffinate 1 wt% acetone

L0 = Hunter_Nash.mass_to_mol(feed_mass,Mr)
V_N1 = Hunter_Nash.mass_to_mol(solvent_mass,Mr)
x_L0 = Hunter_Nash.mol_frac(L0)
x_VN1 = Hunter_Nash.mol_frac(V_N1)

M_flow = L0 + V_N1
M = Hunter_Nash.mol_frac(M_flow)

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()[1:]
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()[1:]

Curve_I, Curve_II = Hunter_Nash.tie_line_curves(data_I,data_II,200001)

x_LN = Hunter_Nash.raffinate_target(Curve_II,Mr,w_target)
x_V1 = Hunter_Nash.extract_point(x_LN,M,Curve_I)
V1_flow, LN_flow = Hunter_Nash.lever_flows(M_flow,x_V1,x_LN)

Dvec = L0 - V1_flow * x_V1

assert np.allclose(Dvec,LN_flow * x_LN - V_N1,atol=1e-6)


Vs, Ls, ops, L_flow, V_flow, Delta = Hunter_Nash.stage_steps(x_V1,x_LN,Dvec,Curve_I,Curve_II,40)
n_ideal, N_frac = Hunter_Nash.fractional_stages(Ls,x_LN)

if __name__ == "__main__":
    np.set_printoptions(precision=4,suppress=True)

    tie_error = Hunter_Nash.check_tie_lines(data_I,data_II,r,q,R,T,u)
    print(np.max(tie_error),"largest iso-activity error of the tie line data")

    print(L0,"feed L0 [mol]")
    print(V_N1,"solvent V_N+1 [mol]")
    print(x_L0,"composition of feed")
    print(M,"mixing point M")
    print(x_LN,"raffinate LN",Hunter_Nash.acetone_mass_frac(x_LN,Mr),"mass fraction of acetone")
    print(x_V1,"extract V1")
    print(V1_flow,"V1 [mol]",LN_flow,"LN [mol]")
    print(Dvec,"difference point flows [mol]")
    print(Delta,"difference point composition")
    for j in range(0,len(Ls)):
        print("stage",j + 1,"V =",Vs[j],"L =",Ls[j])
    print(n_ideal,"whole stages")
    print(N_frac,"stages including the fraction of the last stage")
from UNIQUAC import UNIQUAC
from UNIQUAC import u, R, T, r, q
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import math
class Hunter_Nash:

    def mass_to_mol(F_mass,Mr):
        #molar flow of each component converted from mass flow given
        Components = len(Mr)
        F_mol = np.zeros(Components)
        for i in range(0,Components):
            F_mol[i] = F_mass[i] / Mr[i]

        return F_mol

    def mol_frac(F_mol):
        Components = len(F_mol)
        F_total = np.sum(F_mol)
        x = np.zeros(Components)
        for i in range(0,Components):
            x[i] = F_mol[i] / F_total

        return x

    def acetone_mass_frac(x,Mr):
        x = np.atleast_2d(x)
        mass = x * Mr
        w_acetone = mass[:,1] / np.sum(mass,axis=1)

        return w_acetone

    def tie_line_curves(data_I,data_II,points):
        Tie_lines = data_I.shape[0]
        Components = data_I.shape[1]

        t = np.arange(0,Tie_lines)
        t_fine = np.linspace(0,Tie_lines - 1,points)

        Curve_I = np.zeros((points,Components))
        Curve_II = np.zeros((points,Components))
        for i in range(0,Components):
            Curve_I[:,i] = PchipInterpolator(t,data_I[:,i])(t_fine)
            Curve_II[:,i] = PchipInterpolator(t,data_II[:,i])(t_fine)

        #normalise so that every point sums to one
        Curve_I = Curve_I / np.sum(Curve_I,axis=1,keepdims=True)
        Curve_II = Curve_II / np.sum(Curve_II,axis=1,keepdims=True)

        return Curve_I, Curve_II

    def conjugate(x,Curve_from,Curve_to):
        #tie line through x, nearest point on the curve it is on and the point with the same tie line number on the other curve
        distance = np.sum((Curve_from - x)**2,axis=1)
        k = np.argmin(distance)

        return Curve_from[k], Curve_to[k]

    def line_crossings(Curve,p0,p1):
        #points where the straight line through p0 and p1 cuts the curve (benzene and acetone fractions, water follows from the sum)
        f = (Curve[:,0] - p0[0]) * (p1[1] - p0[1]) - (Curve[:,1] - p0[1]) * (p1[0] - p0[0])
        index = np.where(f[:-1] * f[1:] < 0)[0]

        crossings = []
        for k in index:
            fraction = f[k] / (f[k] - f[k+1])
            crossings.append(Curve[k] + fraction * (Curve[k+1] - Curve[k]))

        return crossings

    def raffinate_target(Curve_II,Mr,w_target):
        #raffinate leaving stage N is on the water rich curve at the required mass fraction of acetone
        w_acetone = Hunter_Nash.acetone_mass_frac(Curve_II,Mr)
        f = w_acetone - w_target
        index = np.where(f[:-1] * f[1:] < 0)[0]
        #the mass fraction of acetone must pass the target only once
        assert len(index) == 1, "raffinate target is not unique"

        k = index[0]
        fraction = f[k] / (f[k] - f[k+1])
        x_LN = Curve_II[k] + fraction * (Curve_II[k+1] - Curve_II[k])

        return x_LN

    def extract_point(x_LN,M,Curve_I):
        #the extract V1 is on the line raffinate LN - mixing point M, beyond M, where it cuts the benzene rich curve
        crossings = Hunter_Nash.line_crossings(Curve_I,x_LN,M)
        direction = M[0:2] - x_LN[0:2]

        x_V1 = None
        t_min = 1e9
        for c in crossings:
            t_line = np.dot(c[0:2] - x_LN[0:2],direction) / np.dot(direction,direction)
            #t = 1 is the mixing point
            if t_line > 1.0 and t_line < t_min:
                t_min = t_line
                x_V1 = c
        assert x_V1 is not None, "no extract point found"

        return x_V1

    def lever_flows(M_flow,x_V1,x_LN):
        #extract and raffinate flows from the overall balance M = V1 + LN (three balances, two unknowns)
        A = np.column_stack([x_V1,x_LN])
        sol, residual, rank, sv = np.linalg.lstsq(A,M_flow,rcond=None)
        error = np.max(np.abs(A @ sol - M_flow))
        assert error < 1e-6, "overall balance does not close"

        return sol[0], sol[1]

    def stage_steps(x_V1,x_LN,Dvec,Curve_I,Curve_II,N_max):
        #difference point (net flow), it is outside the triangle
        Dtot = np.sum(Dvec)
        Delta = Dvec / Dtot

        Vs = [x_V1.copy()]   #extract leaving stage j
        Ls = []              #raffinate leaving stage j
        ops = []             #operating lines (L_j, V_j+1)
        L_flow = []
        V_flow = []

        for n in range(1,N_max + 1):
            #tie line from the extract of stage n to the raffinate of stage n
            x_I, x_L = Hunter_Nash.conjugate(Vs[-1],Curve_I,Curve_II)
            Ls.append(x_L)

            #stop when the raffinate has reached the target
            if x_L[1] <= x_LN[1]:
                break

            #operating line from the raffinate of stage n through the difference point to the extract of stage n+1
            crossings = Hunter_Nash.line_crossings(Curve_I,x_L,Delta)

            found = False
            for y in crossings:
                #net flow balance Dvec = L_n x_n - V_n+1 y_n+1
                A = np.column_stack([x_L,-y])
                flows = np.linalg.lstsq(A,Dvec,rcond=None)[0]
                error = np.max(np.abs(A @ flows - Dvec))
                if flows[0] > 0 and flows[1] > 0 and error < 1e-6:
                    ops.append((x_L,y))
                    L_flow.append(flows[0])
                    V_flow.append(flows[1])
                    Vs.append(y)
                    found = True
                    break
            assert found, "no operating line point on stage " + str(n)

        return Vs, Ls, ops, L_flow, V_flow, Delta

    def fractional_stages(Ls,x_LN):
        #number of whole stages and the fraction of the last stage, interpolated on the acetone fraction of the raffinate
        n_ideal = len(Ls)
        a_prev = Ls[-2][1]
        a_last = Ls[-1][1]
        fraction = (a_prev - x_LN[1]) / (a_prev - a_last)

        return n_ideal, (n_ideal - 1) + fraction

    def check_tie_lines(data_I,data_II,r,q,R,T,u):
        Tie_lines = data_I.shape[0]
        error = np.zeros(Tie_lines)
        for k in range(0,Tie_lines):
            gamma_I = UNIQUAC.gamma(data_I[k,:],r,q,R,T,u)
            gamma_II = UNIQUAC.gamma(data_II[k,:],r,q,R,T,u)
            ln_a_I = np.log(data_I[k,:] * gamma_I)
            ln_a_II = np.log(data_II[k,:] * gamma_II)
            error[k] = np.max(np.abs(ln_a_I - ln_a_II))

        return error


Mr = np.array([78.11,58.08,18.015])       #benzene, acetone, water
feed_mass = np.array([0.0,50.0,50.0])     #50 wt% acetone, 50 wt% water
S_F = 1.5                                 #solvent to feed ratio (mass)
solvent_mass = np.array([100.0 * S_F,0.0,0.0])    #pure benzene
w_target = 0.01                           #raffinate 1 wt% acetone

L0 = Hunter_Nash.mass_to_mol(feed_mass,Mr)
V_N1 = Hunter_Nash.mass_to_mol(solvent_mass,Mr)
x_L0 = Hunter_Nash.mol_frac(L0)
x_VN1 = Hunter_Nash.mol_frac(V_N1)

M_flow = L0 + V_N1
M = Hunter_Nash.mol_frac(M_flow)

data_I = pd.read_csv("Phase_1_data.csv").to_numpy()[1:]
data_II = pd.read_csv("Phase_2_data.csv").to_numpy()[1:]

Curve_I, Curve_II = Hunter_Nash.tie_line_curves(data_I,data_II,200001)

x_LN = Hunter_Nash.raffinate_target(Curve_II,Mr,w_target)
x_V1 = Hunter_Nash.extract_point(x_LN,M,Curve_I)
V1_flow, LN_flow = Hunter_Nash.lever_flows(M_flow,x_V1,x_LN)

Dvec = L0 - V1_flow * x_V1

assert np.allclose(Dvec,LN_flow * x_LN - V_N1,atol=1e-6)


Vs, Ls, ops, L_flow, V_flow, Delta = Hunter_Nash.stage_steps(x_V1,x_LN,Dvec,Curve_I,Curve_II,40)
n_ideal, N_frac = Hunter_Nash.fractional_stages(Ls,x_LN)

if __name__ == "__main__":
    np.set_printoptions(precision=4,suppress=True)

    tie_error = Hunter_Nash.check_tie_lines(data_I,data_II,r,q,R,T,u)
    print(np.max(tie_error),"largest iso-activity error of the tie line data")

    print(L0,"feed L0 [mol]")
    print(V_N1,"solvent V_N+1 [mol]")
    print(x_L0,"composition of feed")
    print(M,"mixing point M")
    print(x_LN,"raffinate LN",Hunter_Nash.acetone_mass_frac(x_LN,Mr),"mass fraction of acetone")
    print(x_V1,"extract V1")
    print(V1_flow,"V1 [mol]",LN_flow,"LN [mol]")
    print(Dvec,"difference point flows [mol]")
    print(Delta,"difference point composition")
    for j in range(0,len(Ls)):
        print("stage",j + 1,"V =",Vs[j],"L =",Ls[j])
    print(n_ideal,"whole stages")
    print(N_frac,"stages including the fraction of the last stage")
