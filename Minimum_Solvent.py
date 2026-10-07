
class Minimum_Solvent:

    def tie_line_range(Curve_II,x_LN,x_F):

        assert np.all(np.diff(Curve_II[:,1]) > -1e-12), "raffinate curve is not monotone in acetone"

        distance = np.sum((Curve_II - x_LN)**2,axis=1)
        k_first = np.argmin(distance)
        k_last = np.where(Curve_II[:,1] <= x_F[1])[0].max()

        return k_first, k_last

    def pinch_positions(E,R,x_S,x_LN):
        d = x_LN[0:2] - x_S[0:2]
        tie = R[:,0:2] - E[:,0:2]
        S_E = x_S[0:2] - E[:,0:2]

        denominator = tie[:,0] * d[1] - tie[:,1] * d[0]
        b = (S_E[:,0] * tie[:,1] - S_E[:,1] * tie[:,0]) / denominator

        return b

    def pinch_tie_line(Curve_I,Curve_II,x_S,x_LN,k_first,k_last):
        k = np.arange(k_first,k_last + 1)
        b = Minimum_Solvent.pinch_positions(Curve_I[k],Curve_II[k],x_S,x_LN)

        k_pinch = k[np.argmax(b)]
        assert k_pinch > k_first and k_pinch < k_last, "pinch tie line is at the end of the range"

        Delta = x_S + np.max(b) * (x_LN - x_S)

        return k_pinch, k, b, Delta

    def limiting_extract(x_F,Delta,Curve_I):
        crossings = Hunter_Nash.line_crossings(Curve_I,x_F,Delta)
        direction = x_F[0:2] - Delta[0:2]

        x_V1 = None
        t_min = 1e9
        for c in crossings:
            t_line = np.dot(c[0:2] - Delta[0:2],direction) / np.dot(direction,direction)
            #t = 1 is the feed
            if t_line > 1.0 and t_line < t_min:
                t_min = t_line
                x_V1 = c
        assert x_V1 is not None, "no extract point found"

        return x_V1

    def solvent_flow(L0,x_V1,x_LN):
        A = np.column_stack([x_V1,x_LN,-np.array([1.0,0.0,0.0])])
        flows = np.linalg.solve(A,L0)
        assert np.all(flows > 0), "negative flow in the overall balance"

        return flows[0], flows[1], flows[2]



#minimum solvent rate
k_first, k_last = Minimum_Solvent.tie_line_range(Curve_II,x_LN,x_L0)
k_pinch, k_all, b_all, Delta_min = Minimum_Solvent.pinch_tie_line(Curve_I,Curve_II,x_VN1,x_LN,k_first,k_last)

x_V1_min = Minimum_Solvent.limiting_extract(x_L0,Delta_min,Curve_I)
V1_min_flow, LN_min_flow, S_min_flow = Minimum_Solvent.solvent_flow(L0,x_V1_min,x_LN)

#mixing point of the minimum solvent rate
M_min_flow = L0 + np.array([S_min_flow,0.0,0.0])
M_min = Hunter_Nash.mol_frac(M_min_flow)

#solvent to feed ratios (mass basis as in the problem statement, and mole basis)
S_F_min = S_min_flow * Mr[0] / np.sum(feed_mass)
S_F_min_mol = S_min_flow / np.sum(L0)

Dvec_min = L0 - V1_min_flow * x_V1_min
assert np.allclose(Dvec_min / np.sum(Dvec_min),Delta_min,atol=1e-6)
assert np.allclose(Dvec_min,LN_min_flow * x_LN - np.array([S_min_flow,0.0,0.0]),atol=1e-6)
assert np.isclose(S_F_min_mol,(x_L0[1] - M_min[1]) / (M_min[1] - x_VN1[1]),rtol=1e-8)

k_V1 = np.argmin(np.sum((Curve_I - x_V1_min)**2,axis=1))
assert k_V1 > k_pinch and k_V1 <= k_last, "extract of the first stage is not beyond the pinch tie line"

if __name__ == "__main__":
    np.set_printoptions(precision=4,suppress=True)

    print(x_LN,"raffinate LN")
    print(Curve_I[k_pinch],"extract end of the pinch tie line")
    print(Curve_II[k_pinch],"raffinate end of the pinch tie line")
    print(np.max(b_all),"position of the pinch point on the line O-L (1 = LN)")
    print(Delta_min,"difference point at the minimum solvent rate")
    print(x_V1_min,"extract V1 at the minimum solvent rate")
    print(M_min,"mixing point at the minimum solvent rate")
    print(V1_min_flow,"V1 [mol]",LN_min_flow,"LN [mol]",S_min_flow,"solvent [mol]")
    print(S_F_min,"minimum solvent to feed ratio (mass)")
    print(S_F_min_mol,"minimum solvent to feed ratio (mol)")

    for factor in [1.5,1.2,1.1,1.05,1.02]:
        solvent_mass = np.array([100.0 * S_F_min * factor,0.0,0.0])
        V_N1 = Hunter_Nash.mass_to_mol(solvent_mass,Mr)
        M_flow = L0 + V_N1
        x_V1 = Hunter_Nash.extract_point(x_LN,Hunter_Nash.mol_frac(M_flow),Curve_I)
        V1_flow, LN_flow = Hunter_Nash.lever_flows(M_flow,x_V1,x_LN)
        Vs, Ls, ops, L_flow, V_flow, Delta = Hunter_Nash.stage_steps(x_V1,x_LN,L0 - V1_flow * x_V1,Curve_I,Curve_II,2000)
        n_ideal, N_frac = Hunter_Nash.fractional_stages(Ls,x_LN)
        print("S/F =",factor * S_F_min,"(",factor,"x minimum ):",N_frac,"stages")
