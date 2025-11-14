import gurobipy as gp
from gurobipy import GRB
# We export the model as an MPS file for CPLEX to solve. (Cplex code given below)
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def load_piecewise_as_slope_intercept(
    txt_path: str,
    Kmax: int = 18,
    PART: int = 20,
    slope_scale: float = 10000.0
):
    """
    Load the C++ piecewise file 'seperation1_p20.txt'.
    For each s (1..Kmax), there are PART rows with columns [range, slope, placeholder].
    Convert them into: range_sa[s,a], kappa_sa[s,a], eta_sa[s,a].
    """
    data = np.loadtxt(txt_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[0] >= Kmax * PART

    data = data[:Kmax * PART, :3].reshape(Kmax, PART, 3)

    range_sa = data[:, :, 0].astype(float).copy()
    kappa_sa = (data[:, :, 1].astype(float) * slope_scale).copy()
    S_MAX, PART = range_sa.shape
    eta_sa = np.zeros_like(kappa_sa)

    for s in range(S_MAX):
        tau = np.zeros(PART + 1, dtype=float)
        for a in range(PART):
            tau[a+1] = tau[a] + float(range_sa[s, a])

        f_at_tau = np.zeros(PART + 1, dtype=float)
        for a in range(PART):
            f_at_tau[a+1] = f_at_tau[a] + float(kappa_sa[s, a]) * float(range_sa[s, a])

        for a in range(PART):
            eta_sa[s, a] = f_at_tau[a] - kappa_sa[s, a] * tau[a]

    range_sa = np.nan_to_num(range_sa)
    kappa_sa = np.nan_to_num(kappa_sa)
    eta_sa   = np.nan_to_num(eta_sa)
    return range_sa, kappa_sa, eta_sa


I = 4
T = 20
Kf_i = [1, 2, 2, 2]
Km = 3
Kf = sum(Kf_i)
K_total = Kf + Km
Delta = 0.5
mu = 197.4

PART = 20
Kmax_piece = 18
range_sa_full, kappa_sa_full, eta_sa_full = load_piecewise_as_slope_intercept(
    "seperation1_p20.txt", Kmax=Kmax_piece, PART=PART, slope_scale=10000.0
)

S_MAX = K_total
assert S_MAX <= Kmax_piece

range_sa = range_sa_full[:S_MAX, :]
kappa_sa = kappa_sa_full[:S_MAX, :]
eta_sa   = eta_sa_full[:S_MAX, :]

move_df = pd.read_csv("move_ave.csv", header=None)
c_ijt = np.repeat(move_df.values[:, :, np.newaxis], T, axis=2)
print("Loaded c_ijt:", c_ijt.shape)

lambda_df = pd.read_csv("arrive_xiao_0.csv", header=None)
lambda_df = lambda_df.dropna(axis=1, how='all')
lambda_df = lambda_df.iloc[:, :4]
lambda_it = lambda_df.to_numpy(dtype=float)
print("Loaded lambda_it:", lambda_it.shape)


G = range(I)
TT = range(T)
K_fixed = range(Kf)
K_mob = range(Kf, K_total)
K_all = range(K_total)

alpha_ik = np.zeros((I, K_total), dtype=int)
cur = 0
for i in G:
    for _ in range(Kf_i[i]):
        alpha_ik[i, cur] = 1
        cur += 1

S_i0 = np.zeros(I, dtype=int)

L_it = np.array([[60]*T, [80]*T, [80]*T, [60]*T])

LBD = 1
UBD = 4
Rmin = 0.5

LBD_p = int(round(LBD / Delta))
UBD_p = int(round(UBD / Delta))
Rmin_p = int(round(Rmin / Delta))

Q_it = [[0 for _ in TT] for _ in G]
delta_iq = [[] for _ in G]

for i in G:
    cuts = set()
    for j in G:
        for t in TT:
            tij = c_ijt[i, j, t]
            tji = c_ijt[j, i, t]
            if 0 < tij < Delta: cuts.add(round(tij, 6))
            if 0 < tji < Delta: cuts.add(round(tji, 6))

    cuts = sorted(list(cuts | {0.0, Delta}))
    delta_list = [cuts[k+1] - cuts[k] for k in range(len(cuts)-1)]
    delta_iq[i] = delta_list

    for t in TT:
        Q_it[i][t] = len(delta_list)

print("Sub-periods per entrance:", [len(delta_iq[i]) for i in G])
print("Example delta_iq[0] =", delta_iq[0])


M_arrive = [[[[[0 for _ in range(Q_it[j][t])]
               for t in TT] for t in TT] for j in G] for i in G]

for i in G:
    for j in G:
        for t in TT:
            move_time = c_ijt[i, j, t]
            if move_time <= 0:
                q_last = Q_it[j][t] - 1
                M_arrive[i][j][t][t][q_last] = 1
                continue

            acc = 0.0
            Qj = Q_it[j][t]
            for q in range(Qj):
                acc += delta_iq[j][q]
                if move_time <= acc + 1e-8:
                    M_arrive[i][j][t][t][q] = 1
                    break
            else:
                M_arrive[i][j][t][t][Qj - 1] = 1


USE_EXPLICIT_PWL = True


m = gp.Model("SASMQ")


x = m.addVars(I, I, K_mob, T, vtype=GRB.BINARY, name="x")
xi = m.addVars(I, K_all, T, vtype=GRB.BINARY, name="xi")
v = m.addVars(K_all, T, vtype=GRB.BINARY, name="v")
gamma = m.addVars(I, K_mob, T, vtype=GRB.BINARY, name="gamma")
zeta = m.addVars(I, K_mob, T, vtype=GRB.BINARY, name="zeta")
r = m.addVars(K_all, T, vtype=GRB.BINARY, name="r")
h = m.addVars(K_all, T, vtype=GRB.BINARY, name="h")

S = {}
lq = {}
z = {}
rou = {}

for i in G:
    for t in TT:
        Q = Q_it[i][t]
        for q in range(Q):
            S[i, t, q] = m.addVar(lb=1, vtype=GRB.INTEGER, name=f"S[{i},{t},{q}]")
            lq[i, t, q] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"l[{i},{t},{q}]")
            z[i, t, q] = m.addVars(K_total, vtype=GRB.BINARY, name=f"z[{i},{t},{q},*]")

l = m.addVars(I, T, lb=0.0, name="l_period")

obj = gp.quicksum((t+1) * (h[k, t] - r[k, t]) for k in K_all for t in TT)
m.setObjective(obj, GRB.MINIMIZE)


for k in K_mob:
    for t in TT:
        m.addConstr(gp.quicksum(x[i, j, k, t] for i in G for j in G) == 1)

for k in K_mob:
    for t in TT:
        for i in G:
            m.addConstr(gamma[i, k, t] >= gp.quicksum(x[i, j, k, t] for j in G) - (1 - v[k, t]))

for k in K_mob:
    for j in G:
        for t2 in TT:
            lhs = gamma[j, k, t2]
            rhs = []
            for i in G:
                for t in TT:
                    if t > t2:
                        continue
                    Qj = Q_it[j][t2]
                    for q in range(Qj):
                        if M_arrive[i][j][t][t2][q] == 1:
                            rhs.append(x[i, j, k, t])
            if rhs:
                m.addConstr(lhs <= gp.quicksum(rhs))


for i in G:
    for k in K_all:
        for t in TT:
            if k in K_fixed:
                if alpha_ik[i, k] == 1:
                    pass
            else:
                m.addConstr(gamma[i, k, t] >= xi[i, k, t])

for i in G:
    for k in K_mob:
        m.addConstr(gamma[i, k, 0] >= xi[i, k, 0])

for k in K_all:
    for t in TT:
        if t == 0:
            m.addConstr(r[k, t] >= v[k, t])
        else:
            m.addConstr(r[k, t] >= v[k, t] - v[k, t-1])

        m.addConstr(r[k, t] <= v[k, t])

        if t == 0:
            m.addConstr(r[k, t] <= 1)
        else:
            m.addConstr(r[k, t] <= 1 - v[k, t-1])


for k in K_all:
    for t in TT:
        if t == T-1:
            m.addConstr(h[k, t] >= v[k, t])
        else:
            m.addConstr(h[k, t] >= v[k, t] - v[k, t+1])
        m.addConstr(h[k, t] <= v[k, t])
        if t == T-1:
            m.addConstr(h[k, t] <= 1)
        else:
            m.addConstr(h[k, t] <= 1 - v[k, t+1])


for k in K_all:
    for t in TT:
        rng = range(t, min(T, t + UBD_p + 1))
        m.addConstr(gp.quicksum(v[k, u] for u in rng) <= UBD_p)

for k in K_all:
    for t in TT:
        rng = range(t, min(T, t + LBD_p))
        m.addConstr(gp.quicksum(v[k, u] for u in rng) >= LBD_p * r[k, t])

for k in K_all:
    for t in TT:
        rng = range(t+1, min(T, t + 1 + Rmin_p))
        if rng:
            m.addConstr(gp.quicksum(1 - v[k, u] for u in rng) >= Rmin_p * h[k, t])


M_BIG = 1e6

for i in G:
    for t in TT:
        Q = Q_it[i][t]
        for q in range(Q):

            m.addConstr(
                gp.quicksum(z[i, t, q][s] for s in range(K_total)) == 1
            )

            m.addConstr(
                gp.quicksum((s+1)*z[i, t, q][s] for s in range(K_total)) == S[i, t, q]
            )

            rou[i, t, q] = m.addVars(K_total, lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                     name=f"rou[{i},{t},{q},*]")

            if q == 0:
                if t == 0:
                    l_prev = 0.0
                else:
                    Qprev = Q_it[i][t-1]
                    l_prev = lq[i, t-1, Qprev-1]
            else:
                l_prev = lq[i, t, q-1]

            lam = float(lambda_it[t, i])
            delta = float(delta_iq[i][q])

            for s in range(K_total):
                s_eff = s + 1

                left = l_prev + lam * delta
                right = lq[i, t, q] + mu * delta * s_eff * rou[i, t, q][s]

                m.addConstr(
                    left >= right - (1 - z[i, t, q][s]) * M_BIG
                )
                m.addConstr(
                    left <= right + (1 - z[i, t, q][s]) * M_BIG
                )

                for v1 in range(PART):
                    kappa = float(kappa_sa[s, v1])
                    eta_ = float(eta_sa[s, v1])
                    m.addConstr(
                        lq[i, t, q] >= kappa * rou[i, t, q][s] + eta_ - (1 - z[i, t, q][s]) * M_BIG
                    )

        m.addConstr(l[i, t] == lq[i, t, Q-1])
        m.addConstr(l[i, t] <= L_it[i, t])


eta = m.addVars(I, K_mob, T, vtype=GRB.BINARY, name="eta")

for k in K_mob:
    for t in TT:
        m.addConstr(gp.quicksum(zeta[i, k, t] for i in G) == r[k, t])

for k in K_mob:
    for t in TT:
        m.addConstr(gp.quicksum(eta[i, k, t] for i in G) == h[k, t])

for k in K_mob:
    for t in TT:
        if t > 0:
            for i in G:
                m.addConstr(zeta[i, k, t] <= gamma[i, k, t-1])

for k in K_mob:
    for t in TT:
        for i in G:
            m.addConstr(eta[i, k, t] <= gamma[i, k, t])


for i in G:
    for t in TT:
        S_begin = S[i, t, 0]
        if t == 0:
            S_prev_end = 0.0
        else:
            Qprev = Q_it[i][t-1]
            S_prev_end = S[i, t-1, Qprev-1]

        mob_start_minus_stop = gp.quicksum(zeta[i, k, t] for k in K_mob) \
                               - (gp.quicksum(eta[i, k, t-1] for k in K_mob) if t > 0 else 0)

        fix_start_minus_stop = gp.quicksum(alpha_ik[i, k] * r[k, t] for k in K_all) \
                               - (gp.quicksum(alpha_ik[i, k] * h[k, t-1] for k in K_all) if t > 0 else 0)

        depart_out = 0.0
        if t > 0:
            depart_out = gp.quicksum(x[i, j, k, t-1] for k in K_mob for j in G if j != i)

        m.addConstr(
            S_begin - S_prev_end == mob_start_minus_stop + fix_start_minus_stop - depart_out
        )

for i in G:
    for t in TT:
        Qcur = Q_it[i][t]
        for q in range(Qcur - 1):
            arrivals_this_cut = []
            for j in G:
                if j == i:
                    continue
                for k in K_mob:
                    if M_arrive[j][i][t][t][q+1] == 1:
                        arrivals_this_cut.append(x[j, i, k, t])

            if arrivals_this_cut:
                m.addConstr(
                    S[i, t, q+1] - S[i, t, q] == gp.quicksum(arrivals_this_cut)
                )
            else:
                m.addConstr(
                    S[i, t, q+1] == S[i, t, q]
                )

for i in G:
    t0, q0 = 0, 0
    fixed_on  = gp.quicksum(alpha_ik[i, k] * v[k, t0] for k in K_all)
    mobile_on = gp.quicksum(xi[i, k, t0]         for k in K_mob)
    m.addConstr(
        fixed_on + mobile_on == S[i, t0, q0]
    )


for k in K_fixed:
    for t in TT:
        for i in G:
            if alpha_ik[i, k] == 1:
                m.addConstr(xi[i, k, t] <= v[k, t])
            else:
                m.addConstr(xi[i, k, t] == 0)

for k in K_mob:
    for t in TT:
        for i in G:
            m.addConstr(xi[i, k, t] <= gamma[i, k, t])
            m.addConstr(xi[i, k, t] <= v[k, t])


for k in K_all:
    for t in TT:
        m.addConstr(gp.quicksum(xi[i, k, t] for i in G) <= 1)


def check_M_arrive_coverage():
    missing = []
    for i in G:
        for j in G:
            for t in TT:
                Qj = Q_it[j][t]
                ok = any(M_arrive[i][j][t][t][q] == 1 for q in range(Qj))
                if not ok and i != j:
                    missing.append((i, j, t))
    if missing:
        print("[WARN] Missing M_arrive entries:", missing[:10], " total:", len(missing))

check_M_arrive_coverage()

def attach_staff_logging():
    for i in G:
        for t in TT:
            staff_it = gp.quicksum(xi[i, k, t] for k in K_all)
            for q in range(Q_it[i][t]):
                m.addConstr(S[i, t, q] <= staff_it + 1000)
attach_staff_logging()

m.Params.InfUnbdInfo = 1
m.optimize()

if m.Status == GRB.INFEASIBLE:
    print("Model infeasible; computing IIS ...")
    m.computeIIS()
    m.write("conflict.ilp")
    print("IIS written to conflict.ilp")

m.Params.MIPGap    = 0.01
m.Params.TimeLimit = 3600

if m.SolCount > 0:
    print(f"Status={m.Status}, Obj={m.ObjVal:.4f}")
    for i in G:
        print("Entrance", i, "period-end queues:",
              [l[i, t].X for t in TT])
else:
    print("No feasible solution.")


# ============================================================
# CPLEX code example (C++), used after exporting the model as .mps
# ============================================================
#
#     IloModel model(env);
#     IloObjective obj;
#     IloNumVarArray vars(env);
#     IloRangeArray rng(env);
#
#     IloCplex cplex(env);
#
#     // Import model from MPS file
#     cplex.importModel(model, "model.mps", obj, vars, rng);
#     cplex.extract(model);
#
#     // Set parameters: 2 threads, MIPGap=0.01, time limit = 3600s
#     cplex.setParam(IloCplex::EpGap, 0.01);
#     cplex.setParam(IloCplex::TiLim, 3600);
#
#     // Optional logging
#     // cplex.setParam(IloCplex::Param::MIP::Display, 4);
#
#     if (!cplex.solve()) {
#         env.out() << "No solution found. Status = " << cplex.getStatus() << "\n";
#     }
#     else {
#         env.out() << "Status: " << cplex.getStatus() << "\n";
#         env.out() << "Obj:    " << cplex.getObjValue() << "\n";
#         cplex.writeSolution("solution.sol");
#     }
#
#     cout << "The run time is: "
#          << (double)(clock() - start_time) / CLOCKS_PER_SEC
#          << "s" << endl;
#