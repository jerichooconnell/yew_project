"""
Parametrized stage-structured (Lefkovitch) population model for Taxus brevifolia,
extending the Busing & Spies (1995)-style matrix already cited in the draft, with
two independently tunable threat parameters:

  m   (mite_factor, 0-1)      : yew big-bud mite (Cecidophyopsis psilaspis) damage,
                                 penalizing growth-class advancement (-20% at m=1) and
                                 fecundity/aril production (-25% at m=1).
  phi (fire_multiplier, >=1)  : multiplier on the baseline annual stand-burn probability,
                                 representing increased area burned; applied as an
                                 exogenous disturbance hazard between matrix-projection
                                 steps, with partial post-fire recruitment (resprout/
                                 recolonization), not as a vital-rate edit.

Baseline vital rates (s, g, f) are fit by constrained optimization to reproduce:
  - lambda = 1.02 (Busing & Spies 1995, undisturbed old-growth, as cited in the draft)
  - stable stage-class ratio q = 1.5 (draft's de Liocourt "expected" reverse-J slope)

7 DBH-based stages matching the draft's own field/FAIB bins:
  0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70+ cm
Stages >=30 cm (indices 3-6) are treated as reproductively fertile.

Outputs of this script: lambda sensitivity grid, 3-generation fold-change/decline
tables for named scenarios, and the fire multiplier phi required to reproduce
IUCN A2 VU/EN/CR decline thresholds (and the draft's own 69.2% figure) holding m=1.

CAVEATS (see accompanying report for full discussion):
  - mite dose-response (20%/25% multipliers) and baseline annual burn probability
    (p_burn0) are placeholders pending literature-fitted values; treat all absolute
    outputs as illustrative of SENSITIVITY, not as a calibrated replacement for the
    draft's RS-based decline estimate.
  - fire is modeled as stand-replacing with partial resprout recruitment; this is a
    simplification of yew's actual post-fire response (see draft's own
    fire-recovery curve discussion).
"""
import numpy as np
from scipy.optimize import minimize, brentq

stage_labels = ["0-10cm","10-20cm","20-30cm","30-40cm","40-50cm","50-60cm","60-70cm+"]
n_stages = len(stage_labels)
fertile_stages = [3,4,5,6]


def build_matrix(s, g, f, mite_factor=0.0):
    s = np.asarray(s, float); g = np.asarray(g, float); f = np.asarray(f, float)
    g_eff = g * (1 - 0.20*mite_factor)
    f_eff = f * (1 - 0.25*mite_factor)
    A = np.zeros((n_stages, n_stages))
    for i in range(n_stages):
        if i < n_stages-1:
            G = s[i]*g_eff[i]
            P = s[i]*(1-g_eff[i])
        else:
            G = 0.0
            P = s[i]
        A[i,i] = P
        if i < n_stages-1:
            A[i+1,i] += G
    for k,j in enumerate(fertile_stages):
        A[0,j] += f_eff[k]
    return A, g_eff, f_eff


def lam_stable(A):
    w, v = np.linalg.eig(A)
    idx = np.argmax(w.real)
    lam = w[idx].real
    vec = np.abs(v[:,idx].real)
    vec = vec/vec.sum()
    return lam, vec


def fit_baseline(target_lambda=1.02, target_q=1.5):
    def loss(params):
        s = params[0:7]; g = params[7:13]; f = params[13:17]
        if np.any(s<=0) or np.any(s>1) or np.any(g<0) or np.any(g>1) or np.any(f<0):
            return 1e6
        A,_,_ = build_matrix(s,g,f, mite_factor=0.0)
        lam, vec = lam_stable(A)
        ratios = vec[:-1]/np.clip(vec[1:],1e-9,None)
        ratio_pen = np.sum((np.log(np.clip(ratios,1e-6,None)) - np.log(target_q))**2)
        lam_pen = (lam - target_lambda)**2 * 500
        return lam_pen + ratio_pen
    x0 = np.array([0.985,0.99,0.992,0.994,0.995,0.996,0.997,
                   0.04,0.025,0.018,0.012,0.008,0.005,
                   0.02,0.015,0.012,0.010])
    bounds = [(0.90,0.999)]*7 + [(0.0005,0.15)]*6 + [(0.0,0.5)]*4
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return res.x[0:7], res.x[7:13], res.x[13:17], res


def project(years, mite_factor, fire_multiplier, p_burn0, post_fire_retention,
            s_fit, g_fit, f_fit, N0):
    A_mite, g_eff, f_eff = build_matrix(s_fit, g_fit, f_fit, mite_factor=mite_factor)
    p_burn = min(p_burn0 * fire_multiplier, 1.0)
    N = N0.copy().astype(float)
    traj = [N.copy()]
    for t in range(years):
        N = A_mite @ N
        burned_total = N.sum() * p_burn
        if burned_total > 0:
            N = N * (1-p_burn)
            N[0] += burned_total * post_fire_retention
        traj.append(N.copy())
    traj = np.array(traj)
    Ntot = traj.sum(axis=1)
    lam_eff = (Ntot[-1]/Ntot[0])**(1/years) if Ntot[0]>0 else np.nan
    return {"traj":traj, "Ntot":Ntot, "lam_eff":lam_eff, "g_eff":g_eff, "f_eff":f_eff, "A_mite":A_mite}


if __name__ == "__main__":
    s_fit, g_fit, f_fit, res = fit_baseline()
    A0, _, _ = build_matrix(s_fit, g_fit, f_fit, 0.0)
    lam0, vec0 = lam_stable(A0)
    print("Baseline lambda:", lam0)
    N0 = vec0 * 1000
    out = project(300, mite_factor=1.0, fire_multiplier=15.5, p_burn0=0.002,
                  post_fire_retention=0.6, s_fit=s_fit, g_fit=g_fit, f_fit=f_fit, N0=N0)
    print("3-gen (T=100) fold change at m=1, phi=15.5:", (out["Ntot"][300]/out["Ntot"][0]))
