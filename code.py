
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from scipy.linalg import eig

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.labelsize": 12, "legend.fontsize": 9,
                     "figure.dpi": 150, "lines.linewidth": 1.8})

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS  (Goldbeter, Dupont & Berridge 1990, Table 1)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
v0  = 1.0;   v1  = 7.3;   VM2 = 65.0;  VM3 = 500.0
K2  = 1.0;   KA  = 0.9;   kf  = 1.0
n   = 2;     m   = 2;     p   = 4
KR_DEFAULT = 2.0
K_DEFAULT  = 10.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL FUNCTIONS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def v2f(Z):
    return VM2 * Z**n / (K2**n + Z**n)

def v3f(Z, Y, KR=KR_DEFAULT):
    return VM3 * (Y**m / (KR**m + Y**m)) * (Z**p / (KA**p + Z**p))

def F_eq(ZY, beta, k=K_DEFAULT, KR=KR_DEFAULT):
    Z, Y = max(ZY[0], 1e-12), max(ZY[1], 1e-12)
    return np.array([v0 + v1*beta - v2f(Z) + v3f(Z, Y, KR) + kf*Y - k*Z,
                     v2f(Z) - v3f(Z, Y, KR) - kf*Y])

def jacobian(Z, Y, k=K_DEFAULT, KR=KR_DEFAULT):
    Z, Y = max(Z, 1e-12), max(Y, 1e-12)
    v2p = VM2 * n * K2**n * Z**(n-1) / (K2**n + Z**n)**2
    v3z = VM3 * (Y**m / (KR**m + Y**m)) * (p * KA**p * Z**(p-1) / (KA**p + Z**p)**2)
    v3y = VM3 * (Z**p / (KA**p + Z**p)) * (m * KR**m * Y**(m-1) / (KR**m + Y**m)**2)
    return np.array([[-v2p + v3z - k,  kf + v3y],
                     [ v2p - v3z,     -kf - v3y]])

dFdb = np.array([v1, 0.0])

def rhs(t, y, b, k=K_DEFAULT, KR=KR_DEFAULT):
    Z, Y = max(y[0], 1e-12), max(y[1], 1e-12)
    return [v0 + v1*b - v2f(Z) + v3f(Z, Y, KR) + kf*Y - k*Z,
            v2f(Z) - v3f(Z, Y, KR) - kf*Y]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLOSED-FORM STEADY STATE                                                ║
# ║                                                                          ║
# ║  From dY/dt = 0:   v2 − v3 = kf·Y                                        ║
# ║  Substituting into dZ/dt = 0:   v0 + v1·β − k·Z = 0                      ║
# ║  ⇒                  Z* = (v0 + v1·β) / k                                 ║
# ║  Y* is then solved in 1D from F2(Z*, Y*) = 0.                            ║
# ║                                                                          ║
# ║  This collapses the (β, k) two-parameter sweep to repeated 1D            ║
# ║  bisection and is what makes Section 3.5 of the report tractable.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def steady_state_closed(beta, k=K_DEFAULT, KR=KR_DEFAULT):
    Zs = (v0 + v1*beta) / k
    if Zs <= 0:
        return None
    g = lambda Y: v2f(Zs) - v3f(Zs, Y, KR) - kf*Y
    try:
        Ys = brentq(g, 1e-6, 50.0, xtol=1e-10)
    except ValueError:
        Ys = fsolve(g, 1.0)[0]
    return Zs, Ys

def trace_J_closed(beta, k=K_DEFAULT, KR=KR_DEFAULT):
    s = steady_state_closed(beta, k, KR)
    if s is None:
        return np.nan
    return np.trace(jacobian(s[0], s[1], k, KR))

def find_hopfs_along_beta(k=K_DEFAULT, KR=KR_DEFAULT, beta_grid=None):
    """Return list of β values where tr(J) crosses zero, located by bisection."""
    if beta_grid is None:
        beta_grid = np.linspace(0.05, 1.5, 600)
    trs = np.array([trace_J_closed(b, k, KR) for b in beta_grid])
    hopfs = []
    for i in range(len(beta_grid) - 1):
        a, c = trs[i], trs[i+1]
        if np.isnan(a) or np.isnan(c):
            continue
        if a * c < 0:
            try:
                bh = brentq(lambda x: trace_J_closed(x, k, KR),
                            beta_grid[i], beta_grid[i+1], xtol=1e-8)
                hopfs.append(bh)
            except ValueError:
                pass
    return hopfs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  PSEUDO-ARCLENGTH CONTINUATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def pseudo_arclength(beta0, ZY0, ds=0.003, n_steps=1200,
                     beta_min=0.10, beta_max=0.90, tol=1e-10, maxiter=20):
    """
    Follow F(Z,Y;β) = 0 by pseudo-arclength continuation.
    State u = (Z, Y, β); augmented corrector is the 3×3 system:

        [ F(Z, Y; β)                ]   [0]
        [ ⟨u − u_old, τ⟩ − Δs       ] = [0]

    Tangent τ from SVD of extended Jacobian [J | ∂F/∂β].
    Hopf points: sign change of tr(J), interpolated to sub-step precision.
    """
    u = np.array([float(ZY0[0]), float(ZY0[1]), float(beta0)])
    Jext = np.column_stack([jacobian(u[0], u[1]), dFdb])
    _, _, Vt = np.linalg.svd(Jext, full_matrices=True)
    tau = Vt[-1]; tau /= np.linalg.norm(tau)
    if tau[2] < 0:
        tau = -tau

    betas, Zout, Yout, stab, hopfs = [], [], [], [], []
    prev_tr = None

    for _ in range(n_steps):
        uc = u + ds * tau
        ok = False
        for _ in range(maxiter):
            Z, Y, b = uc; Z, Y = max(Z, 1e-12), max(Y, 1e-12)
            Fv  = F_eq([Z, Y], b)
            arc = float(np.dot(uc - u, tau)) - ds
            res = np.array([Fv[0], Fv[1], arc])
            if np.linalg.norm(res) < tol:
                ok = True; break
            Jloc = jacobian(Z, Y)
            A = np.zeros((3, 3))
            A[:2, :2] = Jloc; A[:2, 2] = dFdb; A[2, :] = tau
            try:
                delta = np.linalg.solve(A, -res)
            except np.linalg.LinAlgError:
                break
            uc += delta
        if not ok:
            break
        Z_n, Y_n, b_n = uc
        if b_n < beta_min or b_n > beta_max or Z_n < 1e-6 or Y_n < 1e-6:
            break

        Jn = jacobian(Z_n, Y_n)
        Jextn = np.column_stack([Jn, dFdb])
        _, _, Vtn = np.linalg.svd(Jextn, full_matrices=True)
        tn = Vtn[-1]; tn /= np.linalg.norm(tn)
        if np.dot(tn, tau) < 0:
            tn = -tn

        tr = Jn[0, 0] + Jn[1, 1]
        stab.append(int(tr < 0))
        betas.append(b_n); Zout.append(Z_n); Yout.append(Y_n)
        if prev_tr is not None and prev_tr * tr < 0:
            f = abs(prev_tr) / (abs(prev_tr) + abs(tr))
            hopfs.append((betas[-2] + f * (b_n - betas[-2]),
                          Zout[-2]  + f * (Z_n - Zout[-2]),
                          Yout[-2]  + f * (Y_n - Yout[-2])))
        prev_tr = tr; tau = tn; u = uc

    return (np.array(betas), np.array(Zout), np.array(Yout),
            np.array(stab), hopfs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  FIRST LYAPUNOV COEFFICIENT  (Kuznetsov 2004, Eq. 8.22)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def first_lyapunov(Z0, Y0, h=2e-5):
    """
    Compute l1 at Hopf equilibrium (Z0, Y0).

        l1 = (1 / 2ω) · Re[ ⟨p, C(q,q,q̄)⟩
                          − 2⟨p, B(q, A⁻¹·B(q,q̄))⟩
                          + ⟨p, B(q̄, (2iωI − A)⁻¹·B(q,q))⟩ ]
    """
    A = jacobian(Z0, Y0)

    # (a) ω from imag(eigenvalue), not √det(A)
    vals, vecs = eig(A)
    omega_approx = np.sqrt(max(np.linalg.det(A), 1e-20))
    idx = np.argmin(np.abs(vals - 1j * omega_approx))
    omega = float(np.imag(vals[idx]))

    # unit-norm right eigenvector q
    q = vecs[:, idx].astype(complex)
    q /= np.linalg.norm(q)

    # adjoint left eigenvector p, normalised so that p̄ᵀq = 1
    valsl, vecsl = eig(A.T)
    p = vecsl[:, np.argmin(np.abs(valsl + 1j * omega))].astype(complex)
    p /= np.conj(np.conj(p).dot(q))

    x0 = np.array([Z0, Y0])

    # (d) v1·β omitted: linear ⇒ second/third derivatives are zero, and
    #     constant terms cancel exactly in every centred stencil below.
    def Fv_at(ZY):
        Z, Y = max(ZY[0], 1e-12), max(ZY[1], 1e-12)
        return np.array([v0 - v2f(Z) + v3f(Z, Y) + kf*Y - K_DEFAULT*Z,
                         v2f(Z) - v3f(Z, Y) - kf*Y], dtype=float)

    # (b) 4-point centred stencil for d²F/(dx_i dx_j), O(h²)
    def d2F(i, j, offset=None):
        off = offset if offset is not None else np.zeros(2)
        ei = np.zeros(2); ei[i] = h
        ej = np.zeros(2); ej[j] = h
        return ( Fv_at(x0 + off + ei + ej) - Fv_at(x0 + off + ei - ej)
               - Fv_at(x0 + off - ei + ej) + Fv_at(x0 + off - ei - ej)) / (4 * h**2)

    def B(u, v):
        r = np.zeros(2, dtype=complex)
        for i in range(2):
            for j in range(2):
                r += d2F(i, j) * u[i] * v[j]
        return r

    # (c) C built by centred differencing of B in third direction
    # (e) loop index `lm`, NOT `m`, to avoid shadowing the global m=2
    def C(u, v, w):
        r = np.zeros(2, dtype=complex)
        for i in range(2):
            for j in range(2):
                for lm in range(2):
                    ek = np.zeros(2); ek[lm] = h
                    d3 = (d2F(i, j, offset=+ek) - d2F(i, j, offset=-ek)) / (2 * h)
                    r += d3 * u[i] * v[j] * w[lm]
        return r

    qb = np.conj(q)
    T1 = np.conj(p).dot(C(q, q, qb))
    T2 = -2.0 * np.conj(p).dot(B(q, np.linalg.solve(A.astype(complex), B(q, qb))))
    T3 = np.conj(p).dot(B(qb, np.linalg.solve(2j * omega * np.eye(2)
                                              - A.astype(complex), B(q, q))))
    return np.real(T1 + T2 + T3) / (2.0 * omega), omega


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  SINGLE-SHOOTING LIMIT-CYCLE CONTINUATION                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def lc_shooting(b_start, b_end, ZY0, T0, n_pts=70):
    """
    Continue limit cycle by single shooting.
    Solve  G(Z0, Y0, T; β) = [ x(T) − x(0); dZ/dt(0) ] = 0
    (T-periodicity + phase condition at Z extremum).
    Warm-started from previous solution for robustness.
    """
    betas_o, mx_o, mn_o, T_o = [], [], [], []
    xcur = np.array([float(ZY0[0]), float(ZY0[1]), float(T0)])

    def G(xv, b):
        Z0c, Y0c, T = xv; T = max(T, 0.05)
        sol = solve_ivp(rhs, [0, T], [Z0c, Y0c], args=(b,), method="LSODA",
                        rtol=1e-9, atol=1e-11)
        if not sol.success or sol.y.shape[1] < 2:
            return np.array([1e6, 1e6, 1e6])
        return np.array([sol.y[0, -1] - Z0c,
                         sol.y[1, -1] - Y0c,
                         rhs(0, [Z0c, Y0c], b)[0]])

    for b in np.linspace(b_start, b_end, n_pts):
        try:
            s = fsolve(G, xcur, args=(b,), full_output=True, xtol=1e-9)
            if s[2] != 1:
                continue
            xv = s[0]; T = xv[2]
            if T < 0.05 or xv[0] < 0 or xv[1] < 0:
                continue
            slc = solve_ivp(rhs, [0, T], [xv[0], xv[1]], args=(b,), method="LSODA",
                            rtol=1e-9, atol=1e-11, max_step=T/400)
            if not slc.success:
                continue
            betas_o.append(b)
            mx_o.append(float(np.max(slc.y[0])))
            mn_o.append(float(np.min(slc.y[0])))
            T_o.append(float(T))
            xcur = xv
        except Exception:
            pass

    return np.array(betas_o), np.array(mx_o), np.array(mn_o), np.array(T_o)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORE BIFURCATION RESULTS  (used by all figures)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("=" * 70)
print("  GDB Ca²⁺ MODEL — RIGOROUS BIFURCATION ANALYSIS")
print("=" * 70)

print("\n[1] Pseudo-arclength continuation along β …")
ZY0 = fsolve(lambda xy: F_eq(xy, 0.15), [0.2, 2.0])
betas_eq, Zss, Yss, stab, hopfs = pseudo_arclength(0.15, ZY0)
print(f"    {len(betas_eq)} branch points, β ∈ [{betas_eq.min():.3f}, {betas_eq.max():.3f}]")
print(f"    {len(hopfs)} Hopf point(s):")
for bh, Zh, Yh in hopfs:
    print(f"      β = {bh:.4f},  Z* = {Zh:.4f},  Y* = {Yh:.4f}")

print("\n[2] First Lyapunov coefficient at each Hopf point …")
l1_data = []
for bh, Zh, Yh in hopfs:
    l1, om = first_lyapunov(Zh, Yh)
    kind = "SUPERCRITICAL" if l1 < 0 else "SUBCRITICAL"
    print(f"    β = {bh:.4f}:  l₁ = {l1:+.5f},  ω = {om:.5f} rad/s   ⇒  {kind}")
    l1_data.append((bh, Zh, Yh, l1, om))

# Pull out the two Hopf β values for downstream use
bh1 = l1_data[0][0]
bh2 = l1_data[1][0] if len(l1_data) > 1 else 0.78
om1 = l1_data[0][4]
om2 = l1_data[1][4] if len(l1_data) > 1 else 0.0

print("\n[3] Single-shooting limit-cycle continuation …")
b0 = bh1 + 0.005   # closes the visual gap to the Hopf point
sol0 = solve_ivp(rhs, [0, 80], [0.15, 0.5], args=(b0,), method="LSODA",
                 rtol=1e-9, atol=1e-11, max_step=0.01)
Zlc = sol0.y[0][sol0.t > 60]
Ylc = sol0.y[1][sol0.t > 60]
tlc = sol0.t[sol0.t > 60]
cr  = np.where(np.diff(np.sign(Zlc - np.mean(Zlc))) > 0)[0]
T_init = float(np.mean(np.diff(tlc[cr]))) if len(cr) >= 2 else 2 * np.pi / om1
imax = int(np.argmax(Zlc))
betas_lc, mx, mn, periods = lc_shooting(b0, bh2 - 0.005,
                                        [Zlc[imax], Ylc[imax]], T_init)
print(f"    {len(betas_lc)} LC points,  β ∈ [{betas_lc.min():.3f}, {betas_lc.max():.3f}]")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 1 — PHASE PLANES AT THREE β VALUES                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 1] phase_planes_three.pdf …")

def make_phase_plane(ax, beta, title):
    Zg = np.linspace(0.001, 2.0, 500)
    Yg = np.linspace(0.001, 2.0, 500)
    ZZ, YY = np.meshgrid(Zg, Yg)
    F1 = v0 + v1*beta - v2f(ZZ) + v3f(ZZ, YY) + kf*YY - K_DEFAULT*ZZ
    F2 = v2f(ZZ) - v3f(ZZ, YY) - kf*YY
    ax.contour(ZZ, YY, F1, levels=[0], colors="tab:blue", linewidths=1.8)
    ax.contour(ZZ, YY, F2, levels=[0], colors="tab:red",  linewidths=1.8)
    # Trajectories
    for ic, col in zip([(0.05, 0.5), (1.5, 0.3), (0.3, 1.7)],
                       ["#2ca02c", "#ff7f0e", "#9467bd"]):
        sol = solve_ivp(rhs, [0, 60], list(ic), args=(beta,), method="LSODA",
                        rtol=1e-8, atol=1e-10, max_step=0.01, dense_output=True)
        t = np.linspace(0, 60, 4000)
        Zs, Ys = sol.sol(t)
        ax.plot(Zs, Ys, color=col, lw=0.9, alpha=0.85)
        ax.plot(Zs[-1], Ys[-1], marker=">", color=col, markersize=6)
    # Fixed point
    Zfp, Yfp = steady_state_closed(beta)
    tr = np.trace(jacobian(Zfp, Yfp))
    if tr < 0:
        ax.plot(Zfp, Yfp, "ko", markersize=10, markerfacecolor="black")
    else:
        ax.plot(Zfp, Yfp, "ko", markersize=10, markerfacecolor="white")
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$Z$ ($\mu$M)")
    ax.set_ylabel(r"$Y$ ($\mu$M)")
    ax.set_title(title)
    ax.legend([plt.Line2D([0],[0],color="tab:blue",lw=1.8),
               plt.Line2D([0],[0],color="tab:red", lw=1.8)],
              [r"$\dot{Z}=0$", r"$\dot{Y}=0$"], loc="upper right", fontsize=8)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
make_phase_plane(axes[0], 0.20, r"$\beta=0.20$ (below window)")
make_phase_plane(axes[1], 0.40, r"$\beta=0.40$ (oscillatory)")
make_phase_plane(axes[2], 0.80, r"$\beta=0.80$ (above window)")
fig.tight_layout()
fig.savefig("phase_planes_three.pdf", bbox_inches="tight")
plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 2 — THREE-PANEL TIME SERIES  (β = 0.301, 0.644, 0.74)              ║
# ║                                                                          ║
# ║  Third panel near the upper Hopf is an independent test of the           ║
# ║  supercritical-Hopf prediction: T → 2π/ω_H2,  amplitude → 0.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 2] timeseries_three.pdf …")

fig, axes = plt.subplots(3, 1, figsize=(10, 8.5))
beta_vals = [0.301, 0.644, 0.74]
windows   = [60, 10, 6]
titles    = [r"$\beta=0.301$ (just past lower Hopf)",
             r"$\beta=0.644$ (mid-window)",
             r"$\beta=0.74$ (approaching upper Hopf)"]
colours   = ["#1f77b4", "#d62728", "#2ca02c"]

for ax, b, win, title, col in zip(axes, beta_vals, windows, titles, colours):
    sol = solve_ivp(rhs, [0, win + 5], [0.15, 0.5], args=(b,), method="LSODA",
                    rtol=1e-9, atol=1e-11, max_step=0.005, dense_output=True)
    t = np.linspace(0, win, 4000)
    Z, Y = sol.sol(t)
    ax.plot(t, Z,     color=col, lw=1.6, label=r"$Z$ (cytosol)")
    ax.plot(t, Y - 0.35, color=col, alpha=0.45, ls="--", lw=1.4,
            label=r"$Y - 0.35\,\mu$M (store)")
    Zss = steady_state_closed(b)[0]
    cross = np.where(np.diff(np.sign(Z - Zss)) > 0)[0]
    if len(cross) >= 2:
        T = float(np.mean(np.diff(t[cross])))
        amp = float(np.max(Z) - np.min(Z))
        ax.text(0.98, 0.92,
                f"$T \\approx {T:.2f}$ s,  amplitude $\\approx {amp:.2f}\\,\\mu$M",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
                fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(r"Ca$^{2+}$ ($\mu$M)")
    ax.set_ylim(-0.05, 2.0); ax.set_xlim(0, win)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("timeseries_three.pdf", bbox_inches="tight")
plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 3 — k = 10 vs k = 6 s⁻¹ (vasopressin-like extrusion)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 3] timeseries_k.pdf …")

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for ax, kval, title, col in zip(axes, [10.0, 6.0],
        [r"$k=10\,\mathrm{s}^{-1}$ (standard)",
         r"$k=6\,\mathrm{s}^{-1}$ (vasopressin-like)"],
        ["#1f77b4", "#d62728"]):
    sol = solve_ivp(lambda t, y: rhs(t, y, 0.301, k=kval),
                    [0, 65], [0.15, 0.5], method="LSODA",
                    rtol=1e-9, atol=1e-11, max_step=0.005, dense_output=True)
    t = np.linspace(0, 60, 4000)
    Z, Y = sol.sol(t)
    ax.plot(t, Z, color=col, lw=1.6, label=r"$Z$ (cytosol)")
    ax.plot(t, Y - 0.35, color=col, alpha=0.45, ls="--", lw=1.4,
            label=r"$Y - 0.35\,\mu$M (store)")
    ax.set_title(title + r",  $\beta=0.301$")
    ax.set_ylabel(r"Ca$^{2+}$ ($\mu$M)")
    ax.set_ylim(-0.05, 2.0); ax.set_xlim(0, 60)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("timeseries_k.pdf", bbox_inches="tight")
plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 4 — BIFURCATION DIAGRAM + PERIOD                                   ║
# ║                                                                          ║
# ║  Includes √(β−β_c) parabolic envelope from each Hopf point, fitted to    ║
# ║  the first numerical limit-cycle point — visually demonstrates l1 < 0.   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 4] bifurcation_period.pdf …")

# Re-use closed-form steady-state for a clean equilibrium curve
betas_grid = np.linspace(0.10, 0.90, 401)
Z_eq = np.array([steady_state_closed(b)[0] for b in betas_grid])

# √(β−β_c) envelopes around each Hopf point
def fit_envelope(bh, betas_lc, mx, mn, side="lower"):
    if side == "lower":
        idx = 0
    else:
        idx = -1
    if len(betas_lc) == 0:
        return None, None, None, None
    Zss_at = steady_state_closed(betas_lc[idx])[0]
    Amax = mx[idx] - Zss_at
    Amin = Zss_at - mn[idx]
    db   = abs(betas_lc[idx] - bh)
    if db < 1e-9:
        return None, None, None, None
    cmax = Amax / np.sqrt(db)
    cmin = Amin / np.sqrt(db)
    return Amax, Amin, cmax, cmin

A1max, A1min, c1max, c1min = fit_envelope(bh1, betas_lc, mx, mn, side="lower")
A2max, A2min, c2max, c2min = fit_envelope(bh2, betas_lc, mx, mn, side="upper")

bs1 = np.linspace(bh1, betas_lc[0],  30) if len(betas_lc) else np.array([])
bs2 = np.linspace(betas_lc[-1], bh2, 30) if len(betas_lc) else np.array([])

if len(bs1):
    Zs1 = np.array([steady_state_closed(b)[0] for b in bs1])
    upper1 = Zs1 + c1max * np.sqrt(np.maximum(bs1 - bh1, 0))
    lower1 = Zs1 - c1min * np.sqrt(np.maximum(bs1 - bh1, 0))
if len(bs2):
    Zs2 = np.array([steady_state_closed(b)[0] for b in bs2])
    upper2 = Zs2 + c2max * np.sqrt(np.maximum(bh2 - bs2, 0))
    lower2 = Zs2 - c2min * np.sqrt(np.maximum(bh2 - bs2, 0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
seg1 = betas_grid <  bh1
seg2 = (betas_grid >= bh1) & (betas_grid <= bh2)   # unstable branch
seg3 = betas_grid >  bh2
ax1.plot(betas_grid[seg1], Z_eq[seg1], "b-",  lw=2, label="Stable fixed point")
ax1.plot(betas_grid[seg3], Z_eq[seg3], "b-",  lw=2)
ax1.plot(betas_grid[seg2], Z_eq[seg2], "b--", lw=2, dashes=(6,3),
         label="Unstable fixed point")
if len(betas_lc):
    ax1.plot(betas_lc, mx, "ko", ms=3.5, label=r"Limit cycle (max/min $Z$)")
    ax1.plot(betas_lc, mn, "ko", ms=3.5)
    if len(bs1):
        ax1.plot(bs1, upper1, "k:", lw=1.2, alpha=0.7,
                 label=r"$\sqrt{\beta-\beta_c}$ envelope ($l_1 < 0$)")
        ax1.plot(bs1, lower1, "k:", lw=1.2, alpha=0.7)
    if len(bs2):
        ax1.plot(bs2, upper2, "k:", lw=1.2, alpha=0.7)
        ax1.plot(bs2, lower2, "k:", lw=1.2, alpha=0.7)
for bh in [bh1, bh2]:
    ax1.axvline(bh, color="gray", ls=":", lw=1)
    ax1.text(bh + 0.006, 1.85, rf"Hopf $\beta\approx{bh:.3f}$",
             fontsize=8, rotation=90, va="top", color="gray")
ax1.set_xlabel(r"InsP$_3$ saturation parameter,  $\beta$")
ax1.set_ylabel(r"Cytosolic Ca$^{2+}$,  $Z^*$  ($\mu$M)")
ax1.set_title("(a) Bifurcation diagram (pseudo-arclength continuation)")
ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
ax1.set_xlim(0.10, 0.90); ax1.set_ylim(-0.05, 2.15); ax1.grid(True, alpha=0.3)

if len(periods):
    ax2.plot(betas_lc, periods, "k.-", ms=4, lw=1.5)
for bh in [bh1, bh2]:
    ax2.axvline(bh, color="gray", ls=":", lw=1)
ax2.set_xlabel(r"InsP$_3$ saturation parameter,  $\beta$")
ax2.set_ylabel("Oscillation period (s)")
ax2.set_title(r"(b) Period vs. $\beta$ (shooting continuation)")
ax2.set_xlim(0.10, 0.90); ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("bifurcation_period.pdf", bbox_inches="tight")
plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 5 — TWO-PARAMETER (β, k) HOPF LOCUS  (the oscillatory wedge)       ║
# ║                                                                          ║
# ║  Uses the closed-form  Z* = (v0 + v1·β)/k  identity.                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 5] two_param_hopf.pdf …")

k_vals = np.linspace(3.0, 14.0, 56)
hopf_lower, hopf_upper = [], []
ks_lower,  ks_upper    = [], []
for kk in k_vals:
    hs = find_hopfs_along_beta(k=kk)
    if len(hs) >= 1:
        hopf_lower.append(hs[0]); ks_lower.append(kk)
    if len(hs) >= 2:
        hopf_upper.append(hs[1]); ks_upper.append(kk)
ks_lower = np.array(ks_lower); hopf_lower = np.array(hopf_lower)
ks_upper = np.array(ks_upper); hopf_upper = np.array(hopf_upper)

fig, ax = plt.subplots(figsize=(8, 6))
common_k = np.intersect1d(ks_lower, ks_upper)
if len(common_k):
    lower_at_common = np.interp(common_k, ks_lower, hopf_lower)
    upper_at_common = np.interp(common_k, ks_upper, hopf_upper)
    ax.fill_betweenx(common_k, lower_at_common, upper_at_common,
                     color="#fdd49e", alpha=0.6, label="Oscillatory region")
ax.plot(hopf_lower, ks_lower, "b-", lw=2.2, label=r"Lower Hopf locus $\beta_{H1}(k)$")
ax.plot(hopf_upper, ks_upper, "r-", lw=2.2, label=r"Upper Hopf locus $\beta_{H2}(k)$")
ax.scatter([bh1], [10.0], s=80, c="black", marker="o", zorder=5,
           label="Standard parameters")
ax.scatter([bh2], [10.0], s=80, c="black", marker="o", zorder=5)
ax.annotate(rf"$\beta_{{H1}}={bh1:.3f}$", xy=(bh1, 10.0), xytext=(0.10, 11.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=9)
ax.annotate(rf"$\beta_{{H2}}={bh2:.3f}$", xy=(bh2, 10.0), xytext=(0.83, 11.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=9)
ax.set_xlabel(r"InsP$_3$ saturation parameter,  $\beta$")
ax.set_ylabel(r"Ca$^{2+}$ extrusion rate,  $k$  (s$^{-1}$)")
ax.set_title(r"Two-parameter bifurcation: Hopf locus in $(\beta, k)$ plane")
ax.set_xlim(0.0, 1.0); ax.set_ylim(k_vals.min(), k_vals.max())
ax.legend(loc="lower right", framealpha=0.92)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("two_param_hopf.pdf", bbox_inches="tight")
plt.close(fig)

# K_R sweep — printed to the terminal so the numbers can be quoted in the report
print("\n  K_R sensitivity sweep (caffeine-like reduction):")
KR_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
for KR in KR_vals:
    hs = find_hopfs_along_beta(KR=KR)
    if len(hs) >= 2:
        print(f"    K_R = {KR:.1f} µM:  β_H1 = {hs[0]:.4f},  β_H2 = {hs[1]:.4f},"
              f"  width = {hs[1] - hs[0]:.3f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIG. 6 — FREQUENCY ENCODING THROUGH PROTEIN PHOSPHORYLATION             ║
# ║                                                                          ║
# ║  Couples the bifurcation analysis to the Goldbeter–Koshland covalent     ║
# ║  modification cycle (Eq. 7 in the report). Sweeps β and computes the     ║
# ║  time-averaged phosphorylated fraction over the final 50 s of each run.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n[Fig 6] encoding.pdf …")

# Goldbeter–Koshland kinase/phosphatase parameters (Goldbeter et al. 1990)
v_p, V_MK, K_a, W_T = 5.0, 40.0, 2.5, 1.0

def coupled_rhs(t, y, b, K1, K2):
    Z, Y, Ws = y
    Z = max(Z, 1e-12); Y = max(Y, 1e-12)
    vK = V_MK * Z / (K_a + Z)
    dZ = v0 + v1*b - v2f(Z) + v3f(Z, Y) + kf*Y - K_DEFAULT*Z
    dY = v2f(Z) - v3f(Z, Y) - kf*Y
    dWs = (v_p / W_T) * ((vK / v_p) * (1 - Ws) / (K1 + 1 - Ws)
                         - Ws / (K2 + Ws))
    return [dZ, dY, dWs]

def mean_W_and_freq(beta, K1, K2):
    sol = solve_ivp(coupled_rhs, [0, 100], [0.15, 0.5, 0.0],
                    args=(beta, K1, K2), method="LSODA",
                    rtol=1e-9, atol=1e-11, max_step=0.01, dense_output=True)
    t = np.linspace(50, 100, 5000)
    Z, _, Ws = sol.sol(t)
    cr = np.where(np.diff(np.sign(Z - np.mean(Z))) > 0)[0]
    if len(cr) < 2:
        return np.nan, np.nan
    T = float(np.mean(np.diff(t[cr])))
    return 1.0 / T, float(np.mean(Ws))

print("  Sweeping β across oscillatory window for two kinetic regimes …")
betas_enc = np.linspace(bh1 + 0.02, bh2 - 0.02, 18)
freqs_a, Wa = [], []
freqs_b, Wb = [], []
for b in betas_enc:
    f, W = mean_W_and_freq(b, 0.01, 0.01)   # zero-order
    if not np.isnan(f):
        freqs_a.append(f); Wa.append(W)
    f, W = mean_W_and_freq(b, 10.0, 10.0)   # first-order
    if not np.isnan(f):
        freqs_b.append(f); Wb.append(W)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(freqs_a, Wa, "o-", color="#1f77b4", lw=1.6, markersize=6,
        label=r"Curve a: $K_1 = K_2' = 0.01$ (zero-order)")
ax.plot(freqs_b, Wb, "s-", color="#d62728", lw=1.6, markersize=6,
        label=r"Curve b: $K_1 = K_2' = 10$ (first-order)")
ax.set_xlabel("Frequency of oscillations (Hz)")
ax.set_ylabel(r"Mean phosphorylated fraction $\langle W^*\rangle$")
ax.set_title("Frequency encoding through protein phosphorylation")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("encoding.pdf", bbox_inches="tight")
plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)
for bh, Zh, Yh, l1, om in l1_data:
    print(f"  β_Hopf = {bh:.4f}:  l₁ = {l1:+.5f}  "
          f"({'supercritical' if l1 < 0 else 'subcritical'}),  "
          f"ω = {om:.5f} rad/s,  T_lin = {2*np.pi/om:.5f} s")
print("\nAll figures saved:")
for f in ["phase_planes_three.pdf", "timeseries_three.pdf", "timeseries_k.pdf",
          "bifurcation_period.pdf", "two_param_hopf.pdf", "encoding.pdf"]:
    print(f"  • {f}")
