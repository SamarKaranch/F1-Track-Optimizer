"""
F1 Minimum Lap Time Optimizer
==============================
Loads track_data.npz from the geometry pipeline and solves the
minimum lap time SOCP using CVXPY with SCP iterations.
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# STEP 1 — Load track data
# =========================================================================

data = np.load(os.path.join(OUTPUT_DIR, "track_data.npz"))
s = data["s"]
cx = data["cx"]
cy = data["cy"]
kappa_ref = data["kappa_ref"]
n_min = data["n_min"]
n_max = data["n_max"]
L = float(data["L"].flat[0])
N = len(s)
ds = L / N

print(f"Track loaded: N={N}, ds={ds:.2f} m, L={L:.1f} m")

# =========================================================================
# STEP 2 — Vehicle parameters
# =========================================================================

m = 798.0
g = 9.81
mu = 1.8
Cd = 0.9
w_veh = 2.0
v_max = 90.0
v_min = 5.0
F_accel_max = 10000.0
F_brake_max = 18000.0
alpha_max = 0.15

# =========================================================================
# STEP 3 — Scaling factors
# =========================================================================

E_scale = 0.5 * m * v_max**2
F_scale = mu * m * g
E_max_n = 1.0
E_min_n = (0.5 * m * v_min**2) / E_scale
F_accel_n = F_accel_max / F_scale
F_brake_n = F_brake_max / F_scale
c_drag = 2 * Cd / m
c_Fx = F_scale / E_scale
c_Fy = E_scale / F_scale
sqrt_2m_n = np.sqrt(2 * m / E_scale)

print(f"E_scale={E_scale:.1f} J,  F_scale={F_scale:.1f} N")
print(f"E_min_n={E_min_n:.4f},  E_max_n={E_max_n:.4f}")
print(f"F_accel_n={F_accel_n:.4f},  F_brake_n={F_brake_n:.4f}")

# =========================================================================
# STEP 4 — Decision variables (defined once, reused each iteration)
# =========================================================================

n = cp.Variable(N + 1)
alpha = cp.Variable(N + 1)
E = cp.Variable(N + 1)
p = cp.Variable(N + 1)
Fx = cp.Variable(N)
Fy = cp.Variable(N + 1)
kappa_path_v = cp.Variable(N)  # path curvature — now a variable

objective = cp.Minimize(cp.sum(p[:N]) * ds)

# =========================================================================
# STEP 5 — SCP loop
# =========================================================================

# Linearization points
kappa_path_prev = kappa_ref.copy()
E_prev = np.full(N + 1, 0.5)  # scaled, mid-range initial guess

n_prev = np.zeros(N)
max_iter = 15
tol = 1e-3

# Store results
n_opt = None
E_opt = None
Fx_opt = None
v_opt = None
lap_time = None

for iteration in range(max_iter):
    print(f"\n── SCP Iteration {iteration+1} ──────────────────────────")

    constraints = []

    # --- Track boundary ---
    constraints += [
        n[:N] >= n_min + w_veh / 2,
        n[:N] <= n_max - w_veh / 2,
        n[N] >= n_min[0] + w_veh / 2,
        n[N] <= n_max[0] - w_veh / 2,
    ]

    # --- Longitudinal dynamics ---
    for i in range(N):
        constraints.append((E[i + 1] - E[i]) / ds == Fx[i] * c_Fx - c_drag * E[i])

    # --- Kinematics + linearized bilinear Fy ---
    for i in range(N):
        # lateral position
        constraints.append((n[i + 1] - n[i]) / ds == alpha[i])
        # path curvature = ref + heading rate
        constraints.append(
            kappa_path_v[i] == kappa_ref[i] + (alpha[i + 1] - alpha[i]) / ds
        )
        # Fy = 2 * kappa_path * E  (bilinear — linearize via first-order Taylor)
        # kappa * E ≈ kappa_prev*E + kappa*E_prev - kappa_prev*E_prev
        constraints.append(
            Fy[i]
            == 2
            * c_Fy
            * (
                kappa_path_prev[i] * E[i]
                + kappa_path_v[i] * E_prev[i]
                - kappa_path_prev[i] * E_prev[i]
            )
        )

    # --- Friction circle (SOCP) ---
    for i in range(N):
        constraints.append(cp.norm(cp.vstack([Fx[i], Fy[i]]), 2) <= 1.0)

    # --- Velocity-pace coupling (rotated SOCP) ---
    for i in range(N + 1):
        constraints.append(
            cp.norm(cp.vstack([sqrt_2m_n, p[i] - E[i]]), 2) <= p[i] + E[i]
        )

    # --- Box constraints ---
    constraints += [
        E >= E_min_n,
        E <= E_max_n,
        Fx >= -F_brake_n,
        Fx <= F_accel_n,
        alpha >= -alpha_max,
        alpha <= alpha_max,
        p >= 0,
    ]

    # --- Periodicity ---
    constraints += [
        n[0] == n[N],
        alpha[0] == alpha[N],
        E[0] == E[N],
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"  Solver failed: {prob.status}")
        break

    # Extract results
    lap_time = prob.value
    n_opt = n.value[:N]
    alpha_opt = alpha.value[:N]
    E_opt = E.value[:N] * E_scale
    Fx_opt = Fx.value * F_scale
    Fy_opt = Fy.value[:N] * F_scale
    v_opt = np.sqrt(2 * E_opt / m)

    print(f"  Lap time : {lap_time:.3f} s  ({lap_time/60:.2f} min)")
    print(f"  Max speed: {v_opt.max()*3.6:.1f} km/h")
    print(f"  Min speed: {v_opt.min()*3.6:.1f} km/h")
    print(f"  Max |n|  : {np.abs(n_opt).max():.3f} m")

    # Update linearization points
    kappa_path_prev = kappa_path_v.value
    E_prev = E.value

    # Convergence check
    delta = np.max(np.abs(n_opt - n_prev))
    print(f"  Δn       : {delta:.6f} m")
    if delta < tol and iteration > 0:
        print(f"  Converged at iteration {iteration+1}")
        break
    n_prev = n_opt.copy()

print(f"\nFinal lap time: {lap_time:.3f} s  ({lap_time/60:.2f} min)")

# =========================================================================
# STEP 6 — Reconstruct racing line in world coordinates
# =========================================================================

dx_c = np.roll(cx, -1) - np.roll(cx, 1)
dy_c = np.roll(cy, -1) - np.roll(cy, 1)
mag = np.sqrt(dx_c**2 + dy_c**2) + 1e-12
nx_c = -dy_c / mag
ny_c = dx_c / mag

rx = cx + n_opt * nx_c
ry = cy + n_opt * ny_c

# =========================================================================
# STEP 7 — Plot results
# =========================================================================


def plot_results(cx, cy, rx, ry, v_opt, n_opt, Fx_opt, s, L, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d0d0d")
    fig.suptitle(
        title + " — Optimal Racing Line",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    # Recompute boundaries
    dx_c = np.roll(cx, -1) - np.roll(cx, 1)
    dy_c = np.roll(cy, -1) - np.roll(cy, 1)
    mag = np.sqrt(dx_c**2 + dy_c**2) + 1e-12
    nx_c = -dy_c / mag
    ny_c = dx_c / mag
    inner_x = cx - 4.5 * nx_c
    inner_y = cy - 4.5 * ny_c
    outer_x = cx + 4.5 * nx_c
    outer_y = cy + 4.5 * ny_c

    # ── Top left: racing line ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor("#1c1c1c")
    fx = np.concatenate([outer_x, inner_x[::-1], [outer_x[0]]])
    fy = np.concatenate([outer_y, inner_y[::-1], [outer_y[0]]])
    ax.fill(fx, fy, color="#2e2e2e", zorder=1)
    for bx, by in [(inner_x, inner_y), (outer_x, outer_y)]:
        ax.plot(
            np.append(bx, bx[0]), np.append(by, by[0]), color="white", lw=1.0, zorder=2
        )
    ax.plot(
        np.append(cx, cx[0]),
        np.append(cy, cy[0]),
        color="#555",
        lw=1.0,
        ls="--",
        zorder=3,
        label="Centerline",
    )

    pts = np.array([rx, ry]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segs,
        cmap="plasma",
        norm=plt.Normalize(v_opt.min(), v_opt.max()),
        lw=1.0,
        zorder=4,
    )
    lc.set_array(v_opt[:-1])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Speed [m/s]", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_aspect("equal")
    ax.set_title("Racing Line (colored by speed)", color="white", fontsize=11)
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # ── Top right: speed profile ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor("#1c1c1c")
    ax.plot(s[:N], v_opt * 3.6, color="#e10600", lw=1.5)
    ax.fill_between(s[:N], 0, v_opt * 3.6, color="#e10600", alpha=0.2)
    ax.set_xlabel("Arc length s [m]", color="white", fontsize=10)
    ax.set_ylabel("Speed [km/h]", color="white", fontsize=10)
    ax.set_title("Speed Profile", color="white", fontsize=11)
    ax.set_xlim(0, L)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # ── Bottom left: lateral offset ───────────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor("#1c1c1c")
    ax.axhline(0, color="#555", lw=0.8, ls="--", label="Centerline")
    ax.axhline(n_min[0] + w_veh / 2, color="#888", lw=0.8, ls=":", label="Track limits")
    ax.axhline(n_max[0] - w_veh / 2, color="#888", lw=0.8, ls=":")
    ax.plot(s[:N], n_opt, color="#00aaff", lw=1.5)
    ax.fill_between(s[:N], 0, n_opt, color="#00aaff", alpha=0.2)
    ax.set_xlabel("Arc length s [m]", color="white", fontsize=10)
    ax.set_ylabel("Lateral offset n [m]", color="white", fontsize=10)
    ax.set_title("Lateral Offset from Centerline", color="white", fontsize=11)
    ax.set_xlim(0, L)
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # ── Bottom right: throttle/brake ──────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor("#1c1c1c")
    ax.axhline(0, color="#555", lw=0.8, ls="--")
    ax.fill_between(
        s[:N],
        0,
        Fx_opt,
        where=(Fx_opt >= 0),
        color="#00ff88",
        alpha=0.5,
        label="Throttle",
    )
    ax.fill_between(
        s[:N],
        0,
        Fx_opt,
        where=(Fx_opt < 0),
        color="#e10600",
        alpha=0.5,
        label="Braking",
    )
    ax.plot(s[:N], Fx_opt, color="white", lw=0.8, alpha=0.7)
    ax.set_xlabel("Arc length s [m]", color="white", fontsize=10)
    ax.set_ylabel("Longitudinal force Fx [N]", color="white", fontsize=10)
    ax.set_title("Throttle / Brake Profile", color="white", fontsize=11)
    ax.set_xlim(0, L)
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "optimal_racing_line.svg")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"Saved → {out}")
    plt.close()


plot_results(cx, cy, rx, ry, v_opt, n_opt, Fx_opt, s, L, "Bahrain GP")
