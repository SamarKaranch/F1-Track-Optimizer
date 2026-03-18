"""
F1 Track Geometry Extractor
============================
TWO MODES:
  1. LIVE (run locally): uncomment the fastf1 block to pull real GPS data
  2. DEMO (runs anywhere): uses a Bahrain-like synthetic track

Outputs:
  track_geometry.png  — track map + curvature profile
  track_data.npz      — arrays ready for the optimizer:
                        s, cx, cy, kappa_ref, n_min, n_max, L
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# MODE 1 — REAL DATA  (uncomment when running locally: pip install fastf1)
# =========================================================================
import fastf1, os

os.makedirs("/tmp/ff1", exist_ok=True)
fastf1.Cache.enable_cache("/tmp/ff1")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


#
def load_real_track(year=2023, gp="Bahrain", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()
    x = tel["X"].values.astype(float)
    y = tel["Y"].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


x_raw, y_raw = load_real_track()

# =========================================================================
# STEP 2 — Close loop & resample to N uniform arc-length points
# =========================================================================


def close_and_resample(x, y, N=500):
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    dx, dy = np.diff(x), np.diff(y)
    s_ch = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])

    cs_x = CubicSpline(s_ch, x, bc_type="periodic")
    cs_y = CubicSpline(s_ch, y, bc_type="periodic")

    t_fine = np.linspace(0, s_ch[-1], 50_000)
    speed = np.sqrt(cs_x(t_fine, 1) ** 2 + cs_y(t_fine, 1) ** 2)
    s_fine = np.concatenate([[0], cumulative_trapezoid(speed, t_fine)])
    L = s_fine[-1]

    s_uni = np.linspace(0, L, N + 1)[:-1]
    t_uni = np.interp(s_uni, s_fine, t_fine)

    cx = cs_x(t_uni)
    cy = cs_y(t_uni)

    print(f"  Track length      : {L:.1f} m")
    print(f"  Arc-length step ds: {L/N:.2f} m")
    return cx, cy, t_uni, cs_x, cs_y, L, s_uni


# =========================================================================
# STEP 3 — Curvature from spline derivatives (exact)
# =========================================================================


def compute_curvature(t_uni, cs_x, cs_y):
    x1, y1 = cs_x(t_uni, 1), cs_y(t_uni, 1)
    x2, y2 = cs_x(t_uni, 2), cs_y(t_uni, 2)
    return (x1 * y2 - y1 * x2) / (x1**2 + y1**2) ** 1.5


def smooth_curvature(kappa, window=12):
    w = window
    ext = np.concatenate([kappa[-w:], kappa, kappa[:w]])
    sm = uniform_filter1d(ext, size=w)
    return sm[w:-w]


# =========================================================================
# STEP 4 — Track boundaries
# =========================================================================


def track_boundaries(cx, cy, half_width=4.5):
    dx = np.roll(cx, -1) - np.roll(cx, 1)
    dy = np.roll(cy, -1) - np.roll(cy, 1)
    mag = np.sqrt(dx**2 + dy**2) + 1e-12
    nx, ny = -dy / mag, dx / mag  # left-pointing normal

    outer_x, outer_y = cx + half_width * nx, cy + half_width * ny
    inner_x, inner_y = cx - half_width * nx, cy - half_width * ny
    N = len(cx)
    return (
        inner_x,
        inner_y,
        outer_x,
        outer_y,
        np.full(N, -half_width),
        np.full(N, half_width),
    )


# =========================================================================
# STEP 5 — Plot
# =========================================================================


def plot_track(cx, cy, ix, iy, ox, oy, kappa, s_uni, L, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), facecolor="#0d0d0d")
    fig.suptitle(
        title + " — Track Geometry Extraction",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    # ── Track map ────────────────────────────────────────────────────────────
    ax1.set_facecolor("#1c1c1c")

    # Tarmac fill
    fx = np.concatenate([ox, ix[::-1], [ox[0]]])
    fy = np.concatenate([oy, iy[::-1], [oy[0]]])
    ax1.fill(fx, fy, color="#2e2e2e", zorder=1)

    # Track limit boundaries
    for bx, by in [(ix, iy), (ox, oy)]:
        ax1.plot(
            np.append(bx, bx[0]), np.append(by, by[0]), color="white", lw=1.3, zorder=2
        )

    # Centerline colored by curvature
    pts = np.array([cx, cy]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segs, cmap="RdYlGn_r", norm=plt.Normalize(-0.025, 0.025), lw=2.5, zorder=3
    )
    lc.set_array(kappa[:-1])
    ax1.add_collection(lc)

    cb = fig.colorbar(lc, ax=ax1, fraction=0.03, pad=0.04)
    cb.set_label("Curvature κ [1/m]", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Dashed centerline on top of curvature coloring
    ax1.plot(
        np.append(cx, cx[0]),
        np.append(cy, cy[0]),
        color="#002fff",
        lw=1.5,
        ls="--",
        zorder=6,
        alpha=0.8,
        label="Centerline",
    )

    # Start/finish
    ax1.scatter([cx[0]], [cy[0]], color="#e10600", s=120, zorder=7, label="S/F")
    ax1.legend(facecolor="#2a2a2a", labelcolor="white", framealpha=0.9, fontsize=8)

    # Turn numbers
    turns = np.linspace(0, len(cx) - 1, 16, dtype=int)[1:]
    for k, idx in enumerate(turns, 1):
        ax1.text(
            cx[idx] * 1.05,
            cy[idx] * 1.05,
            str(k),
            color="#ffcc00",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )

    ax1.set_aspect("equal")
    ax1.set_title("Centerline & Track Boundaries", color="white", fontsize=11)
    ax1.tick_params(colors="#888")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    # ── Curvature profile ────────────────────────────────────────────────────
    ax2.set_facecolor("#1c1c1c")
    ax2.axhline(0, color="#555", lw=0.8, ls="--")

    ax2.fill_between(
        s_uni,
        kappa,
        where=(kappa >= 0),
        color="#e10600",
        alpha=0.55,
        label="Left turn (κ>0)",
    )
    ax2.fill_between(
        s_uni,
        kappa,
        where=(kappa < 0),
        color="#00aaff",
        alpha=0.55,
        label="Right turn (κ<0)",
    )
    ax2.plot(s_uni, kappa, color="white", lw=0.9, alpha=0.8)
    ax2.fill_between(
        s_uni,
        kappa.min() * 1.1,
        kappa.max() * 1.1,
        where=(np.abs(kappa) > 0.012),
        color="#ffcc00",
        alpha=0.07,
        label="|κ|>0.012 (tight)",
    )

    ax2.set_xlabel("Arc length  s  [m]", color="white", fontsize=10)
    ax2.set_ylabel("Curvature  κ  [1/m]", color="white", fontsize=10)
    ax2.set_title("κ_ref(s) — Centerline Curvature Profile", color="white", fontsize=11)
    ax2.set_xlim(0, L)
    ax2.legend(facecolor="#2a2a2a", labelcolor="white", framealpha=0.9, fontsize=8)
    ax2.tick_params(colors="#888")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")

    # Annotate hairpin peaks
    peaks = np.where(np.abs(kappa) > 0.015)[0]
    for idx in peaks[:: max(1, len(peaks) // 5)]:
        ax2.annotate(
            f"{kappa[idx]:.3f}",
            xy=(s_uni[idx], kappa[idx]),
            xytext=(0, 14),
            textcoords="offset points",
            color="#ffcc00",
            fontsize=7,
            arrowprops=dict(arrowstyle="-", color="#ffcc00", lw=0.8),
            ha="center",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "track_geometry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  Saved → {out}")
    plt.close()


# =========================================================================
# STEP 6 — Save optimizer-ready arrays
# =========================================================================


def save_track_data(s_uni, cx, cy, kappa_ref, n_min, n_max, L):
    out = os.path.join(OUTPUT_DIR, "track_data.npz")
    np.savez(
        out,
        s=s_uni,
        cx=cx,
        cy=cy,
        kappa_ref=kappa_ref,
        n_min=n_min,
        n_max=n_max,
        L=np.array([L]),
    )
    print(f"  Saved → {out}")
    print(f"  Keys: s{s_uni.shape}  cx{cx.shape}  kappa_ref{kappa_ref.shape}")
    print()
    print("  Load in optimizer with:")
    print("    data = np.load('track_data.npz')")
    print("    s          = data['s']")
    print("    kappa_ref  = data['kappa_ref']")
    print("    n_min      = data['n_min']")
    print("    n_max      = data['n_max']")
    print("    L          = float(data['L'])")


if __name__ == "__main__":
    N = 500

    print("── Step 1: Build track ───────────────────")
    x_raw, y_raw = load_real_track(year=2025, gp="MONACO", session_type="R")

    x_raw -= x_raw.mean()
    y_raw -= y_raw.mean()

    # Normalize scale: compute rough perimeter and scale to expected lap length
    dx = np.diff(np.append(x_raw, x_raw[0]))
    dy = np.diff(np.append(y_raw, y_raw[0]))
    rough_len = np.sum(np.sqrt(dx**2 + dy**2))
    expected_len = 5412  # metres — change per track
    scale = expected_len / rough_len
    x_raw *= scale
    y_raw *= scale
    print(f"Applied scale factor: {scale:.4f}")

    print("── Step 2: Fit spline & resample ───────────────────────────")
    cx, cy, t_uni, cs_x, cs_y, L, s_uni = close_and_resample(x_raw, y_raw, N)

    print("── Step 3: Compute curvature ───────────────────────────────")
    kappa = smooth_curvature(compute_curvature(t_uni, cs_x, cs_y))
    print(f"  Max |κ|    : {np.max(np.abs(kappa)):.5f} 1/m")
    print(f"  Min radius : {1/np.max(np.abs(kappa)):.1f} m")

    print("── Step 4: Build track boundaries ──────────────────────────")
    ix, iy, ox, oy, n_min, n_max = track_boundaries(cx, cy, half_width=4.5)

    print("── Step 5: Plot ────────────────────────────────────────────")
    plot_track(cx, cy, ix, iy, ox, oy, kappa, s_uni, L, "Bahrain GP")

    print("── Step 6: Save ────────────────────────────────────────────")
    save_track_data(s_uni, cx, cy, kappa, n_min, n_max, L)
