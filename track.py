import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
import fastf1, os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("/tmp/ff1", exist_ok=True)
fastf1.Cache.enable_cache("/tmp/ff1")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_real_track(year, gp, session_type):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()
    x = tel["X"].values.astype(float)
    y = tel["Y"].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


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


def compute_curvature(t_uni, cs_x, cs_y):
    x1, y1 = cs_x(t_uni, 1), cs_y(t_uni, 1)
    x2, y2 = cs_x(t_uni, 2), cs_y(t_uni, 2)
    return (x1 * y2 - y1 * x2) / (x1**2 + y1**2) ** 1.5


def smooth_curvature(kappa, window=12):
    w = window
    ext = np.concatenate([kappa[-w:], kappa, kappa[:w]])
    sm = uniform_filter1d(ext, size=w)
    return sm[w:-w]


def track_boundaries(cx, cy, half_width=4.5):
    dx = np.roll(cx, -1) - np.roll(cx, 1)
    dy = np.roll(cy, -1) - np.roll(cy, 1)
    mag = np.sqrt(dx**2 + dy**2) + 1e-12
    nx, ny = -dy / mag, dx / mag

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


def plot_track(cx, cy, ix, iy, ox, oy, kappa, s_uni, L, title):
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6), facecolor="#0d0d0d")
    fig.suptitle(
        title + " — Track Geometry Extraction",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

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

    fig.suptitle(
        title + " — Track Geometry Extraction",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 0.93, 0.95], pad=0.5)

    out = os.path.join(OUTPUT_DIR, "track.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  Saved → {out}")
    plt.close()


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


if __name__ == "__main__":
    N = 500
    track = "MIAMI"

    x_raw, y_raw = load_real_track(year=2025, gp=track, session_type="R")

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

    cx, cy, t_uni, cs_x, cs_y, L, s_uni = close_and_resample(x_raw, y_raw, N)

    kappa = smooth_curvature(compute_curvature(t_uni, cs_x, cs_y))

    ix, iy, ox, oy, n_min, n_max = track_boundaries(cx, cy, half_width=4.5)

    plot_track(cx, cy, ix, iy, ox, oy, kappa, s_uni, L, track)

    save_track_data(s_uni, cx, cy, kappa, n_min, n_max, L)
