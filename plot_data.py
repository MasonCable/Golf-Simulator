import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helpers import handle_arguments

# ------------------------------
# Units & helpers
# ------------------------------
deg2rad = lambda d: d * math.pi / 180.0
m2ft    = lambda m: m * 3.28084  # not used but kept for parity

def interp_y(x, X, Y):
    """Linear interpolation of Y at downrange x, clamped to X domain."""
    if x <= X[0]: return float(Y[0])
    if x >= X[-1]: return float(Y[-1])
    i = np.searchsorted(X, x) - 1
    x1, x2 = X[i], X[i+1]
    y1, y2 = Y[i], Y[i+1]
    a = (x - x1) / max(1e-9, (x2 - x1))
    return float(y1 + a * (y2 - y1))

def interp_y_array(x_arr, X, Y):
    x_arr = np.asarray(x_arr, dtype=float)
    return np.array([interp_y(x, X, Y) for x in x_arr], dtype=float)

# ------------------------------
# Lateral curve from HLA & Offline
# ------------------------------
def build_lateral_curve(total_yd, hla_deg, offline_yd, x_samples):
    """
    Smooth lateral curve y(x) with:
      y(0)=0, y'(0)=tan(HLA), y(total)=Offline.
    Quadratic y = a x^2 + b x; b = tan(HLA); a chosen to land at Offline.
    """
    b = math.tan(deg2rad(hla_deg))
    if total_yd > 0:
        a = (offline_yd - b * total_yd) / (total_yd**2)
    else:
        a = 0.0
    y = a * x_samples**2 + b * x_samples
    if len(y):
        y[-1] = offline_yd  # hard-snap the final point to Offline
    return y

# ------------------------------
# Paths from JSON (no distance physics)
# ------------------------------
def build_paths_from_json(bd, bounce_policy=None):
    """
    Build plotting arrays using JSON values directly.
    Only compute:
      - top-view lateral curve from HLA & Offline,
      - a simple, subtle bounce (if roll is meaningful).
    Returns dict with arrays (x in yards; z in feet; t in seconds):
      carry_x/z/t, bounce_x/z/t, roll_x/z/t, top_x/y
    """
    carry_yd   = float(bd.get("Carry", 0.0))
    total_yd   = float(bd.get("TotalDistance", carry_yd))
    peak_ft    = float(bd.get("PeakHeight", 0.0))
    hla_deg    = float(bd.get("HLA", 0.0))
    offline_yd = float(bd.get("Offline", 0.0))

    carry_yd = max(0.0, carry_yd)
    total_yd = max(carry_yd, total_yd)
    roll_yd  = max(0.0, total_yd - carry_yd)

    # --- Side view: Carry arc exactly from 0..Carry with given PeakHeight ---
    # Use a symmetric parabola hitting PeakHeight at mid-carry (shape only).
    n_air = max(200, int(carry_yd * 2))
    x_air = np.linspace(0.0, carry_yd, n_air)
    if carry_yd > 0.0 and peak_ft > 0.0:
        a = (4.0 * peak_ft) / (carry_yd**2)    # z = a * x * (carry - x)
        z_air_ft = a * x_air * (carry_yd - x_air)
    else:
        z_air_ft = np.zeros_like(x_air)

    # --- Decide bounce from roll amount (no physics) ---
    # Default policy: bounce if roll >= 1.0 yd; small hop: len ~ 15% of roll (0.8..3 yd), apex ~ 0.6 ft
    if bounce_policy is None:
        bounce_policy = {
            "min_roll_for_bounce_yd": 1.0,
            "len_frac_of_roll": 0.15,
            "len_min_yd": 0.8,
            "len_max_yd": 3.0,
            "apex_ft": 0.6
        }

    do_bounce = roll_yd >= bounce_policy["min_roll_for_bounce_yd"]
    if do_bounce:
        bounce_len_yd = float(np.clip(bounce_policy["len_frac_of_roll"] * roll_yd,
                                      bounce_policy["len_min_yd"], bounce_policy["len_max_yd"]))
        n_bnc = max(6, int(bounce_len_yd * 8))
        x_bnc = np.linspace(carry_yd, carry_yd + bounce_len_yd, n_bnc)
        mid = 0.5 * (x_bnc[0] + x_bnc[-1])
        half_len = max(1e-9, 0.5 * (x_bnc[-1] - x_bnc[0]))
        z_bnc_ft = bounce_policy["apex_ft"] * (1.0 - ((x_bnc - mid) / half_len)**2)
        z_bnc_ft[z_bnc_ft < 0] = 0.0
    else:
        x_bnc = np.array([])
        z_bnc_ft = np.array([])

    # --- Roll segment: flat from end of bounce (or carry) to TotalDistance ---
    x_roll_start = carry_yd + (x_bnc[-1] - carry_yd if do_bounce else 0.0)
    n_roll = max(2, int(max(2.0, total_yd - x_roll_start) * 2))
    x_roll = np.linspace(x_roll_start, total_yd, n_roll)
    z_roll_ft = np.zeros_like(x_roll)

    # --- Top view: full lateral curve across 0..TotalDistance ---
    x_full = np.concatenate([x_air, x_bnc[1:], x_roll[1:]]) if len(x_roll) > 1 else \
             (np.concatenate([x_air, x_bnc[1:]]) if len(x_bnc) > 1 else x_air)
    y_full = build_lateral_curve(total_yd, hla_deg, offline_yd, x_full)

    # --- Simple timeline (for animation pacing only) ---
    air_time    = 0.012 * carry_yd + 0.9                      # ~0.9–3.0 s
    bounce_time = 0.25 if do_bounce else 0.0                  # short hop visual
    roll_time   = min(3.0, 0.08 * roll_yd)                    # ~0.08 s per yd, capped

    t_air  = np.linspace(0.0, max(0.02, air_time), len(x_air))
    t_bnc  = np.linspace(0.0, bounce_time, len(x_bnc)) if do_bounce else np.array([])
    t_roll = np.linspace(0.0, max(0.02, roll_time), len(x_roll)) if len(x_roll) else np.array([])

    return {
        "carry_x": x_air, "carry_z": z_air_ft, "carry_t": t_air,
        "bounce_x": x_bnc, "bounce_z": z_bnc_ft, "bounce_t": t_bnc,
        "roll_x": x_roll, "roll_z": z_roll_ft, "roll_t": t_roll,
        "top_x": x_full, "top_y": y_full,
        "meta": {
            "carry_yd": carry_yd,
            "total_yd": total_yd,
            "roll_yd": roll_yd,
            "did_bounce": do_bounce,
            "bounce_len_yd": (x_bnc[-1] - x_bnc[0]) if do_bounce else 0.0,
            "bounce_apex_ft": bounce_policy["apex_ft"] if do_bounce else 0.0,
            "peak_ft": peak_ft,
            "offline_yd": offline_yd,
            "hla_deg": hla_deg
        }
    }

# ------------------------------
# Animation (side + top)
# ------------------------------
def animate_paths(paths, playback=1.0):
    Cx = paths["carry_x"];  Cz = paths["carry_z"];  Ct = paths["carry_t"]
    Bx = paths["bounce_x"]; Bz = paths["bounce_z"]; Bt = paths["bounce_t"]
    Rx = paths["roll_x"];   Rz = paths["roll_z"];   Rt = paths["roll_t"]
    X  = paths["top_x"];    Y  = paths["top_y"]

    total_t = Ct[-1] + (Bt[-1] if len(Bt) else 0.0) + (Rt[-1] if len(Rt) else 0.0)
    fps = 60
    frame_dt = 1.0 / fps
    nframes = max(1, int(total_t / frame_dt * max(1e-9, playback)) + 1)

    xmax = max(np.max(Cx),
               np.max(Bx) if len(Bx) else 0.0,
               np.max(Rx) if len(Rx) else 0.0, 5.0) * 1.06
    zmax = max(np.max(Cz), np.max(Bz) if len(Bz) else 0.0, 1.0) * 1.12
    ymax = max(1.0, np.max(np.abs(Y)) * 1.12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Golf Ball Flight – JSON Driven (Carry • Bounce • Roll)", fontsize=13)

    # Side view
    ax1.set_title("Side View"); ax1.set_xlabel("Downrange (yd)"); ax1.set_ylabel("Height (ft)")
    ax1.grid(True, alpha=0.3)
    ax1.plot(Cx, Cz, color="#1f77b4", lw=1.8, label="Carry")
    if len(Bx): ax1.plot(Bx, Bz, color="#2ca02c", lw=2.0, label="Bounce")
    if len(Rx): ax1.plot(Rx, Rz, color="#ff7f0e", lw=2.2, label="Roll")
    ball1, = ax1.plot([], [], 'o', ms=8, color="#1f77b4")
    trail1, = ax1.plot([], [], lw=2, color="#1f77b4", alpha=0.6)
    ax1.set_xlim(0, xmax); ax1.set_ylim(0, zmax); ax1.legend()

    # Top view
    ax2.set_title("Top View"); ax2.set_xlabel("Downrange (yd)"); ax2.set_ylabel("Lateral (yd)")
    ax2.grid(True, alpha=0.3)
    ax2.plot(X, Y, color="#1f77b4", lw=1.2, alpha=0.55, label="Path (lat)")
    ax2.plot(Cx, interp_y_array(Cx, X, Y), color="#1f77b4", lw=1.8, label="Carry (lat)")
    if len(Bx): ax2.plot(Bx, interp_y_array(Bx, X, Y), color="#2ca02c", lw=2.0, label="Bounce (lat)")
    if len(Rx): ax2.plot(Rx, interp_y_array(Rx, X, Y), color="#ff7f0e", lw=2.2, label="Roll (lat)")
    ball2, = ax2.plot([], [], 'o', ms=8, color="#1f77b4")
    trail2, = ax2.plot([], [], lw=2, color="#1f77b4", alpha=0.6)
    ax2.set_xlim(0, xmax); ax2.set_ylim(-ymax, ymax); ax2.set_aspect('equal'); ax2.legend()

    # Time anchors
    T_air = Ct
    T_bnc = (Ct[-1] + Bt) if len(Bt) else np.array([])
    T_rol = (Ct[-1] + (Bt[-1] if len(Bt) else 0.0) + Rt) if len(Rt) else np.array([])

    trail_air  = 0.45
    trail_roll = 0.45

    def update(frame):
        t = frame * frame_dt / max(1e-9, playback)

        # Carry
        if t <= T_air[-1]:
            i = min(len(T_air) - 1, int((t / max(1e-9, T_air[-1])) * (len(T_air) - 1)))
            ball1.set_color("#1f77b4"); trail1.set_color("#1f77b4")
            ball2.set_color("#1f77b4"); trail2.set_color("#1f77b4")
            ball1.set_data([Cx[i]], [Cz[i]])
            s1 = max(0, i - int(trail_air / max(1e-9, T_air[-1]) * len(T_air)))
            trail1.set_data(Cx[s1:i+1], Cz[s1:i+1])

            y_i = interp_y(Cx[i], X, Y)
            ball2.set_data([Cx[i]], [y_i])
            trail2.set_data(Cx[s1:i+1], interp_y_array(Cx[s1:i+1], X, Y))
            return ball1, trail1, ball2, trail2

        # Bounce
        if len(T_bnc) and t <= T_bnc[-1]:
            tb = t - T_air[-1]
            i = min(len(Bt) - 1, int((tb / max(1e-9, Bt[-1])) * (len(Bt) - 1)))
            ball1.set_color("#2ca02c"); trail1.set_color("#2ca02c")
            ball2.set_color("#2ca02c"); trail2.set_color("#2ca02c")
            ball1.set_data([Bx[i]], [Bz[i]])
            s1 = max(0, i - int(trail_air / max(1e-9, Bt[-1]) * len(Bt)))
            trail1.set_data(Bx[s1:i+1], Bz[s1:i+1])

            y_i = interp_y(Bx[i], X, Y)
            ball2.set_data([Bx[i]], [y_i])
            trail2.set_data(Bx[s1:i+1], interp_y_array(Bx[s1:i+1], X, Y))
            return ball1, trail1, ball2, trail2

        # Roll
        if len(T_rol):
            tr = t - (T_air[-1] + (Bt[-1] if len(Bt) else 0.0))
            if Rt[-1] <= 0:
                return ball1, trail1, ball2, trail2
            i = min(len(Rt) - 1, int((tr / max(1e-9, Rt[-1])) * (len(Rt) - 1)))
            ball1.set_color("#ff7f0e"); trail1.set_color("#ff7f0e")
            ball2.set_color("#ff7f0e"); trail2.set_color("#ff7f0e")
            ball1.set_data([Rx[i]], [Rz[i]])
            s2 = max(0, i - int(trail_roll / max(1e-9, Rt[-1]) * len(Rt)))
            trail1.set_data(Rx[s2:i+1], Rz[s2:i+1])

            y_i = interp_y(Rx[i], X, Y)
            trail2.set_data(Rx[s2:i+1], interp_y_array(Rx[s2:i+1], X, Y))
            ball2.set_data([Rx[i]], [y_i])
            return ball1, trail1, ball2, trail2

        return ball1, trail1, ball2, trail2

    FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    args = handle_arguments()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # pick shot by Club if list; otherwise treat as single shot dict
    shots = [s for s in data if s.get("Club") == args.shot] if isinstance(data, list) else [data]
    bd = shots[0] if shots else (data[0] if isinstance(data, list) else data)

    paths = build_paths_from_json(bd)

    carry_yd = paths["meta"]["carry_yd"]
    total_yd = paths["meta"]["total_yd"]
    roll_yd  = paths["meta"]["roll_yd"]
    apex_ft  = paths["meta"]["peak_ft"]
    did_bounce = paths["meta"]["did_bounce"]
    bounce_len = paths["meta"]["bounce_len_yd"]
    bounce_apx = paths["meta"]["bounce_apex_ft"]

    print(f"Carry: {carry_yd:.2f} yd | Total: {total_yd:.2f} yd | "
          f"Roll: {roll_yd:.2f} yd | Apex: {apex_ft:.2f} ft | "
          f"Bounce: {'yes' if did_bounce else 'no'} "
          f"(len {bounce_len:.2f} yd, apex {bounce_apx:.2f} ft)")

    animate_paths(paths, playback=args.playback)
