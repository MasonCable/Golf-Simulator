# --- physics + animation with rollout (single window, two views) ---

import json, math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from handle_roll import simulate_roll, safe_float

# -------- Physics constants --------
g = 9.80665
rho = 1.225
mass = 0.04593
radius = 0.02135
area = math.pi * radius**2
Cd = 0.27
CL_A = 0.00053
CL_B = 0.002
CL_MAX = 0.35

DT_AIR  = 0.002
MAX_T_AIR = 12.0
DT_ROLL = 0.01
MAX_T_ROLL = 8.0

SURFACES = {
    #   mu_r  ,  k_quad  (roll decel a = g*mu_r + k*v^2)
    "rough":  (0.055, 0.0005),
    "fairway":(0.035, 0.00035),
    "firm":   (0.025, 0.00025),
    "green":  (0.020, 0.00020),
}

def mph_to_mps(v_mph): return v_mph * 0.44704
def deg2rad(d): return d * math.pi / 180.0
def meters_to_yards(m): return m * 1.0936133
def meters_to_feet(m):  return m * 3.28084

def lift_coefficient(spin_rpm, speed_mps):
    base = CL_A * spin_rpm + CL_B
    if speed_mps < 10.0:
        base *= (speed_mps / 10.0)
    return max(0.0, min(base, CL_MAX))

def bounce_response(vx, vy, vz, spin_rpm, surface="fairway"):
    """
    Compute post-bounce velocities from pre-bounce (vx, vy, vz<0).
    Simple rigid impact:
      vzn+ = -e_n * vzn-
      vtan+ = vtan- * (1 - k_t)   with k_t from μ_t and spin
    Returns (vx2, vy2, vz2)
    """
    # Surface presets (tuned for "looks right")
    SURF_BOUNCE = {
        "rough":   dict(e_n=0.25, mu_t=0.45),
        "fairway": dict(e_n=0.35, mu_t=0.35),
        "firm":    dict(e_n=0.45, mu_t=0.28),
        "green":   dict(e_n=0.30, mu_t=0.25),
    }
    cfg = SURF_BOUNCE.get(surface, SURF_BOUNCE["fairway"])
    e_n  = cfg["e_n"]
    mu_t = cfg["mu_t"]

    # Spin increases tangential losses a bit (more grip → less skid)
    spin_scale = 1.0 / (1.0 + (spin_rpm / 3000.0))**0.5     # ~0.6–1.0
    k_t = mu_t * spin_scale                                 # 0..~0.45

    # Normal component flips and loses speed
    vz2 = max(0.0, -e_n * vz)   # upward after bounce

    # Tangential (horizontal) loses a fraction
    vx2 = vx * (1.0 - k_t)
    vy2 = vy * (1.0 - k_t)
    return vx2, vy2, vz2


def initial_velocity(speed_mph, vla_deg, hla_deg):
    v = mph_to_mps(speed_mph)
    vla = deg2rad(vla_deg)
    hla = deg2rad(hla_deg)
    vx = v * math.cos(vla) * math.cos(hla)   # +x downrange
    vy = v * math.cos(vla) * math.sin(hla)   # +y right
    vz = v * math.sin(vla)                   # +z up
    return np.array([vx, vy, vz], dtype=float)

def spin_axis_unit(spin_axis_deg):
    th = deg2rad(spin_axis_deg)
    s = np.array([0.0, math.cos(th), math.sin(th)], dtype=float)
    n = np.linalg.norm(s)
    return s / n if n > 0 else np.array([0.0, 1.0, 0.0], dtype=float)

def simulate_air(bd):
    """Return xs, ys, zs, ts until first ground contact (z==0)."""
    speed = float(bd.get("Speed", 150.0))
    vla   = float(bd.get("VLA", 12.0))
    hla   = float(bd.get("HLA", 0.0))
    spin  = float(bd.get("TotalSpin", 2500.0))
    saxis = float(bd.get("SpinAxis", 0.0))

    pos = np.array([0.0, 0.0, 0.0], dtype=float)
    vel = initial_velocity(speed, vla, hla)
    s_hat = spin_axis_unit(saxis)

    xs, ys, zs, ts = [0.0], [0.0], [0.0], [0.0]
    t = 0.0

    while t < MAX_T_AIR:
        vmag = np.linalg.norm(vel) + 1e-9
        vhat = vel / vmag
        q = 0.5 * rho * vmag**2

        Fd = -Cd * q * area * vhat
        Cl = lift_coefficient(spin, vmag)
        lift_dir = np.cross(vhat, s_hat)
        ln = np.linalg.norm(lift_dir)
        if ln > 0: lift_dir /= ln
        Fl = Cl * q * area * lift_dir
        Fg = np.array([0.0, 0.0, -mass * g])

        acc = (Fd + Fl + Fg) / mass
        vel = vel + acc * DT_AIR
        pos = pos + vel * DT_AIR
        t += DT_AIR

        xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2]); ts.append(t)
        # first ground contact
        if pos[2] <= 0.0 and t > 0.05:
            break

    # Interpolate last to z=0
    if zs[-1] < 0 and len(zs) >= 2:
        z1, z2 = zs[-2], zs[-1]
        x1, x2 = xs[-2], xs[-1]
        y1, y2 = ys[-2], ys[-1]
        if z2 != z1:
            a = (0 - z1) / (z2 - z1)
            xs[-1] = x1 + a*(x2 - x1)
            ys[-1] = y1 + a*(y2 - y1)
            zs[-1] = 0.0

    return np.array(xs), np.array(ys), np.array(zs), np.array(ts), vel


def simulate_roll_with_bounce(x0, y0, v_land_xy, v_land_z, spin_rpm, surface="fairway"):
    """
    One-bounce -> hop -> roll. All inputs coerced to float.
    """
    # Coerce scalars
    x0 = safe_float(x0); y0 = safe_float(y0)
    vz = safe_float(v_land_z, 0.0)
    spin_rpm = safe_float(spin_rpm, 0.0)

    # Coerce vector
    vx, vy = (safe_float(v_land_xy[0]), safe_float(v_land_xy[1])) if hasattr(v_land_xy, "__len__") else (0.0, 0.0)

    # 1) Bounce response
    vx2, vy2, vz2 = bounce_response(vx, vy, vz, spin_rpm, surface)

    # If bounce is negligible, go straight to roll
    if vz2 < 0.2:
        return *simulate_roll(x0, y0, np.array([vx2, vy2]), spin_rpm, surface),  # xr, yr, tr
               # stitch z/t to match your animation expectations:
               # convert to zr (zeros) and a time base starting at 0 if needed

    # 2) Short hop (air)
    xs = [x0]; ys = [y0]; zs = [0.0]; ts = [0.0]
    pos = np.array([x0, y0, 0.0], dtype=float)
    vel = np.array([vx2, vy2, vz2], dtype=float)
    t = 0.0
    while t < 2.0:
        vmag = float(np.linalg.norm(vel) + 1e-9)
        vhat = vel / vmag
        q = 0.5 * rho * vmag**2
        Fd = -Cd * q * area * vhat
        Fg = np.array([0.0, 0.0, -mass * g])
        acc = (Fd + Fg) / mass
        vel = vel + acc * DT_AIR
        pos = pos + vel * DT_AIR
        t += DT_AIR
        xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2]); ts.append(t)
        if pos[2] <= 0.0 and t > 0.03:
            # interpolate to ground
            z1, z2 = zs[-2], zs[-1]; x1, x2 = xs[-2], xs[-1]; y1, y2 = ys[-2], ys[-1]
            if z2 != z1:
                a = (0 - z1) / (z2 - z1)
                xs[-1] = x1 + a*(x2 - x1)
                ys[-1] = y1 + a*(y2 - y1)
                zs[-1] = 0.0
            break

    # 3) Roll from hop landing (note spin_rpm is float now)
    v_hop_xy = np.array([vel[0], vel[1]])
    xr, yr, tr = simulate_roll(xs[-1], ys[-1], v_hop_xy, spin_rpm, surface)

    # Stitch arrays (zr is 0 during roll)
    zr = np.concatenate([np.array(zs), np.zeros(max(0, len(xr)-1))])
    tr_full = np.concatenate([np.array(ts), (ts[-1] + np.array(tr[1:]))])
    xr_full = np.concatenate([np.array(xs), np.array(xr[1:])])
    yr_full = np.concatenate([np.array(ys), np.array(yr[1:])])
    return xr_full, yr_full, zr, tr_full


def simulate(bd, surface="fairway", mu_r=None, k_quad=None):
    """
    Full sim: air (carry) + rollout.
    Returns dict with:
      air_x,y,z,t
      roll_x,y,t   (z=0 during roll)
    """
    ax, ay, az, at, vel_land_3d = simulate_air(bd)
    # horizontal landing speed for roll
    v_land_xy = np.array([vel_land_3d[0], vel_land_3d[1]])
      # Extract spin from shot data
    spin_rpm = float(bd.get("TotalSpin", 2500.0))

    # simulate_air should give you vel_land_3d (the velocity at ground intersect)
    v_land_z  = float(vel_land_3d[2])          # typically negative
    spin_rpm  = float(bd.get("TotalSpin", 2500.0))

    xr, yr, zr, tr = simulate_roll_with_bounce(ax[-1], ay[-1], v_land_xy, v_land_z, spin_rpm, surface)
    return {
        "air_x": ax, "air_y": ay, "air_z": az, "air_t": at,
        "roll_x": xr, "roll_y": yr, "roll_t": tr
    }

# ------------- Animation (single window, two views) -------------
def animate_combined(sim, playback=1.0, carry_color="tab:blue", roll_color="tab:orange"):
    import numpy as np

    # Display units
    Xc = meters_to_yards(sim["air_x"]);   Yc = meters_to_yards(sim["air_y"]);   Zc = meters_to_feet(sim["air_z"])
    Tc = sim["air_t"]
    Xr = meters_to_yards(sim["roll_x"]);  Yr = meters_to_yards(sim["roll_y"]);  Tr = sim["roll_t"]

    # One window, two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.suptitle("Ball Flight (Carry + Roll)")

    # Side View
    ax1.set_title("Side View")
    ax1.set_xlabel("Downrange (yards)")
    ax1.set_ylabel("Height (feet)")
    ax1.grid(True)
    ax1.plot(Xc, Zc, linewidth=1.25, alpha=0.85, color=carry_color, label="Carry")
    # Roll on ground (height = 0 line from landing onward)
    if len(Xr) > 1:
        ax1.plot([Xc[-1], *Xr[1:]], [0.0, *([0.0]*(len(Xr)-1))],
                 linewidth=2.0, alpha=0.9, color=roll_color, label="Roll")
    ball1, = ax1.plot([], [], marker="o", markersize=7, color=carry_color)
    trail1, = ax1.plot([], [], linewidth=2, color=carry_color)
    x_max = max(np.max(Xc), np.max(Xr) if len(Xr) else 0) * 1.05
    z_max = max(np.max(Zc), 1.0) * 1.10
    ax1.set_xlim(0, max(1, float(x_max)))
    ax1.set_ylim(0, max(1, float(z_max)))
    ax1.legend(loc="upper right")
    fig.tight_layout()

    # Top View
    ax2.set_title("Top View (Left/Right)")
    ax2.set_xlabel("Downrange (yards)")
    ax2.set_ylabel("Lateral (yards)")
    ax2.grid(True)
    ax2.plot(Xc, Yc, linewidth=1.25, alpha=0.85, color=carry_color, label="Carry")
    if len(Xr) > 1:
        ax2.plot(Xr, Yr, linewidth=2.0, alpha=0.9, color=roll_color, label="Roll")
    ball2, = ax2.plot([], [], marker="o", markersize=7, color=carry_color)
    trail2, = ax2.plot([], [], linewidth=2, color=carry_color)
    x_max2 = max(np.max(Xc), np.max(Xr) if len(Xr) else 0) * 1.05
    y_abs = float(max(1.0, np.max(np.abs(np.concatenate([Yc, Yr])) if len(Yr) else np.max(np.abs(Yc))) * 1.10))
    ax2.set_xlim(0, max(1, float(x_max2)))
    ax2.set_ylim(-y_abs, y_abs)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="upper right")

    # Real-time playback
    start = time.perf_counter()
    dt_air  = Tc[1] - Tc[0] if len(Tc) > 1 else DT_AIR
    dt_roll = Tr[1] - Tr[0] if len(Tr) > 1 else DT_ROLL
    trail_seconds = 0.5
    trail_len_air  = max(5, int(trail_seconds / max(1e-6, dt_air)))
    trail_len_roll = max(5, int(trail_seconds / max(1e-6, dt_roll)))
    frame_dt = 1/60.0

    phase = "air"
    i_air, i_roll = 0, 0
    while True:
        elapsed = (time.perf_counter() - start) * playback

        if phase == "air":
            i_air = min(len(Tc) - 1, int(round(elapsed / max(1e-6, dt_air))))
            # update side
            ball1.set_data([Xc[i_air]], [Zc[i_air]])
            s = max(0, i_air - trail_len_air)
            trail1.set_data(Xc[s:i_air+1], Zc[s:i_air+1])
            # update top
            ball2.set_data([Xc[i_air]], [Yc[i_air]])
            trail2.set_data(Xc[s:i_air+1], Yc[s:i_air+1])

            if i_air >= len(Tc) - 1:
                phase = "roll"
                # switch marker color to roll color
                ball1.set_color(roll_color)
                trail1.set_color(roll_color)
                ball2.set_color(roll_color)
                trail2.set_color(roll_color)
                start = time.perf_counter()  # restart clock for roll
        else:
            # roll time
            i_roll = min(len(Tr) - 1, int(round(((time.perf_counter() - start) * playback) / max(1e-6, dt_roll))))
            # side (z=0)
            ball1.set_data([Xr[i_roll]], [0.0])
            s = max(0, i_roll - trail_len_roll)
            trail1.set_data(Xr[s:i_roll+1], [0.0]*(i_roll - s + 1))
            # top
            ball2.set_data([Xr[i_roll]], [Yr[i_roll]])
            trail2.set_data(Xr[s:i_roll+1], Yr[s:i_roll+1])
            if i_roll >= len(Tr) - 1:
                break

        plt.pause(0.001)
        time.sleep(frame_dt)

    plt.show()

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to JSON with BallData (GSPro-like).")
    ap.add_argument("--surface", default="fairway", choices=list(SURFACES.keys()),
                    help="Roll surface (affects rollout distance).")
    ap.add_argument("--playback", type=float, default=1.0, help="Playback speed (1.0 real-time).")
    # Optional manual roll coefficients
    ap.add_argument("--mu_r", type=float, default=None, help="Rolling resistance coefficient override.")
    ap.add_argument("--k_quad", type=float, default=None, help="Quadratic roll drag coefficient override.")
    args = ap.parse_args()

    data = json.load(open(args.json, "r", encoding="utf-8"))
    bd = data[1].get("BallData")
    sim = simulate(bd, surface=args.surface, mu_r=args.mu_r, k_quad=args.k_quad)

    carry_yd = meters_to_yards(sim["air_x"][-1])
    total_yd = meters_to_yards((sim["roll_x"][-1] if len(sim["roll_x"]) else sim["air_x"][-1]))
    apex_ft  = meters_to_feet(np.max(sim["air_z"]))
    flight_s = sim["air_t"][-1] + (sim["roll_t"][-1] if len(sim["roll_t"]) else 0.0)
    roll = round(total_yd - carry_yd, 2)

    print(f"Carry: {carry_yd:.1f} yd   Total: {total_yd:.1f} yd   Apex: {apex_ft:.1f} ft   Time: {flight_s:.2f} s Roll: {roll}")
    animate_combined(sim, playback=args.playback)
