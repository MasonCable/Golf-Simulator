#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Golf Ball Flight – TrackMan-grade
* Realistic bounce (visible in the plot)
* Your original roll-out (no 200 yd roll)
* Fixed unpacking error
"""

import json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helpers import handle_arguments

# -------------------------------------------------------------------------
# 1. PHYSICS CONSTANTS
# -------------------------------------------------------------------------
g       = 9.80665
rho     = 1.225
mass    = 0.04593
radius  = 0.02135
area    = math.pi * radius**2
Cd      = 0.27
CL_A    = 0.00053
CL_B    = 0.002
CL_MAX  = 0.35

DT_AIR   = 0.002
MAX_T_AIR = 12.0
DT_ROLL  = 0.01
MAX_T_ROLL = 8.0

# (mu_r, k_quad) – rolling resistance + quadratic drag
SURFACES = {
    "rough":   (0.055, 0.0005),
    "fairway": (0.035, 0.00035),
    "firm":    (0.025, 0.00025),
    "green":   (0.020, 0.00020),
}

# -------------------------------------------------------------------------
# 2. HELPERS
# -------------------------------------------------------------------------
def mph_to_mps(v): return v * 0.44704
def deg2rad(d):    return d * math.pi / 180.0
def m2yd(m):       return m * 1.0936133
def m2ft(m):       return m * 3.28084

def safe_float(x, default=0.0):
    try: return float(x)
    except (TypeError, ValueError): return float(default)

# -------------------------------------------------------------------------
# 3. INITIAL VELOCITY
# -------------------------------------------------------------------------
def initial_velocity(speed_mph, vla_deg, hla_deg):
    v   = mph_to_mps(speed_mph)
    vla = deg2rad(vla_deg)
    hla = deg2rad(hla_deg)
    vx = v * math.cos(vla) * math.cos(hla)
    vy = v * math.cos(vla) * math.sin(hla)
    vz = v * math.sin(vla)
    return np.array([vx, vy, vz], dtype=float)

def spin_axis_unit(spin_axis_deg):
    th = deg2rad(spin_axis_deg)
    s  = np.array([0.0, math.cos(th), math.sin(th)])
    n  = np.linalg.norm(s)
    return s / n if n > 0 else np.array([0.0, 1.0, 0.0])

# -------------------------------------------------------------------------
# 4. AIR PHASE (returns 7 values – tilt added)
# -------------------------------------------------------------------------
def simulate_air(bd):
    speed    = safe_float(bd.get("Speed"))
    vla      = safe_float(bd.get("VLA"))
    hla      = safe_float(bd.get("HLA"))
    spin_rpm = safe_float(bd.get("TotalSpin"))
    tilt     = safe_float(bd.get("SpinAxis"))

    pos = np.zeros(3)
    vel = initial_velocity(speed, vla, hla)
    spin_rad = spin_rpm * 2*math.pi/60.0

    xs, ys, zs, ts = [0.0], [0.0], [0.0], [0.0]
    t = 0.0

    while t < MAX_T_AIR:
        vmag = np.linalg.norm(vel) + 1e-12
        vhat = vel / vmag
        q    = 0.5 * rho * vmag**2

        # Drag
        Fd = -Cd * q * area * vhat

        # Magnus lift
        Cl = min(CL_MAX, CL_A*abs(spin_rpm) + CL_B)
        omega_vec = spin_rad * spin_axis_unit(tilt)
        lift_dir  = np.cross(vhat, omega_vec)
        ln = np.linalg.norm(lift_dir)
        if ln > 0: lift_dir /= ln
        Fl = Cl * q * area * lift_dir

        # Gravity
        Fg = np.array([0.0, 0.0, -mass*g])
        acc = (Fd + Fl + Fg) / mass

        vel += acc * DT_AIR
        pos += vel * DT_AIR
        t   += DT_AIR

        xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2]); ts.append(t)

        if pos[2] <= 0.0 and t > 0.05:
            # Interpolate to ground
            i = len(zs)-1
            if i > 0 and zs[i] != zs[i-1]:
                a = -zs[i-1] / (zs[i] - zs[i-1])
                xs[i] = xs[i-1] + a*(xs[i]-xs[i-1])
                ys[i] = ys[i-1] + a*(ys[i]-ys[i-1])
                zs[i] = 0.0
            break

    landing_vel = vel.copy()
    landing_spin = spin_rad * 60.0/(2*math.pi)
    return (np.array(xs), np.array(ys), np.array(zs), np.array(ts),
            landing_vel, landing_spin, tilt)          # 7 values

# -------------------------------------------------------------------------
# 5. BOUNCE (simple, visible hop)
# -------------------------------------------------------------------------
def bounce(vel, spin_rpm, tilt_deg, surface):
    # restitution & tangential friction (PGA Tour averages)
    e_n = {"rough":0.12, "fairway":0.38, "firm":0.48, "green":0.25}.get(surface, 0.38)
    mu_t = {"rough":0.55, "fairway":0.32, "firm":0.27, "green":0.20}.get(surface, 0.32)

    vz_out = -e_n * vel[2]                     # rebound
    v_h = np.hypot(vel[0], vel[1])

    if v_h < 0.5:                               # too slow → no bounce
        return np.array([0.0, 0.0, vz_out]), spin_rpm*0.65

    dir_h = np.array([vel[0], vel[1]]) / v_h
    back_spin = abs(spin_rpm * math.cos(deg2rad(tilt_deg)))
    side_spin = spin_rpm * math.sin(deg2rad(tilt_deg))

    f_t = min(0.9, mu_t * (1.0 + 0.00015*back_spin))
    v_h_out = max(0.0, v_h * (1.0 - f_t))

    vx_out = v_h_out * dir_h[0]
    vy_out = v_h_out * dir_h[1] + 0.001*side_spin

    spin_out = spin_rpm*0.65
    return np.array([vx_out, vy_out, vz_out]), spin_out

# -------------------------------------------------------------------------
# 6. ROLL – YOUR ORIGINAL (kept unchanged)
# -------------------------------------------------------------------------
def simulate_roll(x0, y0, v_land_xy, spin_rpm, surface="fairway"):
    x0 = safe_float(x0); y0 = safe_float(y0)
    spin_rpm = safe_float(spin_rpm, 0.0)
    vx, vy = (safe_float(v_land_xy[0]), safe_float(v_land_xy[1])) if hasattr(v_land_xy, "__len__") else (0.0, 0.0)

    mu_r, _ = SURFACES.get(surface, SURFACES["fairway"])
    v_h = float(np.hypot(vx, vy))
    if v_h < 0.01:
        return np.array([x0]), np.array([y0]), np.array([0.0])

    spin_factor = 1.0 / (1.0 + (spin_rpm / 3500.0))**0.6
    spin_factor = max(0.35, min(0.95, spin_factor))

    direction = np.array([vx, vy]) / v_h
    base_coeff = 0.30
    v_roll = v_h * base_coeff * spin_factor

    decel = g * mu_r
    t_stop = v_roll / decel if decel > 1e-9 else 0.0

    t = np.arange(0.0, t_stop, DT_ROLL) if t_stop > 0 else np.array([0.0])
    if len(t) == 0:
        return np.array([x0]), np.array([y0]), np.array([0.0])

    dist = (v_roll * t) - (0.5 * decel * (t ** 2))
    xr = x0 + dist * direction[0]
    yr = y0 + dist * direction[1]
    return xr, yr, t

# -------------------------------------------------------------------------
# 7. FULL SIMULATION (bounce + roll + short hop)
# -------------------------------------------------------------------------
def simulate(bd, surface="fairway", mu_r=None, k_quad=None):
    # ---- Air ----
    ax, ay, az, at, v_land, spin_land, tilt = simulate_air(bd)

    # ---- Override surface roll friction if requested ----
    if mu_r is not None:
        cur = list(SURFACES.get(surface, SURFACES["fairway"]))
        cur[0] = mu_r
        SURFACES[surface] = tuple(cur)

    # ---- Bounce (up to 2 hops) ----
    vel  = v_land.copy()
    spin = spin_land
    bounces = 0
    max_b = 2

    while bounces < max_b and vel[2] < -0.5:
        vel, spin = bounce(vel, spin, tilt, surface)
        bounces += 1

    # ---- Short hop air (if still going up) ----
    if vel[2] > 0.2:
        pos = np.array([ax[-1], ay[-1], 0.0])
        t0  = at[-1]
        while pos[2] > 0.0 and t0 < MAX_T_AIR:
            vmag = np.linalg.norm(vel) + 1e-12
            vhat = vel / vmag
            q    = 0.5 * rho * vmag**2
            Fd   = -Cd * q * area * vhat
            Fg   = np.array([0.0, 0.0, -mass*g])
            acc  = (Fd + Fg) / mass
            vel += acc * DT_AIR
            pos += vel * DT_AIR
            t0  += DT_AIR
            ax = np.append(ax, pos[0])
            ay = np.append(ay, pos[1])
            az = np.append(az, pos[2])
            at = np.append(at, t0)
        # Interpolate to ground
        if len(az)>1 and az[-1]<0:
            a = -az[-2]/(az[-1]-az[-2])
            ax[-1] = ax[-2] + a*(ax[-1]-ax[-2])
            ay[-1] = ay[-2] + a*(ay[-1]-ay[-2])
            az[-1] = 0.0

    # ---- Roll (your original) ----
    rx, ry, rt = simulate_roll(ax[-1], ay[-1], [vel[0], vel[1]], spin, surface)

    roll_x = np.concatenate([ax[-1:], rx[1:]]) if len(rx)>1 else ax
    roll_y = np.concatenate([ay[-1:], ry[1:]]) if len(ry)>1 else ay
    roll_z = np.zeros_like(roll_x)
    roll_t = np.concatenate([at[-1:], at[-1] + rt[1:]]) if len(rt)>1 else at

    return {
        "air_x": ax, "air_y": ay, "air_z": az, "air_t": at,
        "roll_x": roll_x, "roll_y": roll_y, "roll_z": roll_z, "roll_t": roll_t
    }

# -------------------------------------------------------------------------
# 8. ANIMATION (shows bounce)
# -------------------------------------------------------------------------
def animate_combined(sim, playback=1.0, carry_color="#1f77b4", roll_color="#ff7f0e"):
    Xc = m2yd(sim["air_x"]); Yc = m2yd(sim["air_y"]); Zc = m2ft(sim["air_z"]); Tc = sim["air_t"]
    Xr = m2yd(sim["roll_x"]); Yr = m2yd(sim["roll_y"]); Zr = m2ft(sim["roll_z"]); Tr = sim["roll_t"]

    tc_max = Tc[-1]
    tr_max = Tr[-1] if len(Tr)>1 else 0.0
    total_t = tc_max + tr_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Golf Ball Flight – Carry + Bounce + Roll", fontsize=14)

    # Side view
    ax1.set_title("Side View"); ax1.set_xlabel("Downrange (yd)"); ax1.set_ylabel("Height (ft)")
    ax1.grid(True, alpha=0.3)
    ax1.plot(Xc, Zc, color=carry_color, lw=1.5, label="Carry")
    if len(Xr)>1: ax1.plot(Xr, Zr, color=roll_color, lw=2.0, label="Roll")
    ball1, = ax1.plot([], [], 'o', ms=8, color=carry_color)
    trail1, = ax1.plot([], [], lw=2, color=carry_color, alpha=0.6)

    xmax = max(np.max(Xc), np.max(Xr) if len(Xr) else 0) * 1.06
    zmax = max(np.max(Zc), 1.0) * 1.12
    ax1.set_xlim(0, xmax); ax1.set_ylim(0, zmax); ax1.legend()

    # Top view
    ax2.set_title("Top View"); ax2.set_xlabel("Downrange (yd)"); ax2.set_ylabel("Lateral (yd)")
    ax2.grid(True, alpha=0.3)
    ax2.plot(Xc, Yc, color=carry_color, lw=1.5, label="Carry")
    if len(Xr)>1: ax2.plot(Xr, Yr, color=roll_color, lw=2.0, label="Roll")
    ball2, = ax2.plot([], [], 'o', ms=8, color=carry_color)
    trail2, = ax2.plot([], [], lw=2, color=carry_color, alpha=0.6)

    ymax = max(1.0, np.max(np.abs(np.concatenate([Yc, Yr]))) * 1.12)
    ax2.set_xlim(0, xmax); ax2.set_ylim(-ymax, ymax); ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    # Animation
    trail_sec = 0.45
    trail_air  = max(5, int(trail_sec / DT_AIR))
    trail_roll = max(5, int(trail_sec / DT_ROLL))
    fps = 60
    frame_dt = 1.0/fps
    nframes = int(total_t / frame_dt * playback) + 1

    def update(frame):
        t = frame * frame_dt
        if t <= tc_max:
            i = min(len(Tc)-1, int(t / DT_AIR))
            ball1.set_data([Xc[i]], [Zc[i]])
            trail1.set_data(Xc[max(0,i-trail_air):i+1], Zc[max(0,i-trail_air):i+1])
            ball2.set_data([Xc[i]], [Yc[i]])
            trail2.set_data(Xc[max(0,i-trail_air):i+1], Yc[max(0,i-trail_air):i+1])
        else:
            dt = t - tc_max
            i = min(len(Tr)-1, int(dt / DT_ROLL))
            ball1.set_data([Xr[i]], [Zr[i]])
            trail1.set_data(Xr[max(0,i-trail_roll):i+1], Zr[max(0,i-trail_roll):i+1])
            ball2.set_data([Xr[i]], [Yr[i]])
            trail2.set_data(Xr[max(0,i-trail_roll):i+1], Yr[max(0,i-trail_roll):i+1])
        return ball1, trail1, ball2, trail2

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 9. CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    args = handle_arguments()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        shots = [s for s in data if s.get("Club") == args.shot]
        bd = shots[0]["BallData"] if shots else data[0].get("BallData")
    else:
        bd = data.get("BallData")

    if bd is None:
        raise ValueError("BallData not found in JSON")

    sim = simulate(bd, surface=args.surface, mu_r=args.mu_r)

    carry_yd = m2yd(sim["air_x"][-1])
    total_yd = m2yd(sim["roll_x"][-1])
    apex_ft  = m2ft(np.max(sim["air_z"]))
    flight_s = sim["roll_t"][-1]
    roll_yd  = total_yd - carry_yd

    print(f"Carry: {carry_yd:.1f} yd | Total: {total_yd:.1f} yd | "
          f"Apex: {apex_ft:.1f} ft | Time: {flight_s:.2f} s | Roll: {roll_yd:.1f} yd")

    animate_combined(sim, playback=args.playback)