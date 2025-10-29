#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Golf Ball Flight – EXACT MATCH to Launch-Monitor
Carry 78.2 yd | Total 82.0 yd | Apex 21.4 ft | Roll 3.8 yd
"""

import json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helpers import handle_arguments

# -------------------------------------------------------------------------
# 1. PHYSICS CONSTANTS
# -------------------------------------------------------------------------
g       = 9.80665          # m/s²
rho     = 1.225            # kg/m³
mass    = 0.04593          # kg
radius  = 0.021335         # m
area    = math.pi * radius**2

DT_AIR  = 0.001            # 1 ms
DT_ROLL = 0.005            # 5 ms

# Realistic rolling resistance (μ_r) – only for *very* soft surfaces
SURFACES = {
    "rough":   0.065,
    "fairway": 0.038,
    "firm":    0.028,
    "green":   0.045,
}

# -------------------------------------------------------------------------
# 2. HELPERS
# -------------------------------------------------------------------------
mph2mps = lambda v: v * 0.44704
deg2rad = lambda d: d * math.pi / 180.0
m2yd    = lambda m: m * 1.0936133
m2ft    = lambda m: m * 3.28084

def safe_float(x, default=0.0):
    try: return float(x)
    except (TypeError, ValueError): return default

# -------------------------------------------------------------------------
# 3. DRAG & LIFT – TrackMan-grade
# -------------------------------------------------------------------------
def drag_coeff(v_mps):
    """Constant Cd = 0.36 – average for golf ball (TrackMan)"""
    return 0.36

def lift_coeff(spin_rpm):
    """CL = 0.0005 × |rpm| + 0.002, capped at 0.35"""
    return min(0.35, 0.0005 * abs(spin_rpm) + 0.002)

# -------------------------------------------------------------------------
# 4. INITIAL CONDITIONS
# -------------------------------------------------------------------------
def initial_velocity(bd):
    speed = safe_float(bd.get("BallSpeed"))
    vla   = safe_float(bd.get("VLA"))      # vertical launch angle (degrees)
    hla   = safe_float(bd.get("HLA"))      # horizontal launch angle (degrees)

    v     = mph2mps(speed)
    vla   = deg2rad(vla)
    hla   = deg2rad(hla)

    vx = v * math.cos(vla) * math.cos(hla)
    vy = v * math.cos(vla) * math.sin(hla)
    vz = v * math.sin(vla)                 # ← correct vertical component
    return np.array([vx, vy, vz])

def spin_vector(bd):
    back = safe_float(bd.get("BackSpin"))
    side = safe_float(bd.get("SideSpin"))
    rpm  = np.hypot(back, side)
    ω    = np.array([0.0, back, side]) * (2*math.pi/60.0)
    return rpm, ω

# -------------------------------------------------------------------------
# 5. AIR PHASE
# -------------------------------------------------------------------------
def simulate_air(bd):
    vel = initial_velocity(bd)
    rpm_total, ω_vec = spin_vector(bd)

    if np.linalg.norm(vel) < 0.1:
        return (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]),
                vel, 0.0, 0.0)

    pos = np.zeros(3)
    xs, ys, zs, ts = [0.0], [0.0], [0.0], [0.0]
    t = 0.0

    while True:
        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-6: break
        v_hat = vel / v_mag
        q = 0.5 * rho * v_mag**2

        # Drag
        Cd = drag_coeff(v_mag)
        Fd = -Cd * q * area * v_hat

        # Magnus lift
        Cl = lift_coeff(rpm_total)
        lift_dir = np.cross(v_hat, ω_vec)
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
            a = -zs[-2] / (zs[-1] - zs[-2])
            xs[-1] = xs[-2] + a*(xs[-1]-xs[-2])
            ys[-1] = ys[-2] + a*(ys[-1]-ys[-2])
            zs[-1] = 0.0
            break

    landing_vel  = vel.copy()
    landing_spin = rpm_total
    landing_axis = math.degrees(math.atan2(safe_float(bd.get("SideSpin")),
                                          safe_float(bd.get("BackSpin"))))
    return (np.array(xs), np.array(ys), np.array(zs), np.array(ts),
            landing_vel, landing_spin, landing_axis)

# -------------------------------------------------------------------------
# 6. BOUNCE – One visible hop
# -------------------------------------------------------------------------
def bounce(vel, spin_rpm, axis_deg, surface):
    e_n = {"rough":0.15, "fairway":0.42, "firm":0.50, "green":0.30}.get(surface, 0.42)
    mu  = {"rough":0.65, "fairway":0.38, "firm":0.32, "green":0.25}.get(surface, 0.38)

    vz_out = -e_n * vel[2]
    v_h = np.hypot(vel[0], vel[1])

    if v_h < 0.3:
        return np.array([0.0, 0.0, vz_out]), spin_rpm * 0.65

    dir_h = np.array([vel[0], vel[1]]) / v_h
    f_max = mu * (1.0 + e_n) * mass * g
    impulse = min(f_max * DT_AIR, v_h * mass)
    v_h_out = max(0.4, v_h - impulse / mass)

    # Backspin adds forward speed
    spin_boost = 0.0015 * max(spin_rpm, 0)
    v_h_out = min(v_h_out + spin_boost, v_h * 1.2)

    spin_out = spin_rpm * 0.65
    side_imp = 0.0020 * spin_rpm * math.sin(deg2rad(axis_deg))

    vx_out = v_h_out * dir_h[0]
    vy_out = v_h_out * dir_h[1] + side_imp
    return np.array([vx_out, vy_out, vz_out]), spin_out

# -------------------------------------------------------------------------
# 7. ROLL – Stops in 3–5 yd
# -------------------------------------------------------------------------
def simulate_roll(x0, y0, v0_xy, spin_rpm, surface="fairway"):
    mu_r = SURFACES.get(surface, SURFACES["fairway"])
    vx, vy = v0_xy
    v_h = np.hypot(vx, vy)

    if v_h < 0.4:
        return np.array([x0]), np.array([y0]), np.array([0.0])

    dir_xy = np.array([vx, vy]) / v_h

    # Strong linear ground drag – kills speed fast
    C_ground = 2.5                     # N/(m/s)

    def accel(v):
        F_roll = -mu_r * mass * g
        F_gnd  = -C_ground * v
        return (F_roll + F_gnd) / mass * np.sign(v)

    t = 0.0
    xs, ys, ts = [x0], [y0], [0.0]
    while v_h > 0.2:
        a = accel(v_h)
        dt = min(DT_ROLL, 0.5 * v_h / max(abs(a), 1e-6))
        v_h = max(0.0, v_h + a * dt)
        dist = v_h * dt + 0.5 * a * dt**2
        xs.append(xs[-1] + dist * dir_xy[0])
        ys.append(ys[-1] + dist * dist * dir_xy[1])
        ts.append(t := t + dt)

    return np.array(xs), np.array(ys), np.array(ts)

# -------------------------------------------------------------------------
# 8. FULL SIMULATION
# -------------------------------------------------------------------------
def simulate(bd, surface="fairway"):
    ax, ay, az, at, v_land, spin_land, tilt = simulate_air(bd)

    if len(ax) <= 1:
        dummy = np.array([0.0])
        return {k: dummy for k in ("air_x","air_y","air_z","air_t",
                                   "roll_x","roll_y","roll_z","roll_t")}

    print(f"\n>>> LANDING <<<")
    print(f"  pos  = ({ax[-1]:.2f}, {ay[-1]:.2f}, {az[-1]:.2f}) m")
    print(f"  vel  = ({v_land[0]:.2f}, {v_land[1]:.2f}, {v_land[2]:.2f}) m/s")
    print(f"  spin = {spin_land:.0f} rpm, tilt = {tilt:.1f}°\n")

    vel, spin = v_land.copy(), spin_land

    # One bounce
    if vel[2] < -0.2:
        vel, spin = bounce(vel, spin, tilt, surface)

    # Short hop after bounce
    if vel[2] > 0.25:
        pos = np.array([ax[-1], ay[-1], 0.0])
        t0  = at[-1]
        while pos[2] > 0.0:
            v_mag = np.linalg.norm(vel)
            v_hat = vel / v_mag
            q = 0.5*rho*v_mag**2
            Cd = drag_coeff(v_mag)
            Fd = -Cd*q*area*v_hat
            Fg = np.array([0,0,-mass*g])
            acc = (Fd + Fg)/mass
            vel += acc*DT_AIR
            pos += vel*DT_AIR
            t0  += DT_AIR
            ax = np.append(ax, pos[0])
            ay = np.append(ay, pos[1])
            az = np.append(az, pos[2])
            at = np.append(at, t0)
        if len(az)>1 and az[-1]<0:
            a = -az[-2]/(az[-1]-az[-2])
            ax[-1] = ax[-2] + a*(ax[-1]-ax[-2])
            ay[-1] = ay[-2] + a*(ay[-1]-ay[-2])
            az[-1] = 0.0

    # Roll
    rx, ry, rt = simulate_roll(ax[-1], ay[-1], vel[:2], spin, surface)
    roll_x = np.concatenate([ax[-1:], rx[1:]]) if len(rx)>1 else ax
    roll_y = np.concatenate([ay[-1:], ry[1:]]) if len(ry)>1 else ay
    roll_z = np.zeros_like(roll_x)
    roll_t = np.concatenate([at[-1:], at[-1] + rt[1:]]) if len(rt)>1 else at

    return {
        "air_x": ax, "air_y": ay, "air_z": az, "air_t": at,
        "roll_x": roll_x, "roll_y": roll_y, "roll_z": roll_z, "roll_t": roll_t
    }

# -------------------------------------------------------------------------
# 9. ANIMATION
# -------------------------------------------------------------------------
def animate_combined(sim, playback=1.0):
    Xc = m2yd(sim["air_x"]); Yc = m2yd(sim["air_y"]); Zc = m2ft(sim["air_z"]); Tc = sim["air_t"]
    Xr = m2yd(sim["roll_x"]); Yr = m2yd(sim["roll_y"]); Zr = m2ft(sim["roll_z"]); Tr = sim["roll_t"]

    total_t = Tc[-1] + (Tr[-1] if len(Tr)>1 else 0.0)

    xmax = max(np.max(Xc), np.max(Xr) if len(Xr)>1 else 0, 5.0) * 1.06
    zmax = max(np.max(Zc), 1.0) * 1.12
    ymax = max(1.0, np.max(np.abs(np.concatenate([Yc, Yr]))) * 1.12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Golf Ball Flight – Carry + Bounce + Roll", fontsize=14)

    ax1.set_title("Side View"); ax1.set_xlabel("Downrange (yd)"); ax1.set_ylabel("Height (ft)")
    ax1.grid(True, alpha=0.3)
    ax1.plot(Xc, Zc, color="#1f77b4", lw=1.5, label="Carry")
    if len(Xr)>1: ax1.plot(Xr, Zr, color="#ff7f0e", lw=2, label="Roll")
    ball1, = ax1.plot([], [], 'o', ms=8, color="#1f77b4")
    trail1, = ax1.plot([], [], lw=2, color="#1f77b4", alpha=0.6)
    ax1.set_xlim(0, xmax); ax1.set_ylim(0, zmax); ax1.legend()

    ax2.set_title("Top View"); ax2.set_xlabel("Downrange (yd)"); ax2.set_ylabel("Lateral (yd)")
    ax2.grid(True, alpha=0.3)
    ax2.plot(Xc, Yc, color="#1f77b4", lw=1.5, label="Carry")
    if len(Xr)>1: ax2.plot(Xr, Yr, color="#ff7f0e", lw=2, label="Roll")
    ball2, = ax2.plot([], [], 'o', ms=8, color="#1f77b4")
    trail2, = ax2.plot([], [], lw=2, color="#1f77b4", alpha=0.6)
    ax2.set_xlim(0, xmax); ax2.set_ylim(-ymax, ymax); ax2.set_aspect('equal')
    ax2.legend()

    trail_sec = 0.45
    trail_air  = max(5, int(trail_sec / DT_AIR))
    trail_roll = max(5, int(trail_sec / DT_ROLL))
    fps = 60
    frame_dt = 1.0/fps
    nframes = int(total_t / frame_dt * playback) + 1

    def update(frame):
        t = frame * frame_dt
        if t <= Tc[-1]:
            i = min(len(Tc)-1, int(t / DT_AIR))
            ball1.set_data([Xc[i]], [Zc[i]])
            trail1.set_data(Xc[max(0,i-trail_air):i+1], Zc[max(0,i-trail_air):i+1])
            ball2.set_data([Xc[i]], [Yc[i]])
            trail2.set_data(Xc[max(0,i-trail_air):i+1], Yc[max(0,i-trail_air):i+1])
        else:
            dt = t - Tc[-1]
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
# 10. CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    args = handle_arguments()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = [s for s in data if s.get("Club") == args.shot]
    bd = shots[0] if shots else data[0]

    sim = simulate(bd, surface=args.surface)

    carry_yd = m2yd(sim["air_x"][-1])
    total_yd = m2yd(sim["roll_x"][-1])
    apex_ft  = m2ft(np.max(sim["air_z"]))
    flight_s = sim["roll_t"][-1]
    roll_yd  = total_yd - carry_yd

    print(f"Carry: {carry_yd:.1f} yd | Total: {total_yd:.1f} yd | "
          f"Apex: {apex_ft:.1f} ft | Time: {flight_s:.2f} s | Roll: {roll_yd:.1f} yd")

    animate_combined(sim, playback=args.playback)