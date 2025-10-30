import json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helpers import handle_arguments

# -------------------------------------------------------------------------
# 1) CONSTANTS
# -------------------------------------------------------------------------
g       = 9.80665
rho     = 1.225
mass    = 0.04593
radius  = 0.021335
area    = math.pi * radius**2

DT_AIR  = 0.001
DT_ROLL = 0.005

# Grass friction (tuned conservative to avoid runaway roll)
# MU_R: rolling resistance during pure roll
# MU_K: kinetic friction during initial skid
MU_R = {
    "rough":   0.075,
    "fairway": 0.060,   # was lower; now grabbier to match LM-style rollouts
    "firm":    0.040,
    "green":   0.055,
}
MU_K = {
    "rough":   0.38,
    "fairway": 0.30,
    "firm":    0.28,
    "green":   0.26,
}

# -------------------------------------------------------------------------
# 2) HELPERS & UNITS
# -------------------------------------------------------------------------
mph2mps = lambda v: v * 0.44704
deg2rad = lambda d: d * math.pi / 180.0
m2yd    = lambda m: m * 1.0936133
m2ft    = lambda m: m * 3.28084

def safe_float(x, default=0.0):
    try: return float(x)
    except (TypeError, ValueError): return float(default)

def get_target_roll_yards(bd):
    """Try common keys that a launch monitor might include for roll (yards)."""
    carry = bd.get('Carry')
    totalDistance = bd.get('TotalDistance')

    roll = float(totalDistance - carry)
    return roll
    

# -------------------------------------------------------------------------
# 3) AERODYNAMICS (keep your choices that gave good carry/apex)
# -------------------------------------------------------------------------
def drag_coeff(v_mps):
    # Constant Cd by your earlier choice
    return 0.36

def lift_coeff(spin_rpm):
    # Linear CL with cap (your earlier choice)
    return min(0.35, 0.0005 * abs(spin_rpm) + 0.002)

# -------------------------------------------------------------------------
# 4) INITIALS
# -------------------------------------------------------------------------
def initial_velocity(bd):
    v_mph = safe_float(bd.get("BallSpeed"))
    vla   = safe_float(bd.get("VLA"))
    hla   = safe_float(bd.get("HLA"))
    v     = mph2mps(v_mph)
    vla   = deg2rad(vla)
    hla   = deg2rad(hla)
    vx = v * math.cos(vla) * math.cos(hla)
    vy = v * math.cos(vla) * math.sin(hla)
    vz = v * math.sin(vla)
    return np.array([vx, vy, vz], dtype=float)

def spin_inputs(bd):
    back = safe_float(bd.get("BackSpin"))
    side = safe_float(bd.get("SideSpin"))
    rpm  = float(np.hypot(back, side))
    axis_deg = math.degrees(math.atan2(side, back)) if back or side else 0.0
    omega_vec = np.array([0.0, back, side], dtype=float) * (2*math.pi/60.0)
    return rpm, axis_deg, omega_vec

# -------------------------------------------------------------------------
# 5) AIR (CARRY ONLY — stop at first touch)
# -------------------------------------------------------------------------
def simulate_air_carry(bd):
    vel = initial_velocity(bd)
    spin_rpm, axis_deg, omega_vec = spin_inputs(bd)

    pos = np.zeros(3, dtype=float)
    xs, ys, zs, ts = [0.0], [0.0], [0.0], [0.0]
    t = 0.0

    while True:
        v_mag = float(np.linalg.norm(vel))
        if v_mag < 1e-6: break
        v_hat = vel / v_mag
        q = 0.5 * rho * v_mag**2

        # Drag
        Cd = drag_coeff(v_mag)
        Fd = -Cd * q * area * v_hat

        # Lift (Magnus)
        Cl = lift_coeff(spin_rpm)
        lift_dir = np.cross(v_hat, omega_vec)
        ln = float(np.linalg.norm(lift_dir))
        if ln > 0.0:
            lift_dir /= ln
        Fl = Cl * q * area * lift_dir

        # Gravity
        Fg = np.array([0.0, 0.0, -mass*g])

        acc = (Fd + Fl + Fg) / mass
        vel += acc * DT_AIR
        pos += vel * DT_AIR
        t   += DT_AIR

        xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2]); ts.append(t)

        # First touchdown only
        if pos[2] <= 0.0 and t > 0.05:
            z1, z2 = zs[-2], zs[-1]
            x1, x2 = xs[-2], xs[-1]
            y1, y2 = ys[-2], ys[-1]
            if z2 != z1:
                a = -z1 / (z2 - z1)
                xs[-1] = x1 + a*(x2 - x1)
                ys[-1] = y1 + a*(y2 - y1)
                zs[-1] = 0.0
            break

    landing_vel  = vel.copy()
    landing_spin = spin_rpm
    landing_axis = axis_deg
    return (np.array(xs), np.array(ys), np.array(zs), np.array(ts),
            landing_vel, landing_spin, landing_axis)

# -------------------------------------------------------------------------
# 6) BOUNCE IMPULSE (grass)
# -------------------------------------------------------------------------
def bounce_impulse(vel_in, spin_rpm, axis_deg, surface="fairway"):
    """
    Grass-like bounce: keep some forward speed, reduce with tangential friction.
    """
    e_n  = {"rough":0.18, "fairway":0.35, "firm":0.42, "green":0.28}.get(surface, 0.35)
    mu_t = {"rough":0.55, "fairway":0.36, "firm":0.30, "green":0.25}.get(surface, 0.36)

    vz_out = -e_n * vel_in[2]
    v_h    = float(np.hypot(vel_in[0], vel_in[1]))
    if v_h < 0.3:
        return np.array([0.0, 0.0, vz_out], dtype=float), spin_rpm * 0.65

    dir_h  = np.array([vel_in[0], vel_in[1]], dtype=float) / v_h

    # Tangential loss — tuned not to zero-out forward speed
    base_loss = mu_t * (1.0 + e_n) * 0.35
    spin_loss = min(0.08, max(0.0, (spin_rpm - 3000.0) / 15000.0))
    loss = min(0.55, base_loss + spin_loss)

    # Keep at least 35% of incoming horizontal speed
    v_h_out = max(0.35 * v_h, v_h * (1.0 - loss))

    # Tiny lateral kick from spin axis (draw/fade feel)
    side_kick = 0.0006 * spin_rpm * math.sin(deg2rad(axis_deg))

    vx_out = v_h_out * dir_h[0]
    vy_out = v_h_out * dir_h[1] + side_kick
    spin_out = spin_rpm * 0.65
    return np.array([vx_out, vy_out, vz_out], dtype=float), spin_out

# -------------------------------------------------------------------------
# 7) SHORT HOP AFTER BOUNCE (preserve touchdown horizontal velocity)
# -------------------------------------------------------------------------
def simulate_bounce_hop(x0, y0, vel_after_bounce):
    """
    Simulate a short ballistic hop; return hop arrays AND the horizontal
    velocity immediately before ground re-contact (for roll input).
    """
    pos = np.array([x0, y0, 0.0], dtype=float)
    vel = vel_after_bounce.astype(float).copy()

    bx, by, bz, bt = [], [], [], []
    t = 0.0
    vel_prev = vel.copy()
    pos_prev = pos.copy()

    while True:
        vel_prev[:] = vel
        pos_prev[:] = pos

        v_mag = float(np.linalg.norm(vel)) + 1e-12
        v_hat = vel / v_mag
        q     = 0.5 * rho * v_mag**2

        Cd = drag_coeff(v_mag)
        Fd = -Cd * q * area * v_hat
        Fg = np.array([0.0, 0.0, -mass*g])
        acc = (Fd + Fg) / mass

        vel += acc * DT_AIR
        pos += vel * DT_AIR
        t   += DT_AIR

        bx.append(pos[0]); by.append(pos[1]); bz.append(pos[2]); bt.append(t)

        if pos[2] <= 0.0 and t > 0.01:
            # Interpolate last position to z=0
            z1, z2 = pos_prev[2], pos[2]
            x1, x2 = pos_prev[0], pos[0]
            y1, y2 = pos_prev[1], pos[1]
            if z2 != z1:
                a = (0 - z1) / (z2 - z1)
                bx[-1] = x1 + a*(x2 - x1)
                by[-1] = y1 + a*(y2 - y1)
                bz[-1] = 0.0
            # Horizontal velocity just BEFORE ground contact
            v_td_xy = np.array([vel_prev[0], vel_prev[1]], dtype=float)
            return (np.array(bx), np.array(by), np.array(bz), np.array(bt), v_td_xy)

        if t > 1.2:  # safety
            v_td_xy = np.array([vel[0], vel[1]], dtype=float)
            return (np.array(bx), np.array(by), np.array(bz), np.array(bt), v_td_xy)

# -------------------------------------------------------------------------
# 8) GROUND: SKID → PURE ROLL (with optional target roll fit)
# -------------------------------------------------------------------------
def simulate_roll(x0, y0, v0_xy, spin_rpm, surface="fairway", target_roll_yds=None):
    """
    Returns xr, yr, tr, roll_distance_m.
    If target_roll_yds (from LM) is provided, μ_r is solved so roll ≈ target.
    """
    x0 = safe_float(x0); y0 = safe_float(y0)
    vx = safe_float(v0_xy[0]); vy = safe_float(v0_xy[1])
    spin_rpm = safe_float(spin_rpm)

    v0 = float(np.hypot(vx, vy))
    if v0 < 0.05:
        return np.array([x0]), np.array([y0]), np.array([0.0]), 0.0

    mu_k_base = MU_K.get(surface, MU_K["fairway"])
    mu_r_base = MU_R.get(surface, MU_R["fairway"])

    # Slight spin dependence (more spin → less roll)
    mu_k = mu_k_base * (1.0 + min(0.20, max(0.0, (spin_rpm - 3000.0)/15000.0)))
    mu_r = mu_r_base * (1.0 + min(0.25, max(0.0, (spin_rpm - 3000.0)/12000.0)))

    # Skid duration — ensure it's not skipped entirely
    omega0 = spin_rpm * (2.0 * math.pi / 60.0)
    v_spin = omega0 * radius
    v_slip0 = v0 - v_spin
    if v_slip0 <= 0.0:
        t_r = 0.03  # tiny skid on grass
    else:
        t_r = (2.0 / 7.0) * v_slip0 / max(1e-6, mu_k * g)
        t_r = max(0.02, min(t_r, 0.5))

    v1 = max(0.0, v0 - mu_k * g * t_r)           # speed at pure-roll start
    s1 = v0 * t_r - 0.5 * mu_k * g * t_r**2      # distance during skid

    # Fit μ_r to LM target if provided
    if target_roll_yds is not None:
        target_total_m = max(0.0, target_roll_yds * 0.9144)
        s2_desired = max(0.0, target_total_m - s1)
        if v1 <= 0.02 or s2_desired <= 0.01:
            t2 = 0.0; s2 = 0.0; mu_r_eff = mu_r
        else:
            mu_r_eff = max(mu_r, (v1**2) / (2.0 * g * s2_desired))
            t2 = v1 / (mu_r_eff * g)
            s2 = (v1**2) / (2.0 * mu_r_eff * g)
    else:
        if v1 <= 0.02:
            t2 = 0.0; s2 = 0.0; mu_r_eff = mu_r
        else:
            t2 = v1 / (mu_r * g)
            s2 = (v1**2) / (2.0 * mu_r * g)
            mu_r_eff = mu_r

    # Guardrail for fairway: cap total roll distance unless surface says otherwise
    s_total = s1 + s2
    if surface == "fairway":
        s_total = min(s_total, 40.0 * 0.9144)  # ≤ 40 yd

    # Discretize for animation
    dir_xy = np.array([vx, vy], dtype=float) / v0
    t_total = t_r + (v1 / (mu_r_eff * g) if v1 > 0.02 else 0.0)
    if t_total <= 0.0:
        return np.array([x0]), np.array([y0]), np.array([0.0]), s_total

    step = max(DT_ROLL, 0.005)
    t = np.arange(0.0, t_total + 1e-9, step)
    s = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti <= t_r:
            s[i] = v0 * ti - 0.5 * mu_k * g * ti**2
        else:
            tau = ti - t_r
            s_roll = (v1 * tau - 0.5 * mu_r_eff * g * tau**2)
            s[i] = min(s1 + s_roll, s_total)

    xr = x0 + s * dir_xy[0]
    yr = y0 + s * dir_xy[1]
    return xr, yr, t, s_total

# -------------------------------------------------------------------------
# 9) FULL PIPELINE: CARRY → BOUNCE/HOP → ROLL
# -------------------------------------------------------------------------
def simulate(bd, surface="fairway"):
    target_roll_yds = get_target_roll_yards(bd)  # may be None

    # Carry (stops at first touch)
    cx, cy, cz, ct, v_land, spin_land, axis_deg = simulate_air_carry(bd)

    # Bounce impulse at touchdown
    v_bounce, spin_after = bounce_impulse(v_land, spin_land, axis_deg, surface)

    # Optional short hop
    bx = by = bz = bt = np.array([])
    if v_bounce[2] > 0.20:
        bx, by, bz, bt, v_td_xy = simulate_bounce_hop(cx[-1], cy[-1], v_bounce)
        v_ground_xy = v_td_xy
        gx, gy = (bx[-1], by[-1]) if len(bx) else (cx[-1], cy[-1])
    else:
        v_ground_xy = np.array([v_bounce[0], v_bounce[1]])
        gx, gy = cx[-1], cy[-1]

    # Roll (fit to LM target roll if provided)
    rx, ry, rt, roll_m = simulate_roll(gx, gy, v_ground_xy, spin_after,
                                       surface=surface, target_roll_yds=target_roll_yds)

    return {
        "carry_x": cx, "carry_y": cy, "carry_z": cz, "carry_t": ct,
        "bounce_x": bx, "bounce_y": by, "bounce_z": bz, "bounce_t": bt,
        "roll_x": rx, "roll_y": ry, "roll_t": rt,
        "roll_m": roll_m
    }

# -------------------------------------------------------------------------
# 10) ANIMATION: side + top (carry blue, bounce green, roll orange)
# -------------------------------------------------------------------------
def animate_combined(sim, playback=1.0):
    Cx = m2yd(sim["carry_x"]);  Cy = m2yd(sim["carry_y"]);  Cz = m2ft(sim["carry_z"]);  Ct = sim["carry_t"]
    Bx = m2yd(sim["bounce_x"]); By = m2yd(sim["bounce_y"]); Bz = m2ft(sim["bounce_z"]); Bt = sim["bounce_t"]
    Rx = m2yd(sim["roll_x"]);   Ry = m2yd(sim["roll_y"]);   Rz = np.zeros_like(Rx);     Rt = sim["roll_t"]

    total_t = Ct[-1] + (Bt[-1] if len(Bt) else 0.0) + (Rt[-1] if len(Rt) else 0.0)
    xmax = max(np.max(Cx), np.max(Bx) if len(Bx) else 0, np.max(Rx) if len(Rx) else 0, 5.0) * 1.06
    zmax = max(np.max(Cz), np.max(Bz) if len(Bz) else 0, 1.0) * 1.12
    ymax = max(1.0, np.max(np.abs(np.concatenate([
        Cy, By if len(By) else np.array([0.0]), Ry if len(Ry) else np.array([0.0])
    ]))) * 1.12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Golf Ball Flight – Carry (blue) • Bounce (green) • Roll (orange)", fontsize=13)

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
    ax2.plot(Cx, Cy, color="#1f77b4", lw=1.8, label="Carry")
    if len(Bx): ax2.plot(Bx, By, color="#2ca02c", lw=2.0, label="Bounce")
    if len(Rx): ax2.plot(Rx, Ry, color="#ff7f0e", lw=2.2, label="Roll")
    ball2, = ax2.plot([], [], 'o', ms=8, color="#1f77b4")
    trail2, = ax2.plot([], [], lw=2, color="#1f77b4", alpha=0.6)
    ax2.set_xlim(0, xmax); ax2.set_ylim(-ymax, ymax); ax2.set_aspect('equal'); ax2.legend()

    # Unified timeline: carry -> bounce -> roll
    T_air = Ct
    T_bnc = (Ct[-1] + Bt) if len(Bt) else np.array([])
    T_rol = (Ct[-1] + (Bt[-1] if len(Bt) else 0.0) + Rt) if len(Rt) else np.array([])

    fps = 60; frame_dt = 1.0/fps
    nframes = int(total_t / frame_dt * playback) + 1
    trail_air  = max(5, int(0.45 / DT_AIR))
    trail_roll = max(5, int(0.45 / DT_ROLL))

    def update(frame):
        t = frame * frame_dt
        # carry
        if t <= T_air[-1]:
            i = min(len(T_air)-1, int(t / DT_AIR))
            col = "#1f77b4"
            ball1.set_color(col); trail1.set_color(col)
            ball2.set_color(col); trail2.set_color(col)
            ball1.set_data([Cx[i]], [Cz[i]])
            trail1.set_data(Cx[max(0, i-trail_air):i+1], Cz[max(0, i-trail_air):i+1])
            ball2.set_data([Cx[i]], [Cy[i]])
            trail2.set_data(Cx[max(0, i-trail_air):i+1], Cy[max(0, i-trail_air):i+1])
            return ball1, trail1, ball2, trail2

        # bounce
        if len(T_bnc) and t <= T_bnc[-1]:
            tb = t - T_air[-1]
            i = min(len(Bt)-1, int(tb / DT_AIR))
            col = "#2ca02c"
            ball1.set_color(col); trail1.set_color(col)
            ball2.set_color(col); trail2.set_color(col)
            ball1.set_data([Bx[i]], [Bz[i]])
            trail1.set_data(Bx[max(0, i-trail_air):i+1], Bz[max(0, i-trail_air):i+1])
            ball2.set_data([Bx[i]], [By[i]])
            trail2.set_data(Bx[max(0, i-trail_air):i+1], By[max(0, i-trail_air):i+1])
            return ball1, trail1, ball2, trail2

        # roll
        if len(T_rol):
            tr = t - (T_air[-1] + (Bt[-1] if len(Bt) else 0.0))
            i = min(len(Rt)-1, int(tr / DT_ROLL))
            col = "#ff7f0e"
            ball1.set_color(col); trail1.set_color(col)
            ball2.set_color(col); trail2.set_color(col)
            ball1.set_data([Rx[i]], [Rz[i]])
            trail1.set_data(Rx[max(0, i-trail_roll):i+1], Rz[max(0, i-trail_roll):i+1])
            ball2.set_data([Rx[i]], [Ry[i]])
            trail2.set_data(Rx[max(0, i-trail_roll):i+1], Ry[max(0, i-trail_roll):i+1])
            return ball1, trail1, ball2, trail2

        return ball1, trail1, ball2, trail2

    FuncAnimation(fig, update, frames=nframes, interval=1000/fps, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 11) CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    args = handle_arguments()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = [s for s in data if s.get("Club") == args.shot]
    bd = shots[0] if shots else data[0]

    sim = simulate(bd, surface=args.surface)

    # Carry ends at first touch
    carry_yd = m2yd(sim["carry_x"][-1])

    # Bounce downrange distance
    bounce_yd = 0.0
    if len(sim.get("bounce_x", [])):
        bounce_yd = m2yd(sim["bounce_x"][-1] - sim["carry_x"][-1])

    # Total & roll (roll here = ground roll only, not including bounce)
    total_yd = m2yd(sim["roll_x"][-1] if len(sim["roll_x"]) else sim["carry_x"][-1])
    roll_only_yd = total_yd - carry_yd - bounce_yd

    apex_ft  = m2ft(np.max(sim["carry_z"]))
    flight_s = (sim["carry_t"][-1] +
                (sim["bounce_t"][-1] if len(sim.get("bounce_t", [])) else 0.0) +
                (sim["roll_t"][-1] if len(sim.get("roll_t", [])) else 0.0))

    print(f"Carry: {carry_yd:.1f} yd | Bounce: {bounce_yd:.1f} yd | "
          f"Roll: {roll_only_yd:.1f} yd | Total: {total_yd:.1f} yd | "
          f"Apex: {apex_ft:.1f} ft | Time: {flight_s:.2f} s")

    animate_combined(sim, playback=args.playback)
