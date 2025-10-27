
import math
import numpy as np


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

def safe_float(x, default=0.0):
    try:
        # Handle strings like "2500", "2500.0", " 2500 ", None
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def simulate_roll(x0, y0, v_land_xy, spin_rpm, surface="fairway"):
    """
    Physically grounded rollout (flat ground).
    Inputs may arrive as strings/arrays; we coerce to float.
    """
    # --- Coerce types safely ---
    x0 = safe_float(x0); y0 = safe_float(y0)
    spin_rpm = safe_float(spin_rpm, 0.0)

    # Ensure v_land_xy is a 2-vector of floats
    vx, vy = (safe_float(v_land_xy[0]), safe_float(v_land_xy[1])) if hasattr(v_land_xy, "__len__") else (0.0, 0.0)

    # Surface params
    mu_r, _ = SURFACES.get(surface, SURFACES["fairway"])

    # Landing horizontal speed
    v_h = float(np.hypot(vx, vy))
    if v_h < 0.01:
        return np.array([x0]), np.array([y0]), np.array([0.0])

    # Spin-based roll reduction (coerced spin_rpm is now float)
    spin_factor = 1.0 / (1.0 + (spin_rpm / 3500.0))**0.6
    spin_factor = max(0.35, min(0.95, spin_factor))

    # Direction and initial roll speed
    direction = np.array([vx, vy]) / v_h
    base_coeff = 0.30  # fairway baseline; feel free to keep your surface-specific version
    v_roll = v_h * base_coeff * spin_factor

    # Constant deceleration
    decel = g * mu_r
    t_stop = v_roll / decel if decel > 1e-9 else 0.0

    t = np.arange(0.0, t_stop, DT_ROLL) if t_stop > 0 else np.array([0.0])
    if len(t) == 0:
        return np.array([x0]), np.array([y0]), np.array([0.0])

    dist = (v_roll * t) - (0.5 * decel * (t ** 2))
    xr = x0 + dist * direction[0]
    yr = y0 + dist * direction[1]
    return xr, yr, t
