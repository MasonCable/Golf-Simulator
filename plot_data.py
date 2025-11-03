#!/usr/bin/env python3
# -*- coding: utf-8__

# CRITICAL: Use TkAgg backend — most stable on macOS + asyncio
import matplotlib
matplotlib.use('TkAgg')  # ← MUST BE BEFORE pyplot import
import matplotlib.pyplot as plt

import math
import numpy as np
from matplotlib.animation import FuncAnimation


# Global state: only one figure + animation at a time
_current_fig = None
_current_ani = None


class PlotData:
    def __init__(self, shotNum):
        self.shotNum = shotNum

    # ————————————————————————————————————————————————————————————————
    # Helpers
    # ————————————————————————————————————————————————————————————————
    def deg2rad(self, d):
        return d * math.pi / 180.0

    def interp_y(self, x, X, Y):
        if x <= X[0]: return float(Y[0])
        if x >= X[-1]: return float(Y[-1])
        i = np.searchsorted(X, x) - 1
        x1, x2 = X[i], X[i+1]
        y1, y2 = Y[i], Y[i+1]
        a = (x - x1) / max(1e-9, (x2 - x1))
        return float(y1 + a * (y2 - y1))

    def interp_y_array(self, x_arr, X, Y):
        return np.array([self.interp_y(x, X, Y) for x in np.asarray(x_arr)], dtype=float)

    def build_lateral_curve(self, total_yd, hla_deg, offline_yd, x_samples):
        """
        Build lateral (Y) path.
        - hla_deg: positive = right
        - offline_yd: positive = right
        """
        if total_yd <= 0:
            return np.zeros_like(x_samples)

        # Convert to radians
        hla_rad = self.deg2rad(hla_deg)
        target_y_at_total = offline_yd  # final Y position

        # Initial slope = tan(HLA)
        initial_slope = math.tan(hla_rad)

        # Quadratic term to hit final Y
        a = (target_y_at_total - initial_slope * total_yd) / (total_yd ** 2)
        b = initial_slope

        y = a * x_samples**2 + b * x_samples
        if len(y) > 0:
            y[-1] = target_y_at_total  # Force final point
        return -y

    # ————————————————————————————————————————————————————————————————
    # Build paths from JSON
    # ————————————————————————————————————————————————————————————————
    def build_paths_from_json(self, bd, bounce_policy=None):
        carry_yd = max(0.0, float(bd.get("Carry", 0.0)))
        total_yd = max(carry_yd, float(bd.get("TotalDistance", carry_yd)))
        peak_ft = float(bd.get("PeakHeight", 0.0))
        hla_deg = float(bd.get("HLA", 0.0))
        offline_yd = float(bd.get("Offline", 0.0))
        roll_yd = max(0.0, total_yd - carry_yd)

        # Carry arc
        n_air = max(200, int(carry_yd * 2))
        x_air = np.linspace(0.0, carry_yd, n_air)
        z_air_ft = (4.0 * peak_ft / (carry_yd**2) * x_air * (carry_yd - x_air)) if carry_yd > 0 and peak_ft > 0 else np.zeros_like(x_air)

        # Bounce
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
            bounce_len = np.clip(bounce_policy["len_frac_of_roll"] * roll_yd,
                                 bounce_policy["len_min_yd"], bounce_policy["len_max_yd"])
            n_bnc = max(6, int(bounce_len * 8))
            x_bnc = np.linspace(carry_yd, carry_yd + bounce_len, n_bnc)
            mid = 0.5 * (x_bnc[0] + x_bnc[-1])
            half = max(1e-9, 0.5 * (x_bnc[-1] - x_bnc[0]))
            z_bnc_ft = bounce_policy["apex_ft"] * (1.0 - ((x_bnc - mid) / half)**2)
            z_bnc_ft[z_bnc_ft < 0] = 0.0
        else:
            x_bnc = z_bnc_ft = np.array([])

        # Roll
        x_roll_start = carry_yd + (x_bnc[-1] - carry_yd if do_bounce else 0.0)
        n_roll = max(2, int(max(2.0, total_yd - x_roll_start) * 2))
        x_roll = np.linspace(x_roll_start, total_yd, n_roll)
        z_roll_ft = np.zeros_like(x_roll)

        # Top view
        x_full = np.concatenate([x_air, x_bnc[1:], x_roll[1:]]) if len(x_roll) > 1 else \
                 (np.concatenate([x_air, x_bnc[1:]]) if len(x_bnc) > 1 else x_air)
        y_full = self.build_lateral_curve(total_yd, hla_deg, offline_yd, x_full)

        # Time
        air_time = 0.012 * carry_yd + 0.9
        bounce_time = 0.08 if do_bounce else 0.0  # ← FAST BOUNCE
        roll_speed_yd_per_sec = 25.0
        roll_time = max(0.3, roll_yd / max(1.0, roll_speed_yd_per_sec))  # ← REALISTIC ROLL
        t_air  = np.linspace(0.0, max(0.02, air_time), len(x_air))
        t_bnc  = np.linspace(0.0, bounce_time, len(x_bnc)) if do_bounce else np.array([])
        t_roll = np.linspace(0.0, max(0.02, roll_time), len(x_roll)) if len(x_roll) else np.array([])

        return {
            "carry_x": x_air, "carry_z": z_air_ft, "carry_t": t_air,
            "bounce_x": x_bnc, "bounce_z": z_bnc_ft, "bounce_t": t_bnc,
            "roll_x": x_roll, "roll_z": z_roll_ft, "roll_t": t_roll,
            "top_x": x_full, "top_y": y_full,
            "meta": {
                "carry_yd": carry_yd, "total_yd": total_yd, "roll_yd": roll_yd,
                "did_bounce": do_bounce,
                "bounce_len_yd": bounce_len if do_bounce else 0.0,
                "bounce_apex_ft": bounce_policy["apex_ft"] if do_bounce else 0.0,
                "peak_ft": peak_ft, "offline_yd": offline_yd, "hla_deg": hla_deg
            }
        }

    # ————————————————————————————————————————————————————————————————
    # Animation
    # ————————————————————————————————————————————————————————————————
    def animate_paths(self, paths, playback, club):
        global _current_fig, _current_ani

        # Close old figure
        if _current_fig is not None:
            plt.close(_current_fig)
            _current_fig = None

        # Stop old animation
        if _current_ani is not None:
            try:
                _current_ani.event_source.stop()
            except:
                pass
            _current_ani = None

        Cx, Cz, Ct = paths["carry_x"], paths["carry_z"], paths["carry_t"]
        Bx, Bz, Bt = paths["bounce_x"], paths["bounce_z"], paths["bounce_t"]
        Rx, Rz, Rt = paths["roll_x"], paths["roll_z"], paths["roll_t"]
        X, Y = paths["top_x"], paths["top_y"]

        total_t = Ct[-1] + (Bt[-1] if len(Bt) else 0) + (Rt[-1] if len(Rt) else 0)
        fps = 60
        frame_dt = 1.0 / fps
        nframes = max(1, int(total_t / frame_dt * playback) + 1)

        xmax = max(np.max(Cx), np.max(Bx) if len(Bx) else 0, np.max(Rx) if len(Rx) else 0, 5) * 1.06
        zmax = max(np.max(Cz), np.max(Bz) if len(Bz) else 0, 1) * 1.12
        ymax = max(1.0, np.max(np.abs(Y)) * 1.12)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        _current_fig = fig        
        fig.suptitle(f"Shot #{self.shotNum}: {paths['meta']['carry_yd']:.1f} yd carry -> {club}", fontsize=13)

        # ——— Side View ———
        ax1.set(xlabel="Downrange (yd)", ylabel="Height (ft)", xlim=(0, xmax), ylim=(0, zmax))
        ax1.grid(True, alpha=0.3)

        line_carry, = ax1.plot(Cx, Cz, "#1f77b4", lw=1.8, label="Carry")
        line_bounce = None
        line_roll = None
        if len(Bx):
            line_bounce, = ax1.plot(Bx, Bz, "#2ca02c", lw=2.0, label="Bounce")
        if len(Rx):
            line_roll, = ax1.plot(Rx, Rz, "#ff7f0e", lw=2.2, label="Roll")

        ball1, = ax1.plot([], [], 'o', ms=8, color="#1f77b4")
        trail1, = ax1.plot([], [], lw=2, color="#1f77b4", alpha=0.6)

        side_handles = [line_carry]
        if line_bounce: side_handles.append(line_bounce)
        if line_roll: side_handles.append(line_roll)
        if len(side_handles) > 0:
            ax1.legend(handles=side_handles, loc='upper left')

        # ——— Top View ———
        ax2.set(xlabel="Downrange (yd)", ylabel="Lateral (yd)", xlim=(0, xmax), ylim=(-ymax, ymax))
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        line_path, = ax2.plot(X, Y, "#1f77b4", lw=1.2, alpha=0.55, label="Full Path")
        line_carry_lat, = ax2.plot(Cx, self.interp_y_array(Cx, X, Y), "#1f77b4", lw=1.8, label="Carry")
        line_bounce_lat = None
        line_roll_lat = None
        if len(Bx):
            line_bounce_lat, = ax2.plot(Bx, self.interp_y_array(Bx, X, Y), "#2ca02c", lw=2.0, label="Bounce")
        if len(Rx):
            line_roll_lat, = ax2.plot(Rx, self.interp_y_array(Rx, X, Y), "#ff7f0e", lw=2.2, label="Roll")

        ball2, = ax2.plot([], [], 'o', ms=8, color="#1f77b4")
        trail2, = ax2.plot([], [], lw=2, color="#1f77b4", alpha=0.6)

        top_handles = [line_path, line_carry_lat]
        if line_bounce_lat: top_handles.append(line_bounce_lat)
        if line_roll_lat: top_handles.append(line_roll_lat)
        if len(top_handles) > 0:
            ax2.legend(handles=top_handles, loc='upper left')

                # Time anchors
        T_air = Ct
        T_bnc = Ct[-1] + Bt if len(Bt) else np.array([])
        T_rol = (Ct[-1] + Bt[-1] + Rt) if len(Bt) else (Ct[-1] + Rt)  # ← ROLL STARTS RIGHT AFTER BOUNCE

        def update(frame):
            t = frame * frame_dt / playback

            if t <= T_air[-1]:
                i = min(len(T_air) - 1, int((t / max(1e-9, T_air[-1])) * len(T_air)))
                ball1.set_data([Cx[i]], [Cz[i]])
                s = max(0, i - int(0.45 / max(1e-9, T_air[-1]) * len(T_air)))
                trail1.set_data(Cx[s:i+1], Cz[s:i+1])
                y_i = self.interp_y(Cx[i], X, Y)
                ball2.set_data([Cx[i]], [y_i])
                trail2.set_data(Cx[s:i+1], self.interp_y_array(Cx[s:i+1], X, Y))
                return ball1, trail1, ball2, trail2

            if len(T_bnc) and t <= T_bnc[-1]:
                tb = t - T_air[-1]
                i = min(len(Bt) - 1, int((tb / max(1e-9, Bt[-1])) * len(Bt)))
                ball1.set_data([Bx[i]], [Bz[i]])
                s = max(0, i - int(0.45 / max(1e-9, Bt[-1]) * len(Bt)))
                trail1.set_data(Bx[s:i+1], Bz[s:i+1])
                y_i = self.interp_y(Bx[i], X, Y)
                ball2.set_data([Bx[i]], [y_i])
                trail2.set_data(Bx[s:i+1], self.interp_y_array(Bx[s:i+1], X, Y))
                return ball1, trail1, ball2, trail2

            if len(T_rol):
                tr = t - (T_air[-1] + (Bt[-1] if len(Bt) else 0))
                if tr < 0 or Rt[-1] <= 0:
                    return ball1, trail1, ball2, trail2
                i = min(len(Rt) - 1, int((tr / Rt[-1]) * len(Rt)))
                ball1.set_data([Rx[i]], [Rz[i]])
                s = max(0, i - int(0.45 / max(1e-9, Rt[-1]) * len(Rt)))
                trail1.set_data(Rx[s:i+1], Rz[s:i+1])
                y_i = self.interp_y(Rx[i], X, Y)
                ball2.set_data([Rx[i]], [y_i])
                trail2.set_data(Rx[s:i+1], self.interp_y_array(Rx[s:i+1], X, Y))
                return ball1, trail1, ball2, trail2
            
            return ball1, trail1, ball2, trail2

        # Create animation
        ani = FuncAnimation(fig, update, frames=nframes, interval=60/fps, blit=False, repeat=False)
        _current_ani = ani

        plt.tight_layout()
        plt.show(block=False)  # Non-blocking
        return ani


# ————————————————————————————————————————————————————————————————
# Public function
# ————————————————————————————————————————————————————————————————
def run_plot(bd, shotNum):
    runner = PlotData(shotNum)
    paths = runner.build_paths_from_json(bd)

    m = paths["meta"]
    print(f"Carry: {m['carry_yd']:.2f} yd | Total: {m['total_yd']:.2f} yd | "
          f"Roll: {m['roll_yd']:.2f} yd | Apex: {m['peak_ft']:.2f} ft | "
          f"Bounce: {'yes' if m['did_bounce'] else 'no'} "
          f"(len {m['bounce_len_yd']:.2f} yd, apex {m['bounce_apex_ft']:.2f} ft)")
    club = bd.get("Club")
    return runner.animate_paths(paths, 1.0, club)