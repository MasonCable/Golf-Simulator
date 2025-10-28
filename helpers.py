import argparse

SURFACES = {
    # mu_r (rolling friction), k_quad (quadratic drag), e_n (restitution), mu_t (tangential friction)
    "rough":   (0.060, 0.0006, 0.20, 0.50),
    "fairway": (0.030, 0.0003, 0.40, 0.30),
    "firm":    (0.020, 0.0002, 0.50, 0.25),
    "green":   (0.015, 0.00015, 0.25, 0.20),
}

def handle_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to JSON with BallData (GSPro-like).")
    ap.add_argument("--surface", default="fairway", choices=list(SURFACES.keys()), help="Roll surface.")
    ap.add_argument("--playback", type=float, default=1.0, help="Playback speed (1.0 real-time).")
    ap.add_argument("--mu_r", type=float, default=None, help="Rolling resistance coefficient override.")
    ap.add_argument("--k_quad", type=float, default=None, help="Quadratic roll drag coefficient override.")
    ap.add_argument("--shot", type=str, default="driver", help="Driver, wedge, i9-i1")
    return ap.parse_args()