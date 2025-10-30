# main.py
import asyncio
import signal
import sys
from sim_data_steam import run_simulate_data_stream
from handle_data_steam import shot_stream
from plot_data import run_plot
from helpers import handle_arguments
import matplotlib.pyplot as plt
_sim_instance = None

def _cleanup(*_):
    global _sim_instance
    if _sim_instance:
        print("\nShutting down simulator...")
        _sim_instance.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

async def main():
    global _sim_instance
    args = handle_arguments()
    sim = run_simulate_data_stream(args.json, delay=10.0)
    if not sim: return
    _sim_instance = sim
    print("Simulator started – waiting for shots...")
    shotNum = 0
    try:
        async for shot in shot_stream():
            shotNum += 1
            run_plot(shot, shotNum)  # ← Auto-closes old, opens new            
            await asyncio.sleep(0)
            fig = plt.gcf()
            if fig:
                fig.canvas.flush_events()  # ← Force update
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
        print("Done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled in main()