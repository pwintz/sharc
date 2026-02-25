#!/usr/bin/env python3
"""
SHARC Real-time Dashboard
=========================
Monitors a SHARC example directory and plots state, control, and
computation delay live as the simulation progresses.

Usage
-----
  # Auto-detect latest experiment from the PID_example folder:
  python3 dashboard.py /home/workspace/my_files/PID_example

  # Watch a specific simulation sub-directory:
  python3 dashboard.py --sim_dir /path/to/serial-with-scarab
"""

import argparse
import json
import os
import signal
import sys
import time

# Must be set before importing pyplot.
# Use TkAgg for interactive display; fall back to Agg (headless) if unavailable.
import matplotlib

def _has_display():
    """Return True if a graphical display is likely available."""
    if sys.platform == "darwin":
        return True  # macOS always has a display framework
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

_HAS_DISPLAY = _has_display()
if _HAS_DISPLAY:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        _HAS_DISPLAY = False
if not _HAS_DISPLAY:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_sim_dir(example_dir: str):
    """
    Return the simulation sub-directory of the latest experiment, or None.

    Checks for a 'latest' symlink inside *example_dir* and returns the
    most-recently-modified subdirectory inside it (e.g. 'serial-with-scarab').
    Falls back to scanning 'experiments/' if 'latest' is absent.
    """
    latest = os.path.join(example_dir, "latest")
    if os.path.exists(latest):
        try:
            subdirs = [
                os.path.join(latest, d)
                for d in os.listdir(latest)
                if os.path.isdir(os.path.join(latest, d))
            ]
            if subdirs:
                return max(subdirs, key=lambda p: os.path.getmtime(p))
        except OSError:
            pass

    # Fallback: newest experiment directory in experiments/
    exps_dir = os.path.join(example_dir, "experiments")
    if not os.path.isdir(exps_dir):
        return None
    try:
        exp_dirs = sorted(
            [os.path.join(exps_dir, d) for d in os.listdir(exps_dir)
             if os.path.isdir(os.path.join(exps_dir, d))],
            key=lambda p: os.path.getmtime(p),
        )
        if not exp_dirs:
            return None
        newest_exp = exp_dirs[-1]
        subdirs = [
            os.path.join(newest_exp, d)
            for d in os.listdir(newest_exp)
            if os.path.isdir(os.path.join(newest_exp, d))
        ]
        return max(subdirs, key=lambda p: os.path.getmtime(p)) if subdirs else None
    except OSError:
        return None


def load_data(sim_dir: str):
    """
    Load experiment_data_incremental.json from *sim_dir*.

    Both serial and parallel modes write this file:
      - plant_runner writes it per-step (live progress)
      - run_experiment_sequential / run_experiment_parallelized overwrite it
        at the end with the full experiment data (includes config, etc.)

    Returns a dict on success, None on failure.
    """
    path = os.path.join(sim_dir, "experiment_data_incremental.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            raw = json.load(fh)
        # Normalise the key: plant_runner's TimeStepSeries serialises as
        # "pending_computation" (singular) while the experiment-level code
        # uses "pending_computations" (plural).  Dashboard expects plural.
        if "pending_computation" in raw and "pending_computations" not in raw:
            raw["pending_computations"] = raw.pop("pending_computation")
        return raw
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def extract_steps(raw: dict):
    """
    From the raw incremental JSON dict, extract one sample per completed
    simulation step.

    The incremental format stores two entries per step k:
      index 2k  : state *before* step k (start of step)
      index 2k+1: state *after*  step k (end of step = plant response)

    We return the *after-step* entries (indices 1, 3, 5, ...).

    Returns
    -------
    t      : np.ndarray shape (N,)   -- time after each step
    x      : np.ndarray shape (N, 6) -- state [pos_x, pos_y, yaw, speed, wp_x, wp_y]
    u      : np.ndarray shape (N, 3) -- control [throttle, steer, brake] per u_names
    delays : np.ndarray shape (N,)   -- computation delay in seconds
    """
    k_arr  = raw.get("k", [])
    t_arr  = raw.get("t", [])
    x_arr  = raw.get("x", [])
    u_arr  = raw.get("u", [])
    pc_arr = raw.get("pending_computations", [])

    if not k_arr:
        return (np.array([]),) * 4

    # Take every other entry starting at index 1
    t_s  = np.asarray(t_arr[1::2],  dtype=float)
    x_s  = np.asarray(x_arr[1::2],  dtype=float)
    u_s  = np.asarray(u_arr[1::2],  dtype=float)
    pc_s = pc_arr[1::2] if pc_arr else []

    delays = np.asarray(
        [pc.get("delay", 0.0) if isinstance(pc, dict) else 0.0 for pc in pc_s],
        dtype=float,
    )

    # Ensure 2-D even when there is only one step
    if x_s.ndim == 1:
        x_s = x_s.reshape(1, -1)
    if u_s.ndim == 1:
        u_s = u_s.reshape(1, -1)

    return t_s, x_s, u_s, delays


# ─────────────────────────────────────────────────────────────────────────────
# Figure setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure():
    """Create the dashboard figure and return (fig, axes_dict)."""
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#1e1e2e")

    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.38,
                          left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax_xy    = fig.add_subplot(gs[:, 0])   # trajectory (tall column)
    ax_spd   = fig.add_subplot(gs[0, 1])   # speed
    ax_dly   = fig.add_subplot(gs[0, 2])   # computation delay
    ax_tbrk  = fig.add_subplot(gs[1, 1])   # throttle + brake
    ax_steer = fig.add_subplot(gs[1, 2])   # steering

    _style_ax(ax_xy,    "Vehicle Trajectory",        "X (m)",      "Y (m)")
    _style_ax(ax_spd,   "Speed",                     "Time (s)",   "Speed (km/h)")
    _style_ax(ax_dly,   "Computation Delay",         "Time (s)",   "Delay (s)")
    _style_ax(ax_tbrk,  "Throttle / Brake",          "Time (s)",   "Command [0, 1]")
    _style_ax(ax_steer, "Steering",                  "Time (s)",   "Steer [-1, 1]")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_tbrk.set_ylim(-0.05, 1.05)
    ax_steer.set_ylim(-1.1, 1.1)

    return fig, dict(xy=ax_xy, spd=ax_spd, dly=ax_dly, tbrk=ax_tbrk, steer=ax_steer)


_DARK_BG   = "#1e1e2e"
_GRID_CLR  = "#44475a"
_TEXT_CLR  = "#f8f8f2"
_LABEL_CLR = "#cdd6f4"

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(_DARK_BG)
    ax.tick_params(colors=_TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_CLR)
    ax.set_title(title, color=_TEXT_CLR, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, color=_LABEL_CLR, fontsize=8)
    ax.set_ylabel(ylabel, color=_LABEL_CLR, fontsize=8)
    ax.grid(True, color=_GRID_CLR, linewidth=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard class
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """Encapsulates dashboard state and matplotlib animation logic."""

    def __init__(self, example_dir, sim_dir_override=None, interval_ms=1000):
        self.example_dir       = os.path.abspath(example_dir)
        self.sim_dir_override  = sim_dir_override
        self.interval_ms       = interval_ms

        # State persisted across animation frames
        self._current_sim_dir = None
        self._config_cache    = {}   # parsed system_parameters
        self._status_text     = None
        self._n_steps_last    = 0

        # Build figure
        self.fig, self.axes = _build_figure()
        self.fig.suptitle("SHARC Live Dashboard", color=_TEXT_CLR,
                          fontsize=13, fontweight="bold",
                          x=0.5, y=0.97)

        self._close_requested = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _show_status(self, msg: str):
        """Display a large status message in the center of the figure."""
        self._clear_status()
        self._status_text = self.fig.text(
            0.5, 0.5, msg,
            ha="center", va="center", fontsize=13, color="#a6adc8",
            transform=self.fig.transFigure,
            wrap=True,
        )

    def _clear_status(self):
        if self._status_text is not None:
            try:
                self._status_text.remove()
            except ValueError:
                pass
            self._status_text = None

    def _update_title(self, n_steps: int, n_miss: int, sim_dir: str):
        name  = os.path.basename(sim_dir)
        T     = self._config_cache.get("sample_time", "?")
        label = self._config_cache.get("label", "")
        self.fig.suptitle(
            f"SHARC Live Dashboard  |  {label}  |  T = {T} s  "
            f"|  step {n_steps}  |  deadline misses: {n_miss}",
            color=_TEXT_CLR, fontsize=11, fontweight="bold",
            x=0.5, y=0.97,
        )

    # ── save / close ─────────────────────────────────────────────────────

    def save_final_image(self):
        """
        Save the current dashboard figure to the experiment directory.

        For serial mode the experiment directory is ``sim_dir`` itself.
        For parallel mode it is the *parent* of ``sim_dir`` (one level up,
        because ``sim_dir`` is a batch sub-directory of the experiment dir).
        """
        sim_dir = self._current_sim_dir
        if sim_dir is None:
            sim_dir = self.sim_dir_override or find_sim_dir(self.example_dir)
        if sim_dir is None:
            print("[dashboard] No simulation data found; skipping final image save.")
            return

        # Determine experiment-level directory:
        #   serial  → sim_dir is directly under experiments/ → save there
        #   parallel→ sim_dir is a subdir of the experiment dir → save to parent
        exps_path = os.path.join(self.example_dir, "experiments")
        parent    = os.path.dirname(sim_dir)
        if os.path.abspath(parent) == os.path.abspath(exps_path):
            save_dir = sim_dir   # serial
        else:
            save_dir = parent    # parallel

        save_path = os.path.join(save_dir, "dashboard_final.png")
        try:
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
            print(f"[dashboard] Final image saved \u2192 {save_path}")
        except Exception as exc:
            print(f"[dashboard] WARNING: could not save final image: {exc}")

    # ── animation callback ───────────────────────────────────────────────

    def update(self, _frame):
        # ── 0. Handle deferred shutdown (SIGTERM / SIGINT) ──────────────────
        if self._close_requested:
            plt.close(self.fig)
            return

        # ── 1. Resolve sim_dir ──────────────────────────────────────────────
        if self.sim_dir_override:
            sim_dir = self.sim_dir_override
        else:
            sim_dir = find_sim_dir(self.example_dir)

        if sim_dir is None:
            self._show_status(
                f"Waiting for experiment to start...\n({self.example_dir})"
            )
            return

        # Detect experiment change → reset config cache
        if sim_dir != self._current_sim_dir:
            print(f"[dashboard] Switched to: {sim_dir}")
            self._current_sim_dir = sim_dir
            self._config_cache    = {}
            self._n_steps_last    = 0

        # ── 2. Load data ────────────────────────────────────────────────────
        raw = load_data(sim_dir)
        if raw is None:
            self._show_status(f"Waiting for data...\n{sim_dir}")
            return

        # ── 3. Cache config ─────────────────────────────────────────────────
        if not self._config_cache:
            cfg = raw.get("config", {})
            # The per-step incremental file written by plant_runner
            # doesn't embed config.  Fall back to config.json written by
            # SHARC into the sim directory.
            if not cfg:
                cfg_path = os.path.join(sim_dir, "config.json")
                if os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, "r") as fh:
                            cfg = json.load(fh)
                    except (json.JSONDecodeError, OSError):
                        cfg = {}
            sp  = cfg.get("system_parameters", {})
            self._config_cache = {
                "sample_time": sp.get("sample_time", 0.1),
                "x_names":     sp.get("x_names", ["x","y","yaw","speed","wp_x","wp_y"]),
                "u_names":     sp.get("u_names", ["throttle","steer","brake"]),
                "target_speed":sp.get("target_speed", None),
                "label":       cfg.get("label", ""),
            }

        # ── 4. Extract per-step data ─────────────────────────────────────────
        t, x, u, delays = extract_steps(raw)
        n_steps = len(t)

        if n_steps == 0:
            self._show_status("Simulation initialising...")
            return

        self._clear_status()

        # Only redraw if new steps arrived
        if n_steps == self._n_steps_last:
            return
        self._n_steps_last = n_steps

        # ── 5. Unpack arrays ─────────────────────────────────────────────────
        T         = float(self._config_cache["sample_time"])
        u_names   = self._config_cache["u_names"]
        tgt_spd   = self._config_cache["target_speed"]

        # State columns: [pos_x, pos_y, yaw, speed, wp_x, wp_y]
        px    = x[:, 0]; py    = x[:, 1]
        speed = x[:, 3]
        wp_x  = x[:, 4] if x.shape[1] > 4 else None
        wp_y  = x[:, 5] if x.shape[1] > 5 else None

        # Control columns resolved by u_names
        def _col(name, fallback):
            try:
                return u[:, u_names.index(name)]
            except (ValueError, IndexError):
                return u[:, fallback] if u.shape[1] > fallback else np.zeros(n_steps)

        throttle = _col("throttle", 0)
        steer    = _col("steer",    1)
        brake    = _col("brake",    2)

        # Deadline misses
        miss_mask = (delays >= T) if len(delays) else np.zeros(n_steps, bool)
        n_miss    = int(miss_mask.sum())

        ax = self.axes

        # ── 6. XY Trajectory ───────────────────────────────────────────────
        ax["xy"].cla()
        _style_ax(ax["xy"], "Vehicle Trajectory", "X (m)", "Y (m)")
        ax["xy"].set_aspect("equal", adjustable="datalim")
        ax["xy"].plot(px, py, color="#89b4fa", linewidth=1.5, label="Path")
        ax["xy"].plot(px[-1], py[-1], "o", color="#f38ba8",
                      markersize=7, label="Current")
        if wp_x is not None and wp_y is not None:
            ax["xy"].scatter(wp_x, wp_y, c="#a6e3a1", s=10,
                             alpha=0.4, zorder=3, label="Waypoints")
        ax["xy"].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR)

        # ── 7. Speed ────────────────────────────────────────────────────────
        ax["spd"].cla()
        _style_ax(ax["spd"], "Speed", "Time (s)", "Speed (km/h)")
        ax["spd"].plot(t, speed, color="#89dceb", linewidth=1.5)
        if tgt_spd is not None:
            ax["spd"].axhline(tgt_spd, color="#f38ba8", linewidth=1.2,
                               linestyle="--", label=f"Target {tgt_spd} km/h")
            ax["spd"].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR)

        # ── 8. Computation Delay ────────────────────────────────────────────
        ax["dly"].cla()
        _style_ax(ax["dly"], "Computation Delay", "Time (s)", "Delay (s)")
        if len(delays):
            t_d = t[:len(delays)]
            ax["dly"].plot(t_d, delays, color="#cba6f7",
                           linewidth=1.2, marker=".", markersize=3,
                           label="Delay")
            ax["dly"].axhline(T, color="#f38ba8", linewidth=1.5,
                               linestyle="--", label=f"T = {T} s")
            if miss_mask.any():
                ax["dly"].scatter(t_d[miss_mask], delays[miss_mask],
                                   color="#f38ba8", s=40, zorder=5,
                                   label="Deadline miss")
            ax["dly"].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR)

        # ── 9. Throttle / Brake ─────────────────────────────────────────────
        ax["tbrk"].cla()
        _style_ax(ax["tbrk"], "Throttle / Brake", "Time (s)", "Command [0, 1]")
        ax["tbrk"].set_ylim(-0.05, 1.05)
        ax["tbrk"].plot(t, throttle, color="#a6e3a1",
                         linewidth=1.5, label="Throttle")
        ax["tbrk"].plot(t, brake,    color="#f38ba8",
                         linewidth=1.5, label="Brake")
        ax["tbrk"].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR)

        # ── 10. Steering ────────────────────────────────────────────────────
        ax["steer"].cla()
        _style_ax(ax["steer"], "Steering", "Time (s)", "Steer [-1, 1]")
        ax["steer"].set_ylim(-1.1, 1.1)
        ax["steer"].plot(t, steer, color="#fab387", linewidth=1.5)
        ax["steer"].axhline(0, color=_GRID_CLR, linewidth=0.8)

        # ── 11. Title ────────────────────────────────────────────────────────
        self._update_title(n_steps, n_miss, sim_dir)

        self.fig.canvas.draw_idle()

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        print(f"[dashboard] Watching: {self.example_dir}")
        print("[dashboard] Close the window to exit.")

        # SIGTERM is sent by SHARC's _stop_dashboard() when all experiments
        # finish.  Set the flag; the animation loop will close the window
        # cleanly on the next tick so we can save before exiting.
        def _on_term(signum, frame):
            self._close_requested = True

        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT,  _on_term)

        self._anim = FuncAnimation(
            self.fig,
            self.update,
            interval=self.interval_ms,
            cache_frame_data=False,
        )
        plt.show()
        # plt.show() returns when the window has been closed (either by the
        # user or by the SIGTERM handler above via plt.close).  Save now.
        self.save_final_image()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SHARC real-time dashboard")
    parser.add_argument(
        "example_dir",
        nargs="?",
        default=".",
        help="Path to the SHARC example directory (default: current directory)",
    )
    parser.add_argument(
        "--sim_dir",
        default=None,
        help="Watch a specific simulation sub-directory instead of auto-detecting.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Plot refresh interval in milliseconds (default: 1000)",
    )
    args = parser.parse_args()

    example_dir = os.path.abspath(args.example_dir)
    if not os.path.isdir(example_dir):
        print(f"[dashboard] ERROR: directory not found: {example_dir}", file=sys.stderr)
        sys.exit(1)

    if not _HAS_DISPLAY:
        print("[dashboard] No display available (headless). Dashboard disabled.")
        sys.exit(0)

    # Auto-detect if running from within the example directory
    # (no 'latest' link there but there may be an 'experiments' folder)
    if args.sim_dir is None:
        detected = find_sim_dir(example_dir)
        if detected:
            print(f"[dashboard] Found: {detected}")
        else:
            print(f"[dashboard] Auto-detecting latest experiment in: {example_dir}")

    dash = Dashboard(
        example_dir=example_dir,
        sim_dir_override=args.sim_dir,
        interval_ms=args.interval,
    )
    dash.run()


if __name__ == "__main__":
    main()
