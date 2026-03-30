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
import math
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


def _speed_state_to_kph_scale(cfg: dict) -> float:
    dynamics_name = cfg.get("dynamics_class_name", "")
    controller_type = cfg.get("system_parameters", {}).get("controller_type", "")
    if dynamics_name == "CarlaMPCDynamics" or controller_type == "CarCarlaMPC":
        return 3.6
    return 1.0


def _target_speed_kph_from_config(cfg: dict):
    controller_params = cfg.get("controller_parameters", {})
    if "TerminalVelocity" in controller_params:
        target = float(controller_params.get("TerminalVelocity"))
        units = str(controller_params.get("TerminalVelocityUnits", "mps")).lower()
        if units in ("mps", "m/s"):
            return target * 3.6
        return target

    system_params = cfg.get("system_parameters", {})
    target = system_params.get("target_speed", None)
    if target is None:
        return None
    units = str(system_params.get("target_speed_units", "kmph")).lower()
    if units in ("mps", "m/s"):
        return float(target) * 3.6
    return float(target)


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
    w      : np.ndarray shape (N, D) -- exogenous input (waypoints + obstacles)
    """
    k_arr  = raw.get("k", [])
    t_arr  = raw.get("t", [])
    x_arr  = raw.get("x", [])
    u_arr  = raw.get("u", [])
    w_arr  = raw.get("w", [])
    pc_arr = raw.get("pending_computations", [])

    if not k_arr:
        return (np.array([]),) * 5

    # Take every other entry starting at index 1
    t_s  = np.asarray(t_arr[1::2],  dtype=float)
    x_s  = np.asarray(x_arr[1::2],  dtype=float)
    u_s  = np.asarray(u_arr[1::2],  dtype=float)
    w_s  = np.asarray(w_arr[1::2],  dtype=float) if w_arr else np.array([])
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
    if w_s.ndim == 1 and w_s.size > 0:
        w_s = w_s.reshape(1, -1)

    return t_s, x_s, u_s, delays, w_s


def load_carla_extra(sim_dir: str):
    """
    Aggregate NPC trajectory data and collision events from all
    ``carla_extra.jsonl`` sidecar files produced by CarlaMPCDynamics.

    Looks in *sim_dir* itself and in any immediate sub-directories
    (batch dirs), so both serial and parallel modes are handled.

    Returns
    -------
    npc_tracks : dict  id → {"label": str, "xs": list, "ys": list}
    collisions  : list of dicts  {"t", "ego_x", "ego_y", "other", "intensity"}
    """
    # Gather candidate files (sim_dir first, then sub-dirs sorted
    # *numerically* by batch index so batches appear in chronological order).
    # Lexicographic sort puts batch10 before batch1 — we need batch0, batch1, ..., batch10.
    import re as _re
    def _batch_sort_key(name):
        m = _re.search(r'batch(\d+)', name)
        return int(m.group(1)) if m else float('inf')

    files = []
    top = os.path.join(sim_dir, 'carla_extra.jsonl')
    if os.path.isfile(top):
        files.append(top)
    try:
        for entry in sorted(os.listdir(sim_dir), key=_batch_sort_key):
            sub = os.path.join(sim_dir, entry, 'carla_extra.jsonl')
            if os.path.isfile(sub):
                files.append(sub)
    except OSError:
        pass

    npc_tracks = {}   # actor_id  → {"label", "xs", "ys"}
    collisions  = []
    npc_counter = 0

    for fpath in files:
        try:
            with open(fpath, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for npc in rec.get('npcs', []):
                        nid = npc['id']
                        if nid not in npc_tracks:
                            npc_counter += 1
                            npc_tracks[nid] = {
                                'label': f'NPC {npc_counter}',
                                'xs': [], 'ys': [],
                            }
                        npc_tracks[nid]['xs'].append(npc['x'])
                        npc_tracks[nid]['ys'].append(npc['y'])
                    col = rec.get('collision')
                    if col:
                        collisions.append({
                            't':         rec.get('t', 0),
                            'ego_x':     rec.get('ego_x', 0),
                            'ego_y':     rec.get('ego_y', 0),
                            'other':     col.get('other', 'unknown'),
                            'intensity': col.get('intensity', 0),
                        })
        except OSError:
            pass

    return npc_tracks, collisions


# NPC colour palette — warm/vivid tones that stand out on the dark background
_NPC_COLORS = [
    '#fab387',  # peach
    '#f9e2af',  # yellow
    '#a6e3a1',  # green
    '#94e2d5',  # teal
    '#74c7ec',  # sky
    '#cba6f7',  # mauve
    '#f2cdcd',  # flamingo
    '#eba0ac',  # maroon
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure():
    """Create the dashboard figure and return (fig, axes_dict)."""
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#1e1e2e")

    gs = fig.add_gridspec(2, 4, hspace=0.50, wspace=0.38,
                          left=0.05, right=0.97, top=0.91, bottom=0.09,
                          width_ratios=[1.2, 1, 1, 0.8])

    ax_xy    = fig.add_subplot(gs[:, 0])   # trajectory (tall column)
    ax_spd   = fig.add_subplot(gs[0, 1])   # speed
    ax_dly   = fig.add_subplot(gs[0, 2])   # computation delay
    ax_tbrk  = fig.add_subplot(gs[1, 1])   # throttle + brake
    ax_steer = fig.add_subplot(gs[1, 2])   # steering
    ax_met   = fig.add_subplot(gs[:, 3])   # metrics panel (tall column)

    _style_ax(ax_xy,    "Vehicle Trajectory",        "X (m)",      "Y (m)")
    _style_ax(ax_spd,   "Speed",                     "Time (s)",   "Speed (km/h)")
    _style_ax(ax_dly,   "Computation Delay",         "Time (s)",   "Delay (s)")
    _style_ax(ax_tbrk,  "Throttle / Brake",          "Time (s)",   "Command [0, 1]")
    _style_ax(ax_steer, "Steering",                  "Time (s)",   "Steer [-1, 1]")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_tbrk.set_ylim(-0.05, 1.05)
    ax_steer.set_ylim(-1.1, 1.1)

    # Metrics panel: text-only, no axes
    ax_met.set_facecolor(_DARK_BG)
    ax_met.set_title("Metrics", color=_TEXT_CLR, fontsize=9, fontweight="bold")
    ax_met.set_xticks([])
    ax_met.set_yticks([])
    for spine in ax_met.spines.values():
        spine.set_edgecolor(_GRID_CLR)

    return fig, dict(xy=ax_xy, spd=ax_spd, dly=ax_dly, tbrk=ax_tbrk, steer=ax_steer,
                     met=ax_met)


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

    def _update_title(self, n_steps: int, n_miss: int, n_collisions: int, sim_dir: str):
        name  = os.path.basename(sim_dir)
        T     = self._config_cache.get('sample_time', '?')
        label = self._config_cache.get('label', '')
        collision_tag = f'  \u26a0 {n_collisions} COLLISION(S)' if n_collisions else ''
        self.fig.suptitle(
            f'SHARC Live Dashboard  |  {label}  |  T = {T} s  '
            f'|  step {n_steps}  |  deadline misses: {n_miss}'
            f'{collision_tag}',
            color='#f38ba8' if n_collisions else _TEXT_CLR,
            fontsize=11, fontweight='bold',
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
            mpc_o = sp.get("mpc_options", {})
            lims  = mpc_o.get("input_limits", {})
            self._config_cache = {
                "sample_time": sp.get("sample_time", 0.1),
                "x_names":     sp.get("x_names", ["x","y","yaw","speed","wp_x","wp_y"]),
                "u_names":     sp.get("u_names", ["throttle","steer","brake"]),
                "target_speed":_target_speed_kph_from_config(cfg),
                "speed_to_kph": _speed_state_to_kph_scale(cfg),
                "label":       cfg.get("label", ""),
                "mpc_opts": {
                    "max_accel": lims.get("max_accel",  1.0),
                    "min_accel": lims.get("min_accel", -3.0),
                    "n_waypoints": mpc_o.get("n_waypoints", 0),
                    "n_obstacles": mpc_o.get("n_obstacles", 0),
                },
            }

        # ── 4. Extract per-step data ─────────────────────────────────────────
        t, x, u, delays, w_arr = extract_steps(raw)
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
        speed_to_kph = float(self._config_cache.get("speed_to_kph", 1.0))

        # State columns: [pos_x, pos_y, yaw, speed, wp_x, wp_y]
        px    = x[:, 0]; py    = x[:, 1]
        speed = x[:, 3] * speed_to_kph
        wp_x  = x[:, 4] if x.shape[1] > 4 else None
        wp_y  = x[:, 5] if x.shape[1] > 5 else None

        # Control columns resolved by u_names.
        # Supports two conventions:
        #   (a) PID-style:  u = [throttle, steer, brake]     (CARLA commands, [0,1])
        #   (b) MPC-style:  u = [acceleration, steering_angle] (physical units)
        #       → throttle = max(a, 0) / max_accel,  brake = max(-a, 0) / |min_accel|
        def _col(name, fallback):
            try:
                return u[:, u_names.index(name)]
            except (ValueError, IndexError):
                return u[:, fallback] if u.shape[1] > fallback else np.zeros(n_steps)

        mpc_opts   = (self._config_cache.get("mpc_opts") or {})
        _is_mpc    = ("acceleration" in u_names or "steering_angle" in u_names)

        if _is_mpc:
            accel = _col("acceleration",   0)
            delta = _col("steering_angle", 1)
            _max_a = mpc_opts.get("max_accel", 1.0)
            _min_a = mpc_opts.get("min_accel", -3.0)
            throttle = np.clip( accel / max(_max_a,  1e-6), 0.0, 1.0)
            brake    = np.clip(-accel / max(-_min_a, 1e-6), 0.0, 1.0)
            steer    = np.clip(delta, -1.0, 1.0)
        else:
            throttle = _col("throttle", 0)
            steer    = _col("steer",    1)
            brake    = _col("brake",    2)

        # Deadline misses
        miss_mask = (delays >= T) if len(delays) else np.zeros(n_steps, bool)
        n_miss    = int(miss_mask.sum())

        ax = self.axes

        # ── 6. XY Trajectory ───────────────────────────────────────────────
        npc_tracks, collisions = load_carla_extra(sim_dir)
        n_collisions = len(collisions)

        ax['xy'].cla()
        title_color = '#f38ba8' if n_collisions else _TEXT_CLR
        title_text  = ('Vehicle Trajectory'
                       if not n_collisions
                       else f'Vehicle Trajectory  \u26a0 {n_collisions} COLLISION(S)')
        _style_ax(ax['xy'], title_text, 'X (m)', 'Y (m)')
        ax['xy'].title.set_color(title_color)
        ax['xy'].set_aspect('equal', adjustable='datalim')

        # Ego path — coloured gradient (older = dimmer, newest = bright)
        if len(px) > 1:
            from matplotlib.collections import LineCollection
            points  = np.array([px, py]).T.reshape(-1, 1, 2)
            segs    = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas  = np.linspace(0.15, 1.0, len(segs))
            for seg, a in zip(segs, alphas):
                ax['xy'].plot(seg[:, 0], seg[:, 1],
                              color='#89b4fa', linewidth=2.0, alpha=float(a),
                              solid_capstyle='round')
        elif len(px) == 1:
            ax['xy'].plot(px, py, 'o', color='#89b4fa', markersize=5)

        # Waypoints from w array (MPC mode) or state columns (PID mode)
        n_wp  = mpc_opts.get("n_waypoints", 0)
        n_obs = mpc_opts.get("n_obstacles", 0)
        if n_wp > 0 and w_arr.size > 0 and w_arr.shape[1] >= 2 * n_wp:
            # Show waypoints for the LATEST step (current reference path)
            latest_w = w_arr[-1]
            wp_xs = latest_w[0:2*n_wp:2]
            wp_ys = latest_w[1:2*n_wp:2]
            ax['xy'].scatter(wp_xs, wp_ys, c='#a6e3a1', s=12,
                             alpha=0.6, zorder=7, marker='.', label='Waypoints')
            # Draw waypoint path as a thin green line
            ax['xy'].plot(wp_xs, wp_ys, color='#a6e3a1', linewidth=0.8,
                          alpha=0.4, zorder=6)
        elif wp_x is not None and wp_y is not None:
            ax['xy'].scatter(wp_x, wp_y, c='#a6e3a1', s=8,
                             alpha=0.35, zorder=3, label='Waypoints')

        # Obstacles from w array (MPC mode)
        if n_obs > 0 and n_wp > 0 and w_arr.size > 0:
            latest_w = w_arr[-1]
            offset = 2 * n_wp
            for oi in range(n_obs):
                base = offset + 5 * oi
                if base + 4 < len(latest_w):
                    ox = latest_w[base + 0]
                    oy = latest_w[base + 1]
                    orad = latest_w[base + 4]
                    if ox > 1e5:
                        continue  # sentinel — empty slot
                    circle = plt.Circle((ox, oy), orad, color='#fab387',
                                        fill=False, linewidth=1.5, alpha=0.8,
                                        zorder=8)
                    ax['xy'].add_patch(circle)
                    ax['xy'].plot(ox, oy, 'o', color='#fab387', markersize=4,
                                  alpha=0.9, zorder=9)
            # Add a single legend entry for obstacles
            ax['xy'].plot([], [], 'o', color='#fab387', markersize=4,
                          label='Obstacles')

        # NPC trajectories (with segment-length filter to prevent long jumps)
        _MAX_NPC_JUMP = 15.0  # max plausible single-step NPC displacement (m)
        for i, (nid, track) in enumerate(npc_tracks.items()):
            clr = _NPC_COLORS[i % len(_NPC_COLORS)]
            xs, ys = track['xs'], track['ys']
            if len(xs) > 1:
                # Draw segments individually, skipping impossibly long jumps
                for j in range(len(xs) - 1):
                    dx = xs[j+1] - xs[j]
                    dy = ys[j+1] - ys[j]
                    if dx*dx + dy*dy > _MAX_NPC_JUMP * _MAX_NPC_JUMP:
                        continue  # skip teleport artifact
                    ax['xy'].plot([xs[j], xs[j+1]], [ys[j], ys[j+1]],
                                  color=clr, linewidth=1.0,
                                  linestyle='--', alpha=0.55, zorder=4)
            if xs:
                ax['xy'].plot(xs[-1], ys[-1], 's', color=clr,
                              markersize=5, alpha=0.9, zorder=5,
                              label=track['label'])

        # Collision markers
        for col in collisions:
            ax['xy'].plot(col['ego_x'], col['ego_y'],
                          'X', color='#ff2222', markersize=13,
                          markeredgewidth=2.0, zorder=10)
            ax['xy'].annotate(
                f"\u26a0 COLLISION\n"
                f"{col['other'].split('.')[-1][:16]}\n"
                f"{col['intensity']:.0f} N·m/s",
                xy=(col['ego_x'], col['ego_y']),
                xytext=(8, 8), textcoords='offset points',
                color='#ff4444', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#2a0000', alpha=0.85,
                          edgecolor='#ff4444', linewidth=0.8),
                zorder=11,
            )

        # Ego current position (on top of everything)
        ax['xy'].plot(px[-1], py[-1], 'o', color='#f38ba8',
                      markersize=8, zorder=12, label='Ego')

        ax['xy'].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR,
                        framealpha=0.8, edgecolor=_GRID_CLR)

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
        if _is_mpc:
            _style_ax(ax["tbrk"], "Throttle / Brake (derived)", "Time (s)", "Normalised [0, 1]")
        else:
            _style_ax(ax["tbrk"], "Throttle / Brake", "Time (s)", "Command [0, 1]")
        ax["tbrk"].set_ylim(-0.05, 1.05)
        ax["tbrk"].plot(t, throttle, color="#a6e3a1",
                         linewidth=1.5, label="Throttle")
        ax["tbrk"].plot(t, brake,    color="#f38ba8",
                         linewidth=1.5, label="Brake")
        ax["tbrk"].legend(fontsize=7, facecolor=_DARK_BG, labelcolor=_TEXT_CLR)

        # ── 10. Steering ────────────────────────────────────────────────────
        ax["steer"].cla()
        if _is_mpc:
            _style_ax(ax["steer"], "Steering Angle", "Time (s)", "δ (rad)")
            ax["steer"].set_ylim(-0.55, 0.55)
        else:
            _style_ax(ax["steer"], "Steering", "Time (s)", "Steer [-1, 1]")
            ax["steer"].set_ylim(-1.1, 1.1)
        ax["steer"].plot(t, steer, color="#fab387", linewidth=1.5)
        ax["steer"].axhline(0, color=_GRID_CLR, linewidth=0.8)

        # ── 11. Metrics panel ────────────────────────────────────────────────
        ax["met"].cla()
        ax["met"].set_facecolor(_DARK_BG)
        ax["met"].set_title("Metrics", color=_TEXT_CLR, fontsize=9, fontweight="bold")
        ax["met"].set_xticks([])
        ax["met"].set_yticks([])
        for spine in ax["met"].spines.values():
            spine.set_edgecolor(_GRID_CLR)
        ax["met"].set_xlim(0, 1)
        ax["met"].set_ylim(0, 1)

        # Compute live metrics from current data
        # -- MPC cost from pending_computations metadata
        pc_arr = raw.get("pending_computations", [])
        pc_step = pc_arr[1::2] if pc_arr else []
        mpc_costs = []
        feas_count, solve_count = 0, 0
        seen_ks = set()
        for pc in pc_step:
            if not isinstance(pc, dict):
                continue
            meta = pc.get("metadata", {})
            if not meta:
                continue
            mk = meta.get("k")
            if mk is not None and mk in seen_ks:
                continue
            if mk is not None:
                seen_ks.add(mk)
            c = meta.get("cost")
            if c is not None:
                mpc_costs.append(c)
            if "is_feasible" in meta:
                solve_count += 1
                if meta["is_feasible"]:
                    feas_count += 1

        # -- Path tracking RMSE
        path_errs = []
        if w_arr.size > 0 and n_wp > 0:
            for si in range(min(n_steps, len(w_arr))):
                ew = w_arr[si]
                wpx = ew[0:2*n_wp:2]
                wpy = ew[1:2*n_wp:2]
                d2 = (wpx - px[si])**2 + (wpy - py[si])**2
                path_errs.append(float(np.sqrt(np.min(d2))))

        # -- Speed RMSE
        spd_rmse = None
        if tgt_spd is not None:
            spd_rmse = float(np.sqrt(np.mean((speed - tgt_spd)**2)))

        # -- Min obstacle distance from NPC tracks vs ego position
        min_obs = float('inf')
        for _, track in npc_tracks.items():
            for j in range(min(len(track['xs']), n_steps)):
                d = math.sqrt((track['xs'][j] - float(px[j]))**2 +
                              (track['ys'][j] - float(py[j]))**2)
                min_obs = min(min_obs, d)

        # -- Total distance
        _total_dist = None
        if len(px) > 1:
            _total_dist = float(np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2)))

        # Build metrics text lines
        lines = []
        if n_collisions:
            lines.append(("Collision", f"⚠ {n_collisions}", "#f38ba8"))
        else:
            lines.append(("Collision", "✓ None", "#a6e3a1"))

        lines.append(("Steps", f"{n_steps}", _TEXT_CLR))

        if mpc_costs:
            lines.append(("", "", ""))  # spacer
            lines.append(("Avg MPC Cost", f"{np.mean(mpc_costs):.1f}", "#89dceb"))
            lines.append(("Min / Max", f"{np.min(mpc_costs):.1f} / {np.max(mpc_costs):.1f}", "#89dceb"))
        if solve_count > 0:
            frate = feas_count / solve_count * 100
            clr = "#a6e3a1" if frate >= 95 else "#fab387" if frate >= 80 else "#f38ba8"
            lines.append(("Feasibility", f"{frate:.0f}%", clr))

        if path_errs:
            lines.append(("", "", ""))  # spacer
            prmse = float(np.sqrt(np.mean(np.array(path_errs)**2)))
            lines.append(("Path RMSE", f"{prmse:.3f} m", "#cba6f7"))
            lines.append(("Path Max Err", f"{max(path_errs):.3f} m", "#cba6f7"))
        if spd_rmse is not None:
            lines.append(("Speed RMSE", f"{spd_rmse:.1f} km/h", "#cba6f7"))

        if min_obs < float('inf'):
            lines.append(("", "", ""))  # spacer
            clr = "#f38ba8" if min_obs < 3.0 else "#fab387" if min_obs < 5.0 else "#a6e3a1"
            lines.append(("Min NPC Dist", f"{min_obs:.1f} m", clr))
        if _total_dist is not None:
            lines.append(("Total Dist", f"{_total_dist:.1f} m", _TEXT_CLR))
        if len(speed) > 0:
            lines.append(("Final Speed", f"{speed[-1]:.1f} km/h", _TEXT_CLR))
        if len(delays):
            lines.append(("Avg Delay", f"{np.mean(delays)*1000:.0f} ms", _TEXT_CLR))

        # Render text lines
        y_pos = 0.92
        for label, value, color in lines:
            if label == "" and value == "":
                y_pos -= 0.025  # spacer
                continue
            ax["met"].text(0.05, y_pos, label, fontsize=8, color="#a6adc8",
                           va="top", transform=ax["met"].transAxes,
                           fontfamily="monospace")
            ax["met"].text(0.95, y_pos, value, fontsize=8, color=color,
                           va="top", ha="right", transform=ax["met"].transAxes,
                           fontfamily="monospace", fontweight="bold")
            y_pos -= 0.055

        # ── 12. Title ────────────────────────────────────────────────────────
        self._update_title(n_steps, n_miss, n_collisions, sim_dir)

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
    parser.add_argument(
        "--save",
        action="store_true",
        help="Render the final dashboard and save it to the experiment directory "
             "without opening a window (works headless).",
    )
    args = parser.parse_args()

    example_dir = os.path.abspath(args.example_dir)
    if not os.path.isdir(example_dir):
        print(f"[dashboard] ERROR: directory not found: {example_dir}", file=sys.stderr)
        sys.exit(1)

    if not _HAS_DISPLAY and not args.save:
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

    if args.save:
        dash.update(0)          # render one frame from the completed experiment
        dash.save_final_image()
    else:
        dash.run()


if __name__ == "__main__":
    main()
