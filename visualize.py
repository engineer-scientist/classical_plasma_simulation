#!/usr/bin/env python3
# ============================================================================
#  visualize.py  --  build an animation of the plasma simulation output
# ============================================================================
#
#  Reads the files produced by ./plasma_sim, whose names carry the simulation
#  parameters as a suffix, e.g. for a run with N=500, Lx=1mm, ...:
#     plasma_meta_N500_Lx1.0e-3_..._Bz1.0.txt   - run metadata (N, NP, box, ...)
#     plasma_traj_N500_Lx1.0e-3_..._Bz1.0.bin   - float32 positions [nframes,NP,3]
#
#  It produces two things, both saved next to the input with the *same*
#  parameter suffix:
#
#   1. A 3D scatter animation of all N electrons (blue) and N protons (red)
#      bouncing around the box.  Its caption shows the magnetic field, the
#      temperature and the (calculated) plasma pressure.
#         plasma_animation_N500_..._Bz1.0.gif   (always; via Pillow)
#         plasma_animation_N500_..._Bz1.0.mp4   (only if ffmpeg is installed)
#
#   2. A four-panel diagnostics figure (needs plasma_energy_<suffix>.csv):
#         - energy conservation:  KE, PE and total energy vs time
#         - kinetic temperature vs time (with the set-point)
#         - virial pressure vs time (with the ideal-gas line n k_B T)
#         - Larmor orbits: xy tracks of a few sample electrons and protons
#         plasma_plots_N500_..._Bz1.0.png
#
#  Usage:   python3 visualize.py [plasma_meta_<suffix>.txt]
#
#  With no argument the script auto-discovers the meta file in the output/
#  directory (it errors and lists them if there is more than one, so you can
#  pick).  The trajectory and animation names are derived from the meta name,
#  so the suffix never has to be typed twice.  Everything is written back into
#  the same folder as the meta file (i.e. output/).
#
#  The script is headless (Agg backend) so it runs fine on a compute node with
#  no display.
# ----------------------------------------------------------------------------

import os
import sys
import glob
import shutil
import numpy as np

import matplotlib
matplotlib.use("Agg")                      # headless / no X server needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  (registers 3d proj.)

KB = 1.380649e-23        # Boltzmann constant [J/K] (matches plasma_sim.c)

# All simulation output lives here (matches plasma_sim_serial.c /
# plasma_sim_parallel.c); override with PLASMA_OUTDIR to point elsewhere.
OUTDIR = os.environ.get("PLASMA_OUTDIR", "output")

def resolve_names(argv):
    """Locate the meta file (given on the command line or auto-discovered) and
    derive every matching output name from its parameter suffix.

    Returns a dict with keys: meta, traj, energy, gif, mp4, plots.
    """
    if len(argv) > 1:
        meta_path = argv[1]
        if not os.path.exists(meta_path):
            sys.exit("ERROR: meta file not found: %s" % meta_path)
    else:
        candidates = sorted(glob.glob(os.path.join(OUTDIR, "plasma_meta*.txt")))
        if not candidates:
            sys.exit("ERROR: no plasma_meta*.txt in %s/ -- run "
                     "./plasma_sim_serial (or ./plasma_sim_parallel) first"
                     % OUTDIR)
        if len(candidates) > 1:
            lines = ["ERROR: multiple meta files found -- pass one explicitly:"]
            lines += ["    python3 visualize.py %s" % c for c in candidates]
            sys.exit("\n".join(lines))
        meta_path = candidates[0]

    # The suffix is everything between the "plasma_meta" prefix and ".txt".
    # Reusing it verbatim keeps the trajectory / animation names in lock-step
    # with the meta file (no need to re-derive it from the parameters).
    folder = os.path.dirname(meta_path)
    base = os.path.basename(meta_path)
    prefix, ext = "plasma_meta", ".txt"
    if not (base.startswith(prefix) and base.endswith(ext)):
        sys.exit("ERROR: %r is not a plasma_meta*.txt file" % base)
    suffix = base[len(prefix):len(base) - len(ext)]   # e.g. "_N500_Lx1.0e-3_..."

    def named(stem, ext):
        return os.path.join(folder, stem + suffix + ext)

    return {
        "meta":   meta_path,
        "traj":   named("plasma_traj",      ".bin"),
        "energy": named("plasma_energy",    ".csv"),
        "gif":    named("plasma_animation", ".gif"),
        "mp4":    named("plasma_animation", ".mp4"),
        "plots":  named("plasma_plots",     ".png"),
    }


def read_meta(path):
    """Parse the key=value metadata file into a dict of the right types."""
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k] = v
    out = {
        "N":       int(meta["N"]),
        "NP":      int(meta["NP"]),
        "nframes": int(meta["nframes"]),
        "dt":      float(meta["dt"]),
        "save_stride": int(meta["save_stride"]),
        "Lx":      float(meta["Lx"]),
        "Ly":      float(meta["Ly"]),
        "Lz":      float(meta["Lz"]),
        "Bx":      float(meta["Bx"]),
        "By":      float(meta["By"]),
        "Bz":      float(meta["Bz"]),
        # External E field: .get() with a 0.0 fallback so meta files written
        # before the E-field feature (no Ex/Ey/Ez keys) still load.
        "Ex":      float(meta.get("Ex", 0.0)),
        "Ey":      float(meta.get("Ey", 0.0)),
        "Ez":      float(meta.get("Ez", 0.0)),
        "temperature": float(meta["temperature"]),
    }
    # Mean pressure is calculated by plasma_sim and stored in the meta file.
    # Fall back to the ideal-gas estimate P = n k_B T for older meta files.
    volume = out["Lx"] * out["Ly"] * out["Lz"]
    if "pressure" in meta:
        out["pressure"] = float(meta["pressure"])
    else:
        out["pressure"] = out["NP"] / volume * KB * out["temperature"]
    out["volume"] = float(meta.get("volume", volume))
    return out


def read_energy(path):
    """Read plasma_energy_<suffix>.csv into a dict of 1-D arrays.

    Columns: step, time_s, KE_J, PE_J, Etot_J, T_kin_K, P_Pa[, Uext_J].  The
    trailing Uext_J (external-field potential energy) is only present in newer
    runs and is read when available.  Returns None if the file is missing so the
    caller can skip the CSV-based plots gracefully.
    """
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    cols = ["step", "time", "KE", "PE", "Etot", "Tkin", "P"]
    if d.shape[1] < len(cols):
        return None                     # pre-diagnostics CSV; nothing to plot
    out = {name: d[:, i] for i, name in enumerate(cols)}
    if d.shape[1] > len(cols):          # newer CSV also logs external-field PE
        out["Uext"] = d[:, len(cols)]
    return out


def box_edges(Lx, Ly, Lz):
    """Return a list of (xs, ys, zs) segments tracing the 12 box edges."""
    c = [(0, 0, 0), (Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0),
         (0, 0, Lz), (Lx, 0, Lz), (Lx, Ly, Lz), (0, Ly, Lz)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),      # bottom
             (4, 5), (5, 6), (6, 7), (7, 4),      # top
             (0, 4), (1, 5), (2, 6), (3, 7)]      # verticals
    segs = []
    for a, b in edges:
        xs = [c[a][0], c[b][0]]
        ys = [c[a][1], c[b][1]]
        zs = [c[a][2], c[b][2]]
        segs.append((xs, ys, zs))
    return segs


def make_plots(energy, pos, meta, out_png):
    """Render a four-panel diagnostics figure from the time series (energy) and
    the trajectory (pos, shape [nframes, NP, 3] in metres) and save it to
    out_png.  Returns True on success, False if there is nothing to plot."""
    if energy is None:
        print("  (diagnostics plots skipped: energy CSV not found)")
        return False

    N, NP = meta["N"], meta["NP"]
    t_ns = energy["time"] * 1e9                       # simulation time [ns]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    (ax_e, ax_T), (ax_P, ax_orb) = axes

    # -- (1) energy conservation -------------------------------------------
    ax_e.plot(t_ns, energy["KE"],   color="#1f77b4", label="kinetic")
    ax_e.plot(t_ns, energy["PE"],   color="#2ca02c", label="Coulomb PE")
    # With an external E field the conserved total is KE + Coulomb PE + U_ext.
    # Show U_ext (when the run has a field) so the total visibly balances.
    if "Uext" in energy and np.any(energy["Uext"] != 0.0):
        ax_e.plot(t_ns, energy["Uext"], color="#ff7f0e", label="external-field PE")
        total_label = "total (KE + PE + U$_{ext}$)"
    else:
        total_label = "total"
    ax_e.plot(t_ns, energy["Etot"], color="#111111", lw=1.6, label=total_label)
    E0 = energy["Etot"][0]
    drift = 100.0 * (energy["Etot"][-1] - E0) / abs(E0) if E0 else 0.0
    ax_e.set_title("Energy conservation  (total drift %.2g%%)" % drift)
    ax_e.set_xlabel("time  [ns]"); ax_e.set_ylabel("energy  [J]")
    ax_e.legend(loc="best", fontsize=8); ax_e.grid(alpha=0.3)

    # -- (2) kinetic temperature -------------------------------------------
    ax_T.plot(t_ns, energy["Tkin"], color="#d62728", label="kinetic  T(t)")
    ax_T.axhline(meta["temperature"], color="0.4", ls="--",
                 label="set point  %.2e K" % meta["temperature"])
    ax_T.set_title("Kinetic temperature")
    ax_T.set_xlabel("time  [ns]"); ax_T.set_ylabel("T  [K]")
    ax_T.legend(loc="best", fontsize=8); ax_T.grid(alpha=0.3)

    # -- (3) pressure -------------------------------------------------------
    p_ideal = NP / meta["volume"] * KB * meta["temperature"]
    ax_P.plot(t_ns, energy["P"], color="#9467bd", label="virial  P(t)")
    ax_P.axhline(p_ideal, color="0.4", ls="--",
                 label="ideal gas  n k$_B$T = %.2e Pa" % p_ideal)
    ax_P.set_title("Plasma pressure  (mean %.3e Pa)" % np.mean(energy["P"]))
    ax_P.set_xlabel("time  [ns]"); ax_P.set_ylabel("P  [Pa]")
    ax_P.legend(loc="best", fontsize=8); ax_P.grid(alpha=0.3)

    # -- (4) Larmor orbits: xy tracks of a few sample particles -------------
    um = 1e6
    pe = [i for i in (0, N // 2, N - 1) if 0 <= i < N]
    pp = [i for i in (N, N + N // 2, NP - 1) if N <= i < NP]
    e_colors = ["#1f77b4", "#4c9be8", "#08306b"]
    p_colors = ["#d62728", "#ff9896", "#7f0000"]
    for n, i in enumerate(pe):
        ax_orb.plot(pos[:, i, 0] * um, pos[:, i, 1] * um, lw=0.8,
                    color=e_colors[n % len(e_colors)],
                    label="electron" if n == 0 else None)
        ax_orb.plot(pos[0, i, 0] * um, pos[0, i, 1] * um, "o",
                    color=e_colors[n % len(e_colors)], ms=4)
    for n, i in enumerate(pp):
        ax_orb.plot(pos[:, i, 0] * um, pos[:, i, 1] * um, lw=0.8,
                    color=p_colors[n % len(p_colors)],
                    label="proton" if n == 0 else None)
        ax_orb.plot(pos[0, i, 0] * um, pos[0, i, 1] * um, "o",
                    color=p_colors[n % len(p_colors)], ms=4)
    ax_orb.set_title("Sample particle paths  (xy projection)")
    ax_orb.set_xlabel("x  [µm]"); ax_orb.set_ylabel("y  [µm]")
    ax_orb.set_aspect("equal", "box")
    ax_orb.legend(loc="best", fontsize=8); ax_orb.grid(alpha=0.3)

    B = (meta["Bx"], meta["By"], meta["Bz"])
    E = (meta["Ex"], meta["Ey"], meta["Ez"])
    e_txt = ("   E=(%.2g, %.2g, %.2g) V/m" % E) if any(E) else ""
    fig.suptitle("classical plasma diagnostics   "
                 "N=%d(+%d)   B=(%.2g, %.2g, %.2g) T%s   T=%.1e K   P=%.2e Pa"
                 % (N, N, B[0], B[1], B[2], e_txt, meta["temperature"],
                    meta["pressure"]),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("  wrote %s" % out_png)
    return True


def main():
    paths = resolve_names(sys.argv)
    meta_path, TRAJ, GIF, MP4 = paths["meta"], paths["traj"], paths["gif"], paths["mp4"]
    if not os.path.exists(TRAJ):
        sys.exit("ERROR: trajectory file not found next to the meta file:\n"
                 "       %s\n       (expected %s)" % (meta_path, TRAJ))

    meta = read_meta(meta_path)
    N, NP, nframes = meta["N"], meta["NP"], meta["nframes"]

    data = np.fromfile(TRAJ, dtype=np.float32)
    expected = nframes * NP * 3
    if data.size != expected:
        sys.exit("ERROR: %s has %d floats, expected %d" % (TRAJ, data.size, expected))
    pos = data.reshape(nframes, NP, 3)

    energy = read_energy(paths["energy"])       # time series for the plots (or None)

    # Work in micrometres for readable axis numbers.
    um = 1e6
    pos_um = pos * um
    Lx, Ly, Lz = meta["Lx"] * um, meta["Ly"] * um, meta["Lz"] * um

    e_slice = slice(0, N)        # electrons: first N
    p_slice = slice(N, NP)       # protons:   next  N

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # static box wireframe
    for xs, ys, zs in box_edges(Lx, Ly, Lz):
        ax.plot(xs, ys, zs, color="0.6", lw=0.8, alpha=0.6)

    f0 = pos_um[0]
    scat_e = ax.scatter(f0[e_slice, 0], f0[e_slice, 1], f0[e_slice, 2],
                        s=1, c="#1f77b4", alpha=0.75, label="electrons (%d)" % N)
    scat_p = ax.scatter(f0[p_slice, 0], f0[p_slice, 1], f0[p_slice, 2],
                        s=1, c="#d62728", alpha=0.85, label="protons (%d)" % N)

    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
    ax.set_xlabel("x  [µm]")
    ax.set_ylabel("y  [µm]")
    ax.set_zlabel("z  [µm]")
    ax.legend(loc="upper right")

    B = (meta["Bx"], meta["By"], meta["Bz"])
    E = (meta["Ex"], meta["Ey"], meta["Ez"])
    frame_dt = meta["dt"] * meta["save_stride"]     # sim-time between frames
    e_txt = ("   E=(%.2g, %.2g, %.2g) V/m" % E) if any(E) else ""
    subtitle = "B=(%.2g, %.2g, %.2g) T%s   T=%.1e K   P=%.2e Pa" % (
        B[0], B[1], B[2], e_txt, meta["temperature"], meta["pressure"])
    title = ax.set_title("")

    def update(k):
        f = pos_um[k]
        # 3D scatter positions are updated through the private _offsets3d attr.
        scat_e._offsets3d = (f[e_slice, 0], f[e_slice, 1], f[e_slice, 2])
        scat_p._offsets3d = (f[p_slice, 0], f[p_slice, 1], f[p_slice, 2])
        ax.view_init(elev=22, azim=0.3 * k)          # slow orbit for depth cue
        title.set_text("classical plasma   t = %.2f ns   (frame %d/%d)\n%s"
                       % (k * frame_dt * 1e9, k + 1, nframes, subtitle))
        return scat_e, scat_p, title

    anim = FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)

    print("Rendering %d frames -> %s ..." % (nframes, GIF))
    anim.save(GIF, writer=PillowWriter(fps=20), dpi=90)
    print("  wrote %s" % GIF)

    if shutil.which("ffmpeg"):
        try:
            from matplotlib.animation import FFMpegWriter
            print("ffmpeg found -> also writing %s ..." % MP4)
            anim.save(MP4, writer=FFMpegWriter(fps=20, bitrate=2400), dpi=110)
            print("  wrote %s" % MP4)
        except Exception as exc:            # noqa: BLE001
            print("  (mp4 export skipped: %s)" % exc)
    else:
        print("ffmpeg not found -> GIF only (this is fine).")

    plt.close(fig)

    # Diagnostics figure (energy / temperature / pressure / sample orbits).
    print("Rendering diagnostics plots -> %s ..." % paths["plots"])
    make_plots(energy, pos, meta, paths["plots"])


if __name__ == "__main__":
    main()
