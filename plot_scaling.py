#!/usr/bin/env python3
# ============================================================================
#  plot_scaling.py  --  strong- & weak-scaling analysis of the plasma sim
# ============================================================================
#
#  Reads the timing records written by ./plasma_sim_serial and
#  ./plasma_sim_parallel (one row per run) from plasma_scaling.csv and produces:
#
#    scaling_strong.png  -- speedup and parallel efficiency vs core count, for
#                           each fixed problem size (strong scaling).
#    scaling_weak.png    -- wall time and weak efficiency vs core count, for the
#                           runs that hold work-per-core constant (weak scaling).
#
#  It figures out on its own which rows belong to which study:
#    * Strong scaling  = parallel rows sharing the same (N, nsteps): the size is
#      fixed, only the thread count changes.
#    * Weak scaling    = parallel rows whose work-per-core (NP^2 * nsteps /
#      threads) is (nearly) constant across thread counts.  Because the force
#      kernel is O(N^2), that is the N = N0*sqrt(p) family run_scaling.sh emits.
#
#  Baseline convention (important): speedup/efficiency use the PARALLEL binary on
#  ONE core, T_par(1), as the reference -- the standard way to report the scaling
#  of a parallel code.  The parallel force kernel uses the "full-row" scheme,
#  which does 2x the pair work of the serial upper-triangle loop; measuring
#  against T_par(1) isolates parallel scalability from that constant algorithmic
#  factor.  The serial binary's time is still shown as context (absolute speedup
#  vs the fastest single-core code) in the printout and as a dashed line.
#
#  Generate the data first with:   ./run_scaling.sh
#  Then:                           python3 plot_scaling.py [plasma_scaling.csv]
#
#  Headless (Agg backend) so it runs on a compute node with no display.
# ----------------------------------------------------------------------------

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All simulation output (incl. the timing CSV) lives here; override with
# PLASMA_OUTDIR to match plasma_sim_*.c and run_scaling.sh.
OUTDIR      = os.environ.get("PLASMA_OUTDIR", "output")
CSV_DEFAULT = os.path.join(OUTDIR, "plasma_scaling.csv")
TIME_COL    = "wall_total_s"   # total loop time; == compute time in timing-only runs
WEAK_TOL    = 0.15             # rel. tolerance for "constant work per core"


def load_rows(path):
    """Read the timing CSV into a list of dicts with the fields we need."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                row = {
                    "version": r["version"],
                    "threads": int(r["threads"]),
                    "N":       int(r["N"]),
                    "NP":      int(r["NP"]),
                    "nsteps":  int(r["nsteps"]),
                    "t":       float(r[TIME_COL]),
                }
            except (KeyError, ValueError):
                continue
            if row["t"] > 0.0:
                row["work_per_core"] = (float(row["NP"]) ** 2
                                        * row["nsteps"] / row["threads"])
                rows.append(row)
    return rows


def min_time_by_threads(rows):
    """Collapse repeated runs: {threads: min wall time} (best of any repeats)."""
    out = {}
    for r in rows:
        p = r["threads"]
        if p not in out or r["t"] < out[p]:
            out[p] = r["t"]
    return out


# ----------------------------------------------------------------------------
#  Strong scaling
# ----------------------------------------------------------------------------
def plot_strong(rows, out_png):
    parallel = [r for r in rows if r["version"] == "parallel"]
    serial   = [r for r in rows if r["version"] == "serial"]

    # Group parallel runs by fixed problem size (N, nsteps).
    groups = {}
    for r in parallel:
        groups.setdefault((r["N"], r["nsteps"]), []).append(r)
    groups = {k: v for k, v in groups.items()
              if len({r["threads"] for r in v}) >= 2}
    if not groups:
        print("  strong scaling skipped: need a fixed (N, nsteps) parallel run "
              "at >= 2 core counts (see run_scaling.sh).")
        return False

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(12, 5))
    max_p = 1
    print("\nStrong scaling (fixed problem size, baseline = parallel on 1 core):")
    for (N, nsteps), grp in sorted(groups.items()):
        tbt = min_time_by_threads(grp)
        ps = np.array(sorted(tbt))
        times = np.array([tbt[p] for p in ps])
        t1 = tbt[ps[0]]                     # parallel time at the fewest cores
        speedup = t1 / times
        eff = speedup / (ps / ps[0])        # efficiency relative to that baseline
        max_p = max(max_p, ps.max())

        label = "N=%d (%d particles)" % (N, 2 * N)
        line, = ax_s.plot(ps, speedup, "o-", label="%s  vs 1 core" % label)
        ax_e.plot(ps, eff, "o-", color=line.get_color(), label=label)

        # Serial context: absolute speedup vs the fastest single-core code.
        ser = [s for s in serial if s["N"] == N and s["nsteps"] == nsteps]
        tser = min(s["t"] for s in ser) if ser else None
        if tser:
            ax_s.plot(ps, tser / times, "s--", color=line.get_color(), alpha=0.6,
                      label="%s  vs serial" % label)

        print("  %s, %d steps  [T_par(1 core)=%.4gs%s]" %
              (label, nsteps, t1,
               ", T_serial=%.4gs" % tser if tser else ""))
        for p, tt, sp, e in zip(ps, times, speedup, eff):
            extra = ("   vs-serial %.2fx" % (tser / tt)) if tser else ""
            print("      %3d cores : time %.4gs   speedup %5.2fx   "
                  "efficiency %4.0f%%%s" % (p, tt, sp, 100 * e, extra))
        if tser:
            print("      (note: parallel on 1 core is %.2fx the serial time -- the "
                  "full-row kernel's 2x pair work)" % (t1 / tser))

    ideal = np.array([1, max_p])
    ax_s.plot(ideal, ideal, "k:", lw=1.2, label="ideal (linear)")
    ax_s.set_title("Strong scaling: speedup")
    ax_s.set_xlabel("cores"); ax_s.set_ylabel("speedup  T(1) / T(p)")
    ax_s.grid(alpha=0.3); ax_s.legend(fontsize=8)

    ax_e.axhline(1.0, color="k", ls=":", lw=1.2, label="ideal (100%)")
    ax_e.set_title("Strong scaling: parallel efficiency")
    ax_e.set_xlabel("cores"); ax_e.set_ylabel("efficiency  T(1) / (p·T(p))")
    ax_e.set_ylim(0, 1.15); ax_e.grid(alpha=0.3); ax_e.legend(fontsize=8)

    fig.suptitle("Plasma simulation — strong scaling", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("  wrote %s" % out_png)
    return True


# ----------------------------------------------------------------------------
#  Weak scaling
# ----------------------------------------------------------------------------
def cluster_constant_work(rows, tol):
    """Group rows whose work-per-core is equal to within a relative tolerance.
    Strong-scaling rows (work/core ~ 1/p) fall into small clusters; the weak
    family (work/core constant) forms one cluster spanning many core counts."""
    clusters = []
    for r in sorted(rows, key=lambda x: x["work_per_core"]):
        w = r["work_per_core"]
        for c in clusters:
            if abs(w - c["ref"]) <= tol * c["ref"]:
                c["rows"].append(r)
                c["ref"] = np.median([x["work_per_core"] for x in c["rows"]])
                break
        else:
            clusters.append({"ref": w, "rows": [r]})
    return clusters


def plot_weak(rows, out_png):
    # Use parallel rows only, so a same-size serial run can't distort the series.
    parallel = [r for r in rows if r["version"] == "parallel"]
    clusters = cluster_constant_work(parallel, WEAK_TOL)

    def span(c):
        return len({r["threads"] for r in c["rows"]})

    best = max((c for c in clusters if span(c) >= 2), key=span, default=None)
    if best is None:
        print("  weak scaling skipped: need a constant work-per-core parallel "
              "series at >= 2 core counts (run_scaling.sh's N=N0*sqrt(p) sweep).")
        return False

    # For each core count keep the row whose work-per-core is closest to the
    # cluster reference (guards against a stray strong-scaling row leaking in).
    ref = best["ref"]
    per_thread = {}
    for r in best["rows"]:
        p = r["threads"]
        d = abs(r["work_per_core"] - ref)
        if p not in per_thread or d < per_thread[p][0] or (
                d == per_thread[p][0] and r["t"] < per_thread[p][1]["t"]):
            per_thread[p] = (d, r)

    ps = np.array(sorted(per_thread))
    chosen = [per_thread[p][1] for p in ps]
    times = np.array([r["t"] for r in chosen])
    t1 = times[0]                                  # parallel on the fewest cores
    weak_eff = t1 / times

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(12, 5))

    ax_t.plot(ps, times, "o-", color="#9467bd", label="measured")
    ax_t.axhline(t1, color="k", ls=":", lw=1.2,
                 label="ideal (constant = %.3gs)" % t1)
    ax_t.set_title("Weak scaling: wall time")
    ax_t.set_xlabel("cores  (N = N0·√p, work/core fixed)")
    ax_t.set_ylabel("wall time  [s]")
    ax_t.set_ylim(0, max(times.max(), t1) * 1.2)
    ax_t.grid(alpha=0.3); ax_t.legend(fontsize=8)

    ax_e.plot(ps, weak_eff, "o-", color="#9467bd", label="measured")
    ax_e.axhline(1.0, color="k", ls=":", lw=1.2, label="ideal (100%)")
    ax_e.set_title("Weak scaling: efficiency")
    ax_e.set_xlabel("cores"); ax_e.set_ylabel("weak efficiency  T(1) / T(p)")
    ax_e.set_ylim(0, 1.15); ax_e.grid(alpha=0.3); ax_e.legend(fontsize=8)

    fig.suptitle("Plasma simulation — weak scaling  "
                 "(constant work/core: O(N²) kernel ⇒ N ∝ √p)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print("\nWeak scaling (constant work/core, baseline = parallel on 1 core):")
    for p, r, t, e in zip(ps, chosen, times, weak_eff):
        print("  %3d cores : N=%d (%d particles), time %.4gs, "
              "weak efficiency %4.0f%%" % (p, r["N"], r["NP"], t, 100 * e))
    print("  wrote %s" % out_png)
    return True


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    if not os.path.exists(path):
        sys.exit("ERROR: %s not found -- run ./run_scaling.sh first." % path)

    rows = load_rows(path)
    if not rows:
        sys.exit("ERROR: no usable timing rows in %s." % path)
    print("Loaded %d timing rows from %s" % (len(rows), path))

    os.makedirs(OUTDIR, exist_ok=True)
    did_strong = plot_strong(rows, os.path.join(OUTDIR, "scaling_strong.png"))
    did_weak   = plot_weak(rows, os.path.join(OUTDIR, "scaling_weak.png"))
    if not (did_strong or did_weak):
        print("\nNothing plotted yet. Collect more data with ./run_scaling.sh "
              "(it sweeps several core counts for both studies).")


if __name__ == "__main__":
    main()
