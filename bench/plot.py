#!/usr/bin/env python3
"""
plot.py - draw the graphs required by the FIT3143 Lab #2 specification from
the CSV produced by run_bench.sh.

    python plot.py results/<site>.csv [--site LABEL] [--out graphs/]

Required graphs (numbered as in the Task 4 section of the spec):

  Task 1 (Open MPI)
    1  run time      : MPI vs POSIX Threads / OpenMP, increasing n
    2  speed-up      : MPI vs POSIX Threads / OpenMP, increasing n
    3  speed-up      : MPI vs POSIX Threads / OpenMP, increasing P (= T)
  Task 2 (hybrid)
    4  speed-up      : hybrid vs MPI, increasing threads at fixed P
    5  speed-up      : hybrid vs POSIX Threads / OpenMP, matched total threads
  Task 3 (Amdahl)
    6  MPI    : empirical vs theoretical speed-up, increasing P
    7  hybrid : empirical vs theoretical speed-up, increasing P x T
  Extras used in the discussion
    8  phase breakdown of the MPI run time per P (where the serial time goes)
    9  workload distribution comparison (block / cyclic / blockcyclic / wblock)
   10  per-rank compute time (load balance) for each distribution
   11  block-cyclic chunk size and OpenMP schedule sensitivity
   12  sieve kernel: speed-up vs its own serial sieve, and where the time goes
   15  empirical vs theoretical speed-up with increasing n (MPI at full core
       count and hybrid), from the serial fraction s(n) measured at P = 1

Speed-up is always measured against the Week 4 serial program, as the spec
requires.  Theoretical speed-up uses Amdahl's Law with the serial fraction
measured on the 1-process run (broadcast + gather + merge + write = serial,
compute = parallel), plus a Gustafson curve from the fractions measured at
each P.
"""

import argparse
import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- palette (same identities as the Lab 1 graphs) -------------------------
C = {
    "serial":  "#2a78d6",   # blue
    "pthread": "#eb6834",   # orange
    "openmp":  "#1baf7a",   # aqua
    "mpi":     "#8e44ad",   # purple
    "hybrid":  "#d62d3f",   # red
    "ideal":   "#9aa3ad",
}
LABEL = {"serial": "Serial (Week 4)", "pthread": "POSIX Threads", "openmp": "OpenMP",
         "mpi": "Open MPI (Task 1)", "hybrid": "MPI + OpenMP (Task 2)",
         "serial_sieve": "Serial sieve", "mpi_sieve": "Open MPI sieve", "hybrid_sieve": "hybrid sieve"}
DIST_C = {"block": "#2a78d6", "cyclic": "#eb6834", "blockcyclic": "#1baf7a", "wblock": "#8e44ad"}
PHASE_C = {"bcast": "#9aa3ad", "compute": "#8e44ad", "gather": "#eb6834", "merge": "#1baf7a", "write": "#2a78d6"}

NUM = ["n", "procs", "threads", "chunk", "sched_chunk", "run", "total", "bcast", "comp_max",
       "comp_min", "comp_avg", "gather", "merge", "write", "count", "wall"]


# ---- data ------------------------------------------------------------------
def load(path, site=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if site and r["site"] != site:
                continue
            if r["total"] in ("", "NA"):
                continue
            for k in NUM:
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None) else None
            rows.append(r)
    return rows


def key(r):
    return (r["impl"], r["n"], r["procs"], r["threads"], r["dist"], r["chunk"], r["sched"], r["sched_chunk"])


def aggregate(rows):
    """Median of every numeric column per configuration."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, rs in groups.items():
        m = dict(rs[0])
        for col in ["total", "bcast", "comp_max", "comp_min", "comp_avg", "gather", "merge", "write", "wall"]:
            vals = [r[col] for r in rs if r[col] is not None]
            m[col] = st.median(vals) if vals else None
        m["runs"] = len(rs)
        out[k] = m
    return out


def select(agg, impl, **cond):
    res = []
    for m in agg.values():
        if m["impl"] != impl:
            continue
        if all(m[k] == v for k, v in cond.items()):
            res.append(m)
    return res


def one(agg, impl, **cond):
    r = select(agg, impl, **cond)
    return r[0] if r else None


# ---- styling ----------------------------------------------------------------
def n_fmt(v, _):
    return f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"


def new_fig():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    return fig, ax


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)          # grid behind bars
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


PHYS = {"n": None}


def phys_marker(ax):
    if PHYS["n"]:
        ax.axvline(PHYS["n"], color="#c0392b", ls=":", lw=1)
        ax.annotate(f"{PHYS['n']} physical cores", (PHYS["n"], ax.get_ylim()[1]), fontsize=7,
                    color="#c0392b", ha="right", va="top", xytext=(-3, -2), textcoords="offset points")


def save(fig, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---- graphs ------------------------------------------------------------------
def series_vs_n(agg, impl, cores, dist):
    """(n, median row) for the full-core configuration of one implementation."""
    if impl == "serial":
        rs = select(agg, "serial")
    elif impl in ("pthread", "openmp"):
        rs = select(agg, impl, threads=cores)
    elif impl == "mpi":
        rs = select(agg, "mpi", procs=cores, dist=dist)
    else:
        rs = select(agg, "hybrid", procs=max(cores // 2, 1), threads=2, dist=dist)
    return sorted(((r["n"], r) for r in rs), key=lambda t: t[0])


def graph_1_2(agg, cores, dist, out):
    serial = {n: r["total"] for n, r in series_vs_n(agg, "serial", cores, dist)}
    hp = max(cores // 2, 1)
    lab = {"pthread": f"{LABEL['pthread']} ({cores} threads)", "openmp": f"{LABEL['openmp']} ({cores} threads)",
           "mpi": f"{LABEL['mpi']} ({cores} processes)", "hybrid": f"{LABEL['hybrid']} ({hp} x 2)"}
    fig, ax = new_fig()
    xs = sorted(serial)
    ax.plot(xs, [serial[n] for n in xs], color=C["serial"], marker="o", ms=3, label=LABEL["serial"])
    for impl in ("pthread", "openmp", "mpi", "hybrid"):
        pts = series_vs_n(agg, impl, cores, dist)
        if pts:
            ax.plot([n for n, _ in pts], [r["total"] for _, r in pts], color=C[impl], marker="o", ms=3, label=lab[impl])
    ax.xaxis.set_major_formatter(FuncFormatter(n_fmt))
    style(ax, "Graph 1 - Run time vs n", "n", "wall-clock time [s]")
    ax.legend(fontsize=8)
    save(fig, out, "g1_runtime_vs_n.png")

    fig, ax = new_fig()
    for impl in ("pthread", "openmp", "mpi", "hybrid"):
        pts = [(n, serial[n] / r["total"]) for n, r in series_vs_n(agg, impl, cores, dist) if n in serial]
        if pts:
            ax.plot([n for n, _ in pts], [s for _, s in pts], color=C[impl], marker="o", ms=3, label=lab[impl])
    ax.axhline(cores, color=C["ideal"], ls="--", lw=1, label=f"linear ({cores} cores)")
    ax.xaxis.set_major_formatter(FuncFormatter(n_fmt))
    style(ax, "Graph 2 - Speed-up vs serial, increasing n", "n", "speed-up vs serial")
    ax.legend(fontsize=8)
    save(fig, out, "g2_speedup_vs_n.png")


def graph_3(agg, n_big, cores, dist, out):
    base = one(agg, "serial", n=n_big)
    if not base:
        print("graph 3: no serial run at n =", n_big); return
    fig, ax = new_fig()
    ps = sorted({int(m["procs"]) for m in select(agg, "mpi", n=n_big, dist=dist)} | set(range(1, cores + 1)))
    for impl, cond in (("pthread", "threads"), ("openmp", "threads"), ("mpi", "procs")):
        pts = []
        for p in ps:
            r = one(agg, impl, n=n_big, **{cond: p}) if impl != "mpi" else one(agg, "mpi", n=n_big, procs=p, dist=dist)
            if r:
                pts.append((p, base["total"] / r["total"]))
        if pts:
            ax.plot(*zip(*pts), color=C[impl], marker="o", ms=4, label=LABEL[impl])
    wall_pts = [(p, base["total"] / r["wall"]) for p in ps
                for r in [one(agg, "mpi", n=n_big, procs=p, dist=dist)] if r and r.get("wall")]
    if wall_pts:
        ax.plot(*zip(*wall_pts), color=C["mpi"], ls=":", lw=1.2, marker="x", ms=4,
                label="Open MPI incl. mpirun launch + MPI_Init")
    lin = [p for p in ps if p <= cores]
    ax.plot(lin, lin, color=C["ideal"], ls="--", lw=1, label="linear")
    if max(ps) > cores:
        ax.axvspan(cores + 0.5, max(ps) + 0.5, color="#f4e3e5", alpha=0.6, lw=0)
        ax.annotate("over-subscribed\n(more processes than logical cores)", (cores + 0.7, ax.get_ylim()[1] * 0.97),
                    fontsize=7, color="#c0392b", va="top")
    style(ax, f"Graph 3 - Speed-up vs number of processes / threads (n = {int(n_big):,})",
          "MPI processes  (= threads for POSIX Threads / OpenMP)", "speed-up vs serial")
    ax.set_xticks(ps)
    ax.legend(fontsize=8)
    phys_marker(ax)
    save(fig, out, "g3_speedup_vs_procs.png")


def graph_4(agg, n_big, cores, dist, out):
    base = one(agg, "serial", n=n_big)
    if not base:
        return
    fig, ax = new_fig()
    shades = ["#f4a582", "#d62d3f", "#7a1020"]
    ps = [p for p in (1, 2, 4) if p <= cores]
    for i, p in enumerate(ps):
        hy = sorted(select(agg, "hybrid", n=n_big, procs=p, dist=dist), key=lambda r: r["threads"])
        if hy:
            ax.plot([r["threads"] for r in hy], [base["total"] / r["total"] for r in hy],
                    color=shades[i % 3], marker="o", ms=4, label=f"hybrid, P = {p}")
        m = one(agg, "mpi", n=n_big, procs=p, dist=dist)
        if m:
            ax.axhline(base["total"] / m["total"], color=shades[i % 3], ls=":", lw=1.2, label=f"MPI only, P = {p}")
    style(ax, f"Graph 4 - Hybrid vs Open MPI, increasing threads per process (n = {int(n_big):,})",
          "OpenMP threads per MPI process", "speed-up vs serial")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.legend(fontsize=8, ncol=2)
    save(fig, out, "g4_hybrid_vs_mpi_threads.png")


def graph_5(agg, n_big, cores, dist, out):
    base = one(agg, "serial", n=n_big)
    if not base:
        return
    fig, ax = new_fig()
    for impl in ("pthread", "openmp"):
        rs = sorted(select(agg, impl, n=n_big), key=lambda r: r["threads"])
        if rs:
            ax.plot([r["threads"] for r in rs], [base["total"] / r["total"] for r in rs],
                    color=C[impl], marker="o", ms=3, label=LABEL[impl])
    hy = select(agg, "hybrid", n=n_big, dist=dist)
    for r in hy:
        tot = r["procs"] * r["threads"]
        ax.scatter(tot, base["total"] / r["total"], color=C["hybrid"], s=22, zorder=3)
        ax.annotate(f"{int(r['procs'])}x{int(r['threads'])}", (tot, base["total"] / r["total"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color=C["hybrid"])
    ax.scatter([], [], color=C["hybrid"], s=22, label="hybrid P x T (total = P*T)")
    ax.plot(range(1, cores + 1), range(1, cores + 1), color=C["ideal"], ls="--", lw=1, label="linear")
    style(ax, f"Graph 5 - Hybrid vs POSIX Threads / OpenMP at matched total threads (n = {int(n_big):,})",
          "total threads  (POSIX/OpenMP: T;  hybrid: P x T)", "speed-up vs serial")
    ax.legend(fontsize=8)
    save(fig, out, "g5_hybrid_vs_threads_total.png")


def serial_parts(r):
    """Time the root spends outside the parallel compute phase.

    `gather` on the root also contains the time spent waiting for the slowest
    rank (load imbalance), which is not serial work.  The root's own wait is
    unknown here, so the mean wait comp_max - comp_avg is subtracted as an
    estimate; at P = 1 the correction is zero."""
    wait = max((r["comp_max"] or 0) - (r["comp_avg"] or 0), 0.0)
    gather_comm = max((r["gather"] or 0) - wait, 0.0)
    return (r["bcast"] or 0) + gather_comm + (r["merge"] or 0) + (r["write"] or 0)


def graph_6(agg, n_big, cores, dist, out):
    base = one(agg, "serial", n=n_big)
    r1 = one(agg, "mpi", n=n_big, procs=1, dist=dist)
    if not (base and r1):
        print("graph 6: need serial and MPI P=1 runs at n =", n_big); return
    t1 = r1["total"]
    s = serial_parts(r1) / t1                  # Amdahl serial fraction, measured at P = 1
    p = r1["comp_max"] / t1
    ps, emp, amd, amd_meas, gus = [], [], [], [], []
    for P in range(1, cores + 1):
        r = one(agg, "mpi", n=n_big, procs=P, dist=dist)
        if not r:
            continue
        ps.append(P)
        emp.append(base["total"] / r["total"])
        amd.append(base["total"] / (t1 * (s + p / P)))                      # ideal Amdahl
        amd_meas.append(base["total"] / (serial_parts(r) + r1["comp_max"] / P))  # Amdahl with measured serial parts(P)
        sP = serial_parts(r) / r["total"]
        gus.append(sP + (1 - sP) * P)                                      # Gustafson at P
    fig, ax = new_fig()
    ax.plot(ps, emp, color=C["mpi"], marker="o", ms=4, label="empirical (vs serial)")
    wall = [(P, base["total"] / r["wall"]) for P in ps for r in [one(agg, "mpi", n=n_big, procs=P, dist=dist)] if r.get("wall")]
    if wall:
        ax.plot(*zip(*wall), color=C["mpi"], ls=":", lw=1.2, marker="x", ms=4, label="empirical incl. mpirun launch + MPI_Init")
    ax.plot(ps, amd, color="#333333", ls="--", marker="s", ms=3, label=f"Amdahl, s = {100*s:.1f}% (measured at P = 1)")
    ax.plot(ps, amd_meas, color="#333333", ls="-.", marker="^", ms=3, label="Amdahl with communication measured at each P")
    ax.plot(ps, gus, color=C["openmp"], ls=":", marker="d", ms=3, label="Gustafson (fractions measured at each P)")
    ax.plot(ps, ps, color=C["ideal"], ls="--", lw=1, label="linear")
    style(ax, f"Graph 6 - Open MPI: empirical vs theoretical speed-up (n = {int(n_big):,})",
          "MPI processes", "speed-up")
    ax.set_xticks(ps)
    ax.legend(fontsize=8)
    phys_marker(ax)
    save(fig, out, "g6_mpi_empirical_vs_theory.png")
    return s


def graph_7(agg, n_big, cores, dist, out):
    base = one(agg, "serial", n=n_big)
    r1 = one(agg, "hybrid", n=n_big, procs=1, threads=1, dist=dist)
    if not (base and r1):
        print("graph 7: need serial and hybrid 1x1 runs at n =", n_big); return
    t1 = r1["total"]
    s = serial_parts(r1) / t1
    p = r1["comp_max"] / t1
    hy = sorted(select(agg, "hybrid", n=n_big, dist=dist), key=lambda r: (r["procs"] * r["threads"], r["procs"]))
    fig, ax = new_fig()
    for r in hy:
        tot = r["procs"] * r["threads"]
        ax.scatter(tot, base["total"] / r["total"], color=C["hybrid"], s=24, zorder=3)
        ax.annotate(f"{int(r['procs'])}x{int(r['threads'])}", (tot, base["total"] / r["total"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color=C["hybrid"])
    ax.scatter([], [], color=C["hybrid"], s=24, label="empirical, hybrid P x T (vs serial)")
    totals = sorted({int(r["procs"] * r["threads"]) for r in hy})
    ax.plot(totals, [base["total"] / (t1 * (s + p / N)) for N in totals], color="#333333", ls="--",
            marker="s", ms=3, label=f"Amdahl, s = {100*s:.1f}% (measured at 1 x 1)")
    ax.plot(totals, totals, color=C["ideal"], ls="--", lw=1, label="linear")
    style(ax, f"Graph 7 - Hybrid: empirical vs theoretical speed-up (n = {int(n_big):,})",
          "total cores used (P x T)", "speed-up")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.legend(fontsize=8)
    save(fig, out, "g7_hybrid_empirical_vs_theory.png")


def graph_8(agg, n_big, cores, dist, out):
    rs = sorted(select(agg, "mpi", n=n_big, dist=dist), key=lambda r: r["procs"])
    rs = [r for r in rs if r["bcast"] is not None]
    if not rs:
        return
    fig, ax = new_fig()
    xs = [r["procs"] for r in rs]
    bottom = [0.0] * len(rs)
    for phase, col in (("bcast", "bcast"), ("compute", "comp_max"), ("gather", "gather"), ("merge", "merge"), ("write", "write")):
        vals = [r[col] for r in rs]
        ax.bar(xs, vals, bottom=bottom, color=PHASE_C[phase], width=0.7,
               label=phase + (" (max over ranks)" if phase == "compute" else " (incl. wait for slowest rank)" if phase == "gather" else ""))
        bottom = [b + v for b, v in zip(bottom, vals)]
    style(ax, f"Graph 8 - Where the Open MPI run time goes (n = {int(n_big):,}, {dist})", "MPI processes", "wall-clock time [s]")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    save(fig, out, "g8_phase_breakdown.png")


def graph_9_10(agg, ranks, n_big, cores, out):
    base = one(agg, "serial", n=n_big)
    if not base:
        return
    fig, ax = new_fig()
    for d in ("block", "cyclic", "blockcyclic", "wblock"):
        rs = sorted((r for r in select(agg, "mpi", n=n_big, dist=d) if r["procs"] <= cores), key=lambda r: r["procs"])
        if rs:
            ax.plot([r["procs"] for r in rs], [base["total"] / r["total"] for r in rs],
                    color=DIST_C[d], marker="o", ms=4, label=d)
    ax.plot(range(1, cores + 1), range(1, cores + 1), color=C["ideal"], ls="--", lw=1, label="linear")
    style(ax, f"Graph 9 - Workload distribution schemes (n = {int(n_big):,})", "MPI processes", "speed-up vs serial")
    ax.legend(fontsize=8)
    phys_marker(ax)
    save(fig, out, "g9_distribution_compare.png")

    # per-rank compute time at the largest process count that has rank data
    by = defaultdict(lambda: defaultdict(list))
    for r in ranks:
        if r["impl"] == "mpi" and r["n"] == n_big:
            by[(r["dist"], r["procs"])][r["rank"]].append(r["compute"])
    if not by:
        return
    pmax = max(p for _, p in by)
    dists = [d for d in ("block", "cyclic", "blockcyclic", "wblock") if (d, pmax) in by]
    fig, ax = new_fig()
    w = 0.8 / max(len(dists), 1)
    for i, d in enumerate(dists):
        rk = sorted(by[(d, pmax)])
        ax.bar([r + i * w for r in rk], [st.median(by[(d, pmax)][r]) for r in rk], width=w, color=DIST_C[d], label=d)
    style(ax, f"Graph 10 - Compute time per rank, P = {int(pmax)} (n = {int(n_big):,})", "rank", "compute time [s]")
    ax.set_xticks([r + w * (len(dists) - 1) / 2 for r in range(int(pmax))])
    ax.set_xticklabels([str(r) for r in range(int(pmax))])
    ax.legend(fontsize=8)
    save(fig, out, "g10_load_balance.png")


def graph_11(agg, n_big, cores, dist, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=150)
    rs = sorted(select(agg, "mpi", n=n_big, procs=cores, dist="blockcyclic"), key=lambda r: r["chunk"])
    if rs:
        a1.plot([r["chunk"] for r in rs], [r["total"] for r in rs], color=DIST_C["blockcyclic"], marker="o", ms=4)
        a1.set_xscale("log", base=2)
    style(a1, f"Graph 11a - blockcyclic chunk size, P = {cores}", "chunk (candidates per block)", "wall-clock time [s]")
    hp = max(cores // 2, 1)
    for s_name, col in (("static", C["serial"]), ("dynamic", C["pthread"]), ("guided", C["openmp"])):
        rs = sorted(select(agg, "hybrid", n=n_big, procs=2, threads=hp, dist=dist, sched=s_name), key=lambda r: r["sched_chunk"])
        if rs:
            a2.plot([r["sched_chunk"] for r in rs], [r["total"] for r in rs], color=col, marker="o", ms=4, label=s_name)
            a2.set_xscale("log", base=2)
    style(a2, f"Graph 11b - OpenMP schedule, 2 x {hp} hybrid", "schedule chunk", "wall-clock time [s]")
    a2.legend(fontsize=8)
    save(fig, out, "g11_chunk_and_schedule.png")


def graph_12(agg, cores, out):
    """Sieve kernel: speed-up vs the serial sieve and the phase breakdown."""
    ns = sorted({m["n"] for m in agg.values() if m["impl"] == "mpi_sieve"})
    if not ns:
        return
    # the n with the most process counts is the P sweep
    cnt = defaultdict(set)
    for m in agg.values():
        if m["impl"] == "mpi_sieve":
            cnt[m["n"]].add(m["procs"])
    n_s = max(cnt, key=lambda n: (len(cnt[n]), n))
    base = one(agg, "serial_sieve", n=n_s)
    r1 = one(agg, "mpi_sieve", n=n_s, procs=1, dist="block")
    if not (base and r1):
        print("graph 12: need serial_sieve and mpi_sieve P=1 at n =", n_s); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    rs = sorted(select(agg, "mpi_sieve", n=n_s, dist="block"), key=lambda r: r["procs"])
    ps = [r["procs"] for r in rs]
    a1.plot(ps, [base["total"] / r["total"] for r in rs], color=C["mpi"], marker="o", ms=4, label="MPI sieve, block (vs serial sieve)")
    t1 = r1["total"]; s = serial_parts(r1) / t1; p = r1["comp_max"] / t1
    a1.plot(ps, [base["total"] / (t1 * (s + p / P)) for P in ps], color="#333333", ls="--", marker="s", ms=3,
            label=f"Amdahl, s = {100*s:.0f}% (write + gather at P = 1)")
    for r in select(agg, "hybrid_sieve", n=n_s):
        tot = r["procs"] * r["threads"]
        a1.scatter(tot, base["total"] / r["total"], color=C["hybrid"], s=20, zorder=3)
        a1.annotate(f"{int(r['procs'])}x{int(r['threads'])}", (tot, base["total"] / r["total"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7, color=C["hybrid"])
    a1.scatter([], [], color=C["hybrid"], s=20, label="hybrid sieve P x T")
    a1.plot(ps, ps, color=C["ideal"], ls="--", lw=1, label="linear")
    style(a1, f"Graph 12a - Sieve kernel speed-up (n = {int(n_s):,})", "MPI processes / total cores", "speed-up vs serial sieve")
    a1.legend(fontsize=7)
    bottom = [0.0] * len(rs)
    for phase, col in (("bcast", "bcast"), ("compute", "comp_max"), ("gather", "gather"), ("merge", "merge"), ("write", "write")):
        vals = [r[col] for r in rs]
        a2.bar(ps, vals, bottom=bottom, color=PHASE_C[phase], width=0.7, label=phase)
        bottom = [b + v for b, v in zip(bottom, vals)]
    a2.axhline(base["total"], color=C["serial"], ls=":", lw=1, label="serial sieve (whole process)")
    style(a2, "Graph 12b - Where the sieve run time goes", "MPI processes", "wall-clock time [s]")
    a2.set_xticks(ps)
    a2.legend(fontsize=7)
    save(fig, out, "g12_sieve.png")


def graph_15(agg, cores, dist, out):
    """Empirical vs theoretical speed-up with increasing n.  For every n the
    serial fraction s(n) comes from the P = 1 run of the same program
    (broadcast + gather + merge + write over total), and Amdahl gives
    S(n, P) = 1 / (s(n) + (1 - s(n)) / P) relative to that P = 1 run."""
    serial = {r["n"]: r["total"] for r in select(agg, "serial")}
    hp = max(cores // 2, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    for ax, impl, P, T, lab in ((a1, "mpi", cores, 1, f"Open MPI, {cores} processes"),
                                (a2, "hybrid", hp, 2, f"hybrid {hp} x 2")):
        emp, theo, svals = [], [], []
        for n in sorted(serial):
            r1 = one(agg, impl, n=n, procs=1, threads=1, dist=dist)
            rP = one(agg, impl, n=n, procs=P, threads=T, dist=dist)
            if not (r1 and rP):
                continue
            s_n = serial_parts(r1) / r1["total"]
            emp.append((n, serial[n] / rP["total"]))
            theo.append((n, serial[n] / (r1["total"] * (s_n + (1 - s_n) / (P * T)))))
            svals.append((n, 100 * s_n))
        if not emp:
            continue
        ax.plot(*zip(*emp), color=C[impl], marker="o", ms=3, label=f"empirical, {lab}")
        ax.plot(*zip(*theo), color="#333333", ls="--", marker="s", ms=3, label="Amdahl from s(n) measured at P = 1")
        ax.axhline(P * T, color=C["ideal"], ls="--", lw=1, label=f"linear ({P * T} cores)")
        ax2 = ax.twinx()
        ax2.plot(*zip(*svals), color=C["pthread"], ls=":", lw=1.2, label="serial fraction s(n) [%]")
        ax2.set_ylabel("serial fraction s(n) [%]", color=C["pthread"])
        ax2.set_ylim(0, max(v for _, v in svals) * 1.15)
        ax2.tick_params(axis="y", colors=C["pthread"])
        ax.xaxis.set_major_formatter(FuncFormatter(n_fmt))
        style(ax, f"Graph 15 - {lab}: empirical vs Amdahl, increasing n", "n", "speed-up vs serial")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    save(fig, out, "g15_speedup_vs_n_theory.png")


# ---- summary ---------------------------------------------------------------------
def summary(agg, n_big, cores, dist, s_frac):
    base = one(agg, "serial", n=n_big)
    if not base:
        return
    print(f"\nSummary at n = {int(n_big):,} (medians, {cores} cores, dist = {dist})")
    print(f"  serial                : {base['total']:.3f} s")
    for impl, cond in (("pthread", {"threads": cores}), ("openmp", {"threads": cores}), ("mpi", {"procs": cores, "dist": dist})):
        r = one(agg, impl, n=n_big, **cond)
        if r:
            print(f"  {LABEL[impl]:<22}: {r['total']:.3f} s   speed-up {base['total']/r['total']:.2f}x")
    hy = select(agg, "hybrid", n=n_big, dist=dist)
    if hy:
        b = min(hy, key=lambda r: r["total"])
        print(f"  best hybrid {int(b['procs'])}x{int(b['threads'])}       : {b['total']:.3f} s   speed-up {base['total']/b['total']:.2f}x")
    m = one(agg, "mpi", n=n_big, procs=cores, dist=dist)
    if m and m.get("wall"):
        print(f"  MPI P={cores} incl. launch    : {m['wall']:.3f} s (launch + MPI_Init/Finalize = {m['wall']-m['total']:.3f} s)"
              f"   speed-up {base['total']/m['wall']:.2f}x")
    if s_frac is not None:
        print(f"  Amdahl serial fraction (MPI, P=1): {100*s_frac:.2f}%  ->  max speed-up 1/s = {1/s_frac:.1f}x")
    ms = one(agg, "mpi_sieve", n=n_big, procs=cores, dist="block")
    ss = one(agg, "serial_sieve", n=n_big)
    if ms and ss:
        mt = one(agg, "mpi", n=n_big, procs=cores, dist=dist)
        print(f"  sieve kernel at same n    : serial sieve {ss['total']:.3f} s, MPI sieve P={cores} {ms['total']:.3f} s"
              + (f"  ({mt['total']/ms['total']:.0f}x faster than MPI trial division)" if mt else ""))
    ns = sorted({m["n"] for m in agg.values() if m["impl"] == "mpi_sieve"})
    if ns:
        n_s = max(ns)
        ss = one(agg, "serial_sieve", n=n_s); ms = one(agg, "mpi_sieve", n=n_s, procs=cores, dist="block")
        if ss and ms:
            print(f"  sieve at n = {int(n_s):,}: serial {ss['total']:.3f} s, MPI P={cores} {ms['total']:.3f} s, "
                  f"speed-up {ss['total']/ms['total']:.2f}x  (write {ms['write']:.3f} s = {100*ms['write']/ms['total']:.0f}% of total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--site", default=None, help="only rows with this site label")
    ap.add_argument("--out", default="graphs")
    ap.add_argument("--dist", default="blockcyclic", help="MPI distribution used for the main graphs")
    ap.add_argument("--physical-cores", type=int, default=None, help="draw a marker at the physical core count")
    ap.add_argument("--n-big", type=float, default=None, help="n for the P/T graphs (default: largest n with a P sweep)")
    ap.add_argument("--cores", type=int, default=None, help="core count (default: max procs seen)")
    args = ap.parse_args()

    rows = load(args.csv, args.site)
    if not rows:
        raise SystemExit("no rows")
    agg = aggregate(rows)
    ranks_path = args.csv.replace(".csv", "_ranks.csv")
    ranks = []
    if os.path.exists(ranks_path):
        with open(ranks_path, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("n", "procs", "threads", "chunk", "run", "rank", "compute", "count"):
                    r[k] = float(r[k])
                ranks.append(r)

    mpi_rows = [r for r in rows if r["impl"] in ("mpi", "mpi_sieve")]
    cores = args.cores or int(max(r["procs"] for r in mpi_rows))
    if args.n_big is None:
        # the n that has the most distinct MPI process counts = the P sweep
        cnt = defaultdict(set)
        for r in rows:
            if r["impl"] == "mpi":
                cnt[r["n"]].add(r["procs"])
        n_big = max(cnt, key=lambda n: (len(cnt[n]), n)) if cnt else max(r["n"] for r in mpi_rows)
    else:
        n_big = args.n_big
    os.makedirs(args.out, exist_ok=True)
    PHYS["n"] = args.physical_cores

    graph_1_2(agg, cores, args.dist, args.out)
    graph_3(agg, n_big, cores, args.dist, args.out)
    graph_4(agg, n_big, cores, args.dist, args.out)
    graph_5(agg, n_big, cores, args.dist, args.out)
    s = graph_6(agg, n_big, cores, args.dist, args.out)
    graph_7(agg, n_big, cores, args.dist, args.out)
    graph_8(agg, n_big, cores, args.dist, args.out)
    graph_9_10(agg, ranks, n_big, cores, args.out)
    graph_11(agg, n_big, cores, args.dist, args.out)
    graph_12(agg, cores, args.out)
    graph_15(agg, cores, args.dist, args.out)
    summary(agg, n_big, cores, args.dist, s)


if __name__ == "__main__":
    main()
