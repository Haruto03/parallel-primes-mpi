#!/usr/bin/env python3
"""
plot_caas.py - graphs for the CAAS runs: one node vs two nodes.

    python plot_caas.py results/caas-1node.csv results/caas-2node.csv [--out graphs]

  13  speed-up vs n on CAAS (MPI 8 procs on 1 node, MPI 8 procs on 2 nodes,
      hybrid 4 x 4 on 2 nodes) against the serial program run on CAAS
  14  where the time goes, 1 node vs 2 nodes: trial division (n = 2e7,
      10 MB of primes gathered) and the sieve (n = 5e8, 210 MB gathered) -
      the fabric cost s_fabric of Topic 7A becomes visible with data volume

The serial baseline lives in the 1-node file (serial_baseline.job); all
CAAS nodes are the same AMD Epyc VM class, so it is used for both.
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

C1, C2, CH, CS = "#8e44ad", "#d62d3f", "#1baf7a", "#2a78d6"
PHASE_C = {"bcast": "#9aa3ad", "compute": "#8e44ad", "gather": "#eb6834", "merge": "#1baf7a", "write": "#2a78d6"}


def load(path):
    d = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["total"] in ("", "NA"):
                continue
            d[(r["impl"], int(float(r["n"])), r["dist"], int(float(r["procs"])), int(float(r["threads"])))].append(r)
    return d


def med(rs, k):
    return st.median(float(r[k]) for r in rs) if rs else None


def n_fmt(v, _):
    return f"{v/1e6:g}M"


def style(ax, title, xl, yl):
    ax.set_title(title, loc="left", fontsize=10, pad=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("one"); ap.add_argument("two"); ap.add_argument("--out", default="graphs")
    ap.add_argument("--dist", default="blockcyclic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A, B = load(a.one), load(a.two)
    ns = sorted({k[1] for k in A if k[0] == "serial"})
    serial = {n: med(A[("serial", n, "-", 1, 1)], "total") for n in ns}

    # ---- graph 13: speed-up vs n ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    series = [
        ("mpi", A, a.dist, 8, 1, C1, "-", "o", "Open MPI, 8 processes, 1 node"),
        ("mpi", A, a.dist, 16, 1, C1, "--", "s", "Open MPI, 16 processes, 1 node"),
        ("mpi", B, a.dist, 8, 1, C2, "-", "o", "Open MPI, 8 processes, 2 nodes (4 + 4)"),
        ("mpi", B, a.dist, 16, 1, C2, "-.", "^", "Open MPI, 16 processes, 2 nodes (8 + 8)"),
        ("mpi", B, a.dist, 32, 1, C2, "--", "s", "Open MPI, 32 processes, 2 nodes (16 + 16)"),
        ("hybrid", B, a.dist, 4, 4, CH, "-", "o", "hybrid 4 processes x 4 threads, 2 nodes (16 cores)"),
    ]
    for impl, D, dist, p, t, col, ls, mk, lab in series:
        pts = [(n, serial[n] / med(D[(impl, n, dist, p, t)], "total")) for n in ns if D.get((impl, n, dist, p, t))]
        if pts:
            ax.plot(*zip(*pts), color=col, ls=ls, marker=mk, ms=4, label=lab)
    for c in (8, 16, 32):
        ax.axhline(c, color="#9aa3ad", ls=":", lw=1)
        ax.annotate(f"linear, {c} cores", (ns[0], c), fontsize=7, color="#9aa3ad", va="bottom")
    ax.xaxis.set_major_formatter(FuncFormatter(n_fmt))
    style(ax, "Graph 13 - CAAS: speed-up vs serial (serial run on a CAAS node)", "n", "speed-up vs serial")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "g13_caas_speedup.png")); plt.close(fig)
    print("wrote g13_caas_speedup.png")

    # ---- graph 14: phases, 1 node vs 2 nodes ------------------------------
    n_trial = max(ns)
    n_sieve = max((k[1] for k in A if k[0] == "mpi_sieve"), default=None)
    cfgs = [("trial, n = %s" % f"{n_trial:,}", A[("mpi", n_trial, a.dist, 8, 1)], B[("mpi", n_trial, a.dist, 8, 1)])]
    if n_sieve and B.get(("mpi_sieve", n_sieve, "block", 8, 1)):
        cfgs.append(("sieve, n = %s" % f"{n_sieve:,}", A[("mpi_sieve", n_sieve, "block", 8, 1)], B[("mpi_sieve", n_sieve, "block", 8, 1)]))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.2 * len(cfgs), 4.4), dpi=150, squeeze=False)
    for ax, (title, r1, r2) in zip(axes[0], cfgs):
        xs, labels = [0, 1], ["1 node", "2 nodes"]
        bottom = [0.0, 0.0]
        for phase, col in (("bcast", "bcast"), ("compute", "comp_max"), ("gather", "gather"), ("merge", "merge"), ("write", "write")):
            vals = [med(r1, col), med(r2, col)]
            ax.bar(xs, vals, bottom=bottom, color=PHASE_C[phase], width=0.55, label=phase)
            for x, b, v in zip(xs, bottom, vals):
                if phase == "gather":
                    ax.annotate(f"gather {1000*v:.0f} ms", (x + 0.3, b + v / 2), fontsize=7, va="center")
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        style(ax, f"Graph 14 - MPI 8 processes, {title}", "", "wall-clock time [s]")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "g14_caas_phases.png")); plt.close(fig)
    print("wrote g14_caas_phases.png")

    # ---- numbers ------------------------------------------------------------
    print(f"\nCAAS, n = {n_trial:,} (medians):  serial {serial[n_trial]:.3f} s")
    for impl, D, p, t, lab in (("mpi", A, 8, 1, "MPI 8 / 1 node"), ("mpi", A, 16, 1, "MPI 16 / 1 node"), ("mpi", B, 8, 1, "MPI 8 / 2 nodes"),
                               ("mpi", B, 16, 1, "MPI 16 / 2 nodes"), ("mpi", B, 32, 1, "MPI 32 / 2 nodes"), ("hybrid", B, 4, 4, "hybrid 4x4 / 2 nodes")):
        rs = D.get((impl, n_trial, a.dist, p, t))
        if not rs:
            continue
        cores = p * t
        print(f"  {lab:22s} {med(rs,'total'):.3f} s  speed-up {serial[n_trial]/med(rs,'total'):5.2f}x on {cores:2d} cores ({100*serial[n_trial]/med(rs,'total')/cores:.0f}% efficiency)  gather {1000*med(rs,'gather'):.0f} ms  launch {med(rs,'wall')-med(rs,'total'):.2f} s")
    if n_sieve:
        ss = med(A[("serial_sieve", n_sieve, "-", 1, 1)], "total")
        for D, lab in ((A, "1 node"), (B, "2 nodes")):
            rs = D[("mpi_sieve", n_sieve, "block", 8, 1)]
            print(f"  sieve n={n_sieve:,} {lab:8s} {med(rs,'total'):.3f} s (serial sieve {ss:.3f} s) gather {1000*med(rs,'gather'):.0f} ms write {med(rs,'write'):.3f} s")
    print("\ndistribution schemes on CAAS (8 procs, 1 node / 2 nodes, n = %s):" % f"{n_trial:,}")
    for d in ("block", "cyclic", "blockcyclic", "wblock"):
        r1, r2 = A.get(("mpi", n_trial, d, 8, 1)), B.get(("mpi", n_trial, d, 8, 1))
        if r1 and r2:
            print(f"  {d:12s} {med(r1,'total'):.3f} / {med(r2,'total'):.3f} s   imbalance {100*(1-med(r1,'comp_min')/med(r1,'comp_max')):.0f}% / {100*(1-med(r2,'comp_min')/med(r2,'comp_max')):.0f}%")


if __name__ == "__main__":
    main()
