#!/usr/bin/env python3

import re
import statistics
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------- File Path Constants ---------- 
CPU_LOG_PATH = "python_comparison/python_timing.log"
CUDA_LOG_PATH = "cuda_img_processing/cuda_timing.log"


# ---------- Parsing helpers ----------

CPU_PATTERN = re.compile(
    r'^\s*([A-Za-z0-9_]+)\s*:\s*([\d.]+)\s*ms',
    re.MULTILINE,
)

CUDA_PATTERN = re.compile(
    r'^\s*([A-Za-z0-9_ ()]+?)\s*:\s*([\d.]+)\s*ms',
    re.MULTILINE,
)

# Map raw log keys -> canonical stage names
CPU_TO_CANON = {
    "color_filter_ms":  "color_filter",
    "erode_ms":         "erode",
    "dilate_ms":        "dilate",
    "gaussian_blur_ms": "gaussian_blur",
    "canny_ms":         "canny_edges",
    "bitwiseAND_ms":    "bitwiseAND",
    "total_ms":         "Total time",
    # Add more if needed
}

CUDA_TO_CANON = {
    "color_filter":   "color_filter",
    "erode (both)":   "erode",
    "dilate (both)":  "dilate",
    "gaussian_blur":  "gaussian_blur",
    "canny_edges":    "canny_edges",
    "bitwiseAND":     "bitwiseAND",
    "Total time":     "Total time",

    # Explicit mappings for memcpy entries (cover common variants)
    "H2D copy":       "H2D memcpy",
    "H2D copies":     "H2D memcpy",
    "H2D memcpy":     "H2D memcpy",

    "D2H copy":       "D2H memcpy",
    "D2H copies":     "D2H memcpy",
    "D2H memcpy":     "D2H memcpy",
}


def parse_cpu_log(text: str):
    """
    Returns dict: raw_key -> list of times (ms)
    """
    out = {}
    for key, val in CPU_PATTERN.findall(text):
        out.setdefault(key, []).append(float(val))
    return out


def parse_cuda_log(text: str):
    """
    Returns dict: raw_key -> list of times (ms)
    """
    out = {}
    for key, val in CUDA_PATTERN.findall(text):
        key = key.strip()
        out.setdefault(key, []).append(float(val))
    return out


def average_dict_lists(d):
    return {k: statistics.mean(v) for k, v in d.items() if v}


def to_canonical(avg_dict, mapping):
    """
    Map raw keys (from avg_dict) into canonical stage names using mapping.
    If multiple raw keys map to same canonical name, they are averaged again.
    """
    canon = {}
    canon_lists = {}
    for raw_key, value in avg_dict.items():
        name = mapping.get(raw_key)
        if not name:
            continue
        canon_lists.setdefault(name, []).append(value)
    for name, vals in canon_lists.items():
        canon[name] = statistics.mean(vals)
    return canon


# ---------- Plotting ----------

def plot_comparison(stages, cpu_times, cuda_times, speedups, output_path=None):
    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, cpu_times, width, label="CPU (Python/OpenCV)")
    ax.bar(x + width/2, cuda_times, width, label="CUDA")

    ax.set_ylabel("Time per image (ms)")
    ax.set_title("Per-stage timing: CPU vs CUDA (including H2D/D2H memcpy)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.legend()

    # Annotate speedup above bars
    for i, s in enumerate(speedups):
        if math.isnan(s):
            label = "n/a"
        else:
            label = f"{s:.1f}×"
        ymax = max(cpu_times[i], cuda_times[i])
        ax.text(
            x[i],
            ymax * 1.02 if ymax > 0 else 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.show()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compare per-stage timing between Python (CPU) and CUDA logs."
    )
    parser.add_argument(
        "--cpu-log",
        default=CPU_LOG_PATH,
        help=f"Path to Python timing log (default: {CPU_LOG_PATH})",
    )
    parser.add_argument(
        "--cuda-log",
        default=CUDA_LOG_PATH,
        help=f"Path to CUDA timing log (default: {CUDA_LOG_PATH})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional path to save the figure instead of showing it",
    )
    args = parser.parse_args()

    # Read logs
    with open(args.cpu_log, "r") as f:
        cpu_text = f.read()
    with open(args.cuda_log, "r") as f:
        cuda_text = f.read()

    cpu_raw = parse_cpu_log(cpu_text)
    cuda_raw = parse_cuda_log(cuda_text)

    if not cpu_raw:
        raise SystemExit("No CPU timing entries found. Check the CPU log format.")
    if not cuda_raw:
        raise SystemExit("No CUDA timing entries found. Check the CUDA log format.")

    cpu_avg_raw = average_dict_lists(cpu_raw)
    cuda_avg_raw = average_dict_lists(cuda_raw)

    cpu_canon = to_canonical(cpu_avg_raw, CPU_TO_CANON)
    cuda_canon = to_canonical(cuda_avg_raw, CUDA_TO_CANON)

    # Use the UNION of stages so we can include memcpy-only stages
    all_stages = sorted(set(cpu_canon) | set(cuda_canon))
    if not all_stages:
        raise SystemExit(
            "No stages found after canonical mapping. "
            "Check CPU_TO_CANON and CUDA_TO_CANON."
        )

    print("\nPer-stage averages (ms) and speedup (CPU / CUDA):\n")
    print(f"{'Stage':<16} {'CPU(ms)':>10} {'CUDA(ms)':>10} {'Speedup':>10}")
    print("-" * 50)

    cpu_vals_plot = []
    cuda_vals_plot = []
    speedups = []

    for stage in all_stages:
        cpu_t = cpu_canon.get(stage, float("nan"))
        cuda_t = cuda_canon.get(stage, float("nan"))

        # Values for printing
        cpu_str = f"{cpu_t:10.3f}" if not math.isnan(cpu_t) else f"{'n/a':>10}"
        cuda_str = f"{cuda_t:10.3f}" if not math.isnan(cuda_t) else f"{'n/a':>10}"

        # Values for plotting (use 0 for missing so bar shows as 0 height)
        cpu_plot = cpu_t if not math.isnan(cpu_t) else 0.0
        cuda_plot = cuda_t if not math.isnan(cuda_t) else 0.0

        # Speedup only if we have both and cuda>0
        if (not math.isnan(cpu_t)) and (not math.isnan(cuda_t)) and cuda_t > 0:
            speedup = cpu_t / cuda_t
        else:
            speedup = float("nan")

        speedup_str = "n/a" if math.isnan(speedup) else f"{speedup:>7.2f}×"

        print(f"{stage:<16} {cpu_str} {cuda_str} {speedup_str:>10}")

        cpu_vals_plot.append(cpu_plot)
        cuda_vals_plot.append(cuda_plot)
        speedups.append(speedup)

    # Plot
    plot_comparison(all_stages, cpu_vals_plot, cuda_vals_plot, speedups, args.output)


if __name__ == "__main__":
    main()
