#!/usr/bin/env python3

import re
import statistics
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt


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
    # you *can* add "bgr2gray_ms" here if you later add a CUDA timing for it
}

CUDA_TO_CANON = {
    "color_filter":   "color_filter",
    "erode (both)":   "erode",
    "dilate (both)":  "dilate",
    "gaussian_blur":  "gaussian_blur",
    "canny_edges":    "canny_edges",
    "bitwiseAND":     "bitwiseAND",
    "Total time":     "Total time",
    # "D2H copies" and "Total kernels" left out on purpose
}


def parse_cpu_log(text: str):
    """
    Returns dict: raw_key -> list of times (ms), e.g.
    {
      'color_filter_ms': [2.44, 0.39, ...],
      ...
    }
    """
    out = {}
    for key, val in CPU_PATTERN.findall(text):
        out.setdefault(key, []).append(float(val))
    return out


def parse_cuda_log(text: str):
    """
    Returns dict: raw_key -> list of times (ms), e.g.
    {
      'color_filter': [0.12, 0.13, ...],
      ...
    }
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
    ax.set_title("Per-stage timing: CPU vs CUDA")
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
            ymax * 1.02,
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
        default="python_comparison/python_timing.log",
        help="Path to Python timing log (default: python_timing.log)",
    )
    parser.add_argument(
        "--cuda-log",
        default="cuda_img_processing/cuda_timing.log",
        help="Path to CUDA timing log (default: cuda_timing.log)",
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

    common_stages = sorted(set(cpu_canon) & set(cuda_canon))
    if not common_stages:
        raise SystemExit(
            "No common stages between CPU and CUDA logs. "
            "Check the mapping dictionaries CPU_TO_CANON and CUDA_TO_CANON."
        )

    print("\nPer-stage averages (ms) and speedup (CPU / CUDA):\n")
    print(f"{'Stage':<16} {'CPU(ms)':>10} {'CUDA(ms)':>10} {'Speedup':>10}")
    print("-" * 50)

    cpu_vals = []
    cuda_vals = []
    speedups = []

    for stage in common_stages:
        cpu_t = cpu_canon[stage]
        cuda_t = cuda_canon[stage]
        speedup = cpu_t / cuda_t if cuda_t > 0 else float("nan")

        cpu_vals.append(cpu_t)
        cuda_vals.append(cuda_t)
        speedups.append(speedup)

        speedup_str = "n/a" if math.isnan(speedup) else f"{speedup:>7.2f}×"
        print(f"{stage:<16} {cpu_t:10.3f} {cuda_t:10.3f} {speedup_str:>10}")

    # Plot
    plot_comparison(common_stages, cpu_vals, cuda_vals, speedups, args.output)


if __name__ == "__main__":
    main()
