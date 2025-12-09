#!/usr/bin/env python3
"""
CPU baseline: color filter (yellow + white) + Canny edge detection using OpenCV.
Runs the pipeline on ALL images in a folder.

Usage:
    python cpu_lane_canny_batch.py -i sample_images/ -o output/
"""

import argparse
import os
import time
import cv2
import numpy as np

NEW_IMAGE_HEIGHT = 260

# -------------------------------------------------------------
# ---- Utility functions
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU baseline: color filter + Canny edge detection (batch mode)"
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Directory containing JPEG/PNG images"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output_cpu",
        help="Directory to save results"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per image for stable timing"
    )
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_images_in_dir(directory):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(exts):
            files.append(os.path.join(directory, f))
    return sorted(files)


# -------------------------------------------------------------
# ---- Crop helper (vertical crop from bottom)
# -------------------------------------------------------------
def crop_vertically_from_bottom(bgr_img, new_height):
    h, w, c = bgr_img.shape
    if new_height > h:
        raise ValueError("new_height exceeds current height")
    start_row = h - new_height
    return bgr_img[start_row:h, :, :]


# -------------------------------------------------------------
# ---- Color filter + Canny pipeline
# -------------------------------------------------------------
def color_filter_yellow_white(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([15, 60, 150], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([179, 40, 255], dtype=np.uint8)

    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white = cv2.inRange(hsv, white_lower, white_upper)


    return yellow, white


def process_image(bgr_img):

    bgr_img = crop_vertically_from_bottom(bgr_img, new_height=NEW_IMAGE_HEIGHT)

    timings = {}

    t0 = time.perf_counter()
    yellow, white = color_filter_yellow_white(bgr_img)
    t1 = time.perf_counter()
    timings["color_filter_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.erode(yellow, kernel)
    white = cv2.erode(white, kernel)
    t1 = time.perf_counter()
    timings["erode_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.dilate(yellow, kernel)
    white = cv2.dilate(white, kernel)
    t1 = time.perf_counter()
    timings["dilate_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    t1 = time.perf_counter()
    timings["bgr2gray_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    t1 = time.perf_counter()
    timings["gaussian_blur_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    edges = cv2.Canny(blurred, 50, 150)
    t1 = time.perf_counter()
    timings["canny_ms"] = (t1 - t0) * 1000

    t0 = time.perf_counter()
    yellow_edges = cv2.bitwise_and(edges, edges, mask=yellow)
    white_edges = cv2.bitwise_and(edges, edges, mask=white)
    t1 = time.perf_counter()
    timings["bitwiseAND_ms"] = (t1 - t0) * 1000

    timings["total_ms"] = sum(timings.values())

    return {
        "gray": gray,
        "yellow": yellow,
        "white": white,
        "blurred": blurred,
        "edges": edges,
        "yellow_edges" : yellow_edges,
        "white_edges" : white_edges,
        "timings": timings
    }


# -------------------------------------------------------------
# ---- Main batch loop
# -------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    images = list_images_in_dir(args.input_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    print(f"Found {len(images)} images:")
    for img in images:
        print("  -", img)

    all_totals = []

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n=== Processing {base} ===")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  ERROR: unable to load.")
            continue

        # Repeat to average timing
        timings_accum = {}
        for _ in range(args.repeats):
            out = process_image(bgr)
            for k, v in out["timings"].items():
                timings_accum[k] = timings_accum.get(k, 0.0) + v

        # Average
        for k in timings_accum:
            timings_accum[k] /= args.repeats

        all_totals.append(timings_accum["total_ms"])

        print("Timing (avg over repeats):")
        for k, v in timings_accum.items():
            print(f"  {k:18s}: {v:8.3f} ms")

        # Save output images
        cv2.imwrite(f"{args.output_dir}/{base}_gray.png", out["gray"])
        cv2.imwrite(f"{args.output_dir}/{base}_yellow.png", out["yellow"])
        cv2.imwrite(f"{args.output_dir}/{base}_white.png", out["white"])
        cv2.imwrite(f"{args.output_dir}/{base}_blurred.png", out["blurred"])
        cv2.imwrite(f"{args.output_dir}/{base}_edges.png", out["edges"])
        cv2.imwrite(f"{args.output_dir}/{base}_yellow_edges.png", out["yellow_edges"])
        cv2.imwrite(f"{args.output_dir}/{base}_white_edges.png", out["white_edges"])

    # Final summary
    print("\n==============================")
    print(" Average total time per image ")
    print("==============================")
    avg = sum(all_totals) / len(all_totals)
    print(f"   {avg:.3f} ms (CPU, Python - OpenCV)")

    print("\nDone!")


if __name__ == "__main__":
    print("Running Main")
    main()
