#!/usr/bin/env python3

import argparse
import numpy as np
"""
Generate an HSV threshold lookup table (LUT) and save it as a binary file.

LUT layout:
    index = r * 256 * 256 + g * 256 + b
    value = 0 or 1 (uint8)

HSV ranges used:
    H: 0–360 degrees
    S: 0–255
    V: 0–255
"""

YELLOW_UPPER_THRESH_H = 44 
YELLOW_UPPER_THRESH_S = 179 
YELLOW_UPPER_THRESH_V = 254

YELLOW_LOWER_THRESH_H = 16 
YELLOW_LOWER_THRESH_S = 51 
YELLOW_LOWER_THRESH_V = 91 

WHITE_UPPER_THRESH_H = 170 
WHTIE_UPPER_THRESH_S = 150 
WHITE_UPPER_THRESH_V = 255 

WHITE_LOWER_THRESH_H = 70 
WHTIE_LOWER_THRESH_S = 0 
WHITE_LOWER_THRESH_V = 150 


def generate_hsv_lut() -> np.ndarray:
    """
    Generate a 256^3 LUT as a flat uint8 array.
    Each entry is 0 or 1 depending on whether (R,G,B) is within HSV thresholds.
    """

    lut = np.empty(256 * 256 * 256, dtype=np.uint8)

    # Precompute G,B grids in [0,1]
    r_vals = np.linspace(0.0, 1.0, 255, endpoint=True)
    g_vals = np.linspace(0.0, 1.0, 255, endpoint=True)
    b_vals = np.linspace(0.0, 1.0, 255, endpoint=True)


    for r in range(256):
        for g in range(256):
            for b in range(256):
                lut_val = 0
                h,s,v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                if isYellow(h, s, v):
                    lut_val += 2
                if isWhite(h, s, v):
                    lut_val += 1
                lut[r * 256 * 256 + g * 256 + b] = lut_val
    return lut

def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert RGB [0,1] to HSV (H in degrees [0,360], S,V in [0,255])."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    # Hue calculation
    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:  # mx == b
        h = (60 * ((r - g) / diff) + 240) % 360

    # Saturation calculation
    s = 0 if mx == 0 else (diff / mx)

    # Value calculation
    v = mx

    return h, s * 255, v * 255 

def isYellow(h, s, v):
    return (YELLOW_LOWER_THRESH_H <= h <= YELLOW_UPPER_THRESH_H and
            YELLOW_LOWER_THRESH_S <= s <= YELLOW_UPPER_THRESH_S and
            YELLOW_LOWER_THRESH_V <= v <= YELLOW_UPPER_THRESH_V)

def isWhite(h, s, v):
    return (WHITE_LOWER_THRESH_H <= h <= WHITE_UPPER_THRESH_H and
            WHTIE_LOWER_THRESH_S <= s <= WHTIE_UPPER_THRESH_S and
            WHITE_LOWER_THRESH_V <= v <= WHITE_UPPER_THRESH_V)


def write_lut_bin(lut: np.ndarray, path: str) -> None:
    """Write the LUT (flat uint8 array) to a binary file."""
    with open(path, "wb") as f:
        f.write(lut.tobytes())
    print(f"Wrote LUT to {path} ({lut.size} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HSV threshold LUT and save as binary file."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="HSV_LUT.bin",
        help="Output filename for LUT binary (default: HSV_LUT.bin)",
    )

    args = parser.parse_args()

    lut = generate_hsv_lut()

    write_lut_bin(lut, args.output)


if __name__ == "__main__":
    main()
