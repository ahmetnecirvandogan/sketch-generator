"""Project per-sample lighting to order-2 spherical harmonic coefficients.

Reads dataset/metadata.jsonl, projects each sample's lights[] (constant +
directional, in any combination) to 9 luminance SH coefficients, and writes
the result back as a new `lighting_sh` field on each line. Idempotent.

Output format: 9 floats (order-2 real SH, scalar luminance via Rec 709).
The dataset currently uses neutral (R=G=B) lighting per sample, so a single
luminance channel preserves all of the information. If the renderer ever
emits chromatic lighting, switch to per-channel RGB (27 floats) by replacing
the luminance() call with the raw RGB triplet.

Acceptance criteria for issue #18:
- pbr_model/preprocess_lighting_sh.py
- reads existing metadata without breaking other fields
- 9 floats per sample
- idempotent: overwrites lighting_sh on rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SH_COEFFS = 9  # order-2 real SH


def luminance(rgb: list[float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sh_basis_order2(direction: np.ndarray) -> np.ndarray:
    """Real spherical harmonic basis Y_l^m for l=0..2, evaluated at a unit vector."""
    x, y, z = direction
    Y = np.empty(SH_COEFFS)
    Y[0] = 0.5 * np.sqrt(1.0 / np.pi)                      # l=0, m=0
    Y[1] = np.sqrt(3.0 / (4.0 * np.pi)) * y                # l=1, m=-1
    Y[2] = np.sqrt(3.0 / (4.0 * np.pi)) * z                # l=1, m=0
    Y[3] = np.sqrt(3.0 / (4.0 * np.pi)) * x                # l=1, m=1
    Y[4] = 0.5 * np.sqrt(15.0 / np.pi) * x * y             # l=2, m=-2
    Y[5] = 0.5 * np.sqrt(15.0 / np.pi) * y * z             # l=2, m=-1
    Y[6] = 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z * z - 1.0)  # l=2, m=0
    Y[7] = 0.5 * np.sqrt(15.0 / np.pi) * x * z             # l=2, m=1
    Y[8] = 0.25 * np.sqrt(15.0 / np.pi) * (x * x - y * y)  # l=2, m=2
    return Y


def project_lights(lights: list[dict]) -> list[float]:
    coeffs = np.zeros(SH_COEFFS)
    for light in lights:
        t = light.get("type")
        if t == "directional":
            d = np.asarray(light["direction"], dtype=float)
            n = np.linalg.norm(d)
            if n == 0.0:
                continue
            d = d / n
            intensity = luminance(light["irradiance"])
            coeffs += intensity * sh_basis_order2(d)
        elif t == "constant":
            # Constant ambient projects only onto the DC term Y_0^0.
            # Integral of Y_0^0 over the sphere = sqrt(4*pi).
            intensity = luminance(light["radiance"])
            coeffs[0] += intensity * np.sqrt(4.0 * np.pi)
        else:
            # Unknown light types are skipped rather than failing — keeps the
            # script forward-compatible with new emitters added to Stage 1.
            continue
    return coeffs.tolist()


def process(metadata_path: Path) -> tuple[int, int]:
    """Returns (samples_processed, samples_skipped)."""
    lines = metadata_path.read_text().splitlines()
    out_lines = []
    processed = 0
    skipped = 0
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        sample = json.loads(line)
        if "lights" not in sample:
            skipped += 1
            out_lines.append(line)
            continue
        sample["lighting_sh"] = project_lights(sample["lights"])
        out_lines.append(json.dumps(sample))
        processed += 1
    metadata_path.write_text("\n".join(out_lines) + "\n")
    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("dataset/metadata.jsonl"),
        help="Path to the centralised metadata.jsonl (default: dataset/metadata.jsonl)",
    )
    args = parser.parse_args()
    if not args.metadata.exists():
        raise SystemExit(f"metadata file not found: {args.metadata}")
    processed, skipped = process(args.metadata)
    print(f"lighting_sh written to {processed} samples ({skipped} skipped, no lights[] field)")


if __name__ == "__main__":
    main()
