"""PyTorch Dataset + DataLoader for the cloth sketch → PBR maps task (issue #19).

Reads ``dataset/metadata.jsonl``. Two variant modes selectable via config:

- **Variant A** (primary): sketch + prompt → albedo, roughness, lighting_sh
  Returns ``{sketch, prompt, albedo, roughness, lighting_sh}``.
- **Variant B** (ablation): sketch + prompt → render
  Returns ``{sketch, prompt, render}``.

Variant A requires the ``lighting_sh`` field in each metadata entry (a list of
9 floats) — populated by ``pbr_model/preprocess_lighting_sh.py`` (issue #18).
If it's missing, the loader raises a clear error pointing at the preprocessing
step.

CLI smoke test::

    python -m pbr_model.dataset --variant b --batch-size 4
    python -m pbr_model.dataset --variant a --batch-size 4   # needs lighting_sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _load_rgb(path: Path) -> torch.Tensor:
    """Load PNG → CHW float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_grayscale(path: Path) -> torch.Tensor:
    """Load PNG → 1HW float tensor in [0, 1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _sketch_field(sample: dict) -> str | None:
    """Older metadata uses ``sketch_path``; ControlNet-style uses ``conditioning_image``.
    Accept either."""
    return sample.get("sketch_path") or sample.get("conditioning_image")


class ClothDataset(Dataset):
    """Dataset over a metadata.jsonl file.

    Parameters
    ----------
    metadata_path : str | Path
        Path to ``dataset/metadata.jsonl``. Can be relative to repo root.
    variant : {"a", "b"}
        Which training variant to feed: A returns separated PBR maps + lighting,
        B returns the combined render.
    repo_root : str | Path | None
        Used to resolve relative paths in metadata (e.g. ``dataset/.../render.png``).
        Defaults to the metadata file's parent's parent directory.
    """

    def __init__(
        self,
        metadata_path: str | Path = "dataset/metadata.jsonl",
        variant: Literal["a", "b"] = "a",
        repo_root: str | Path | None = None,
    ) -> None:
        if variant not in ("a", "b"):
            raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
        self.variant: Literal["a", "b"] = variant

        mp = Path(metadata_path)
        self.metadata_path = mp if mp.is_absolute() else (Path.cwd() / mp).resolve()
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata file not found: {self.metadata_path}")

        # Resolve relative paths in metadata against repo root. Default = cwd
        # (works when run from project root, the canonical case). Fall back to
        # parent-of-parent only for the standard ``dataset/metadata.jsonl`` layout.
        if repo_root:
            self.repo_root = Path(repo_root).resolve()
        elif self.metadata_path.name == "metadata.jsonl" and self.metadata_path.parent.name == "dataset":
            self.repo_root = self.metadata_path.parent.parent
        else:
            self.repo_root = Path.cwd()

        with open(self.metadata_path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise RuntimeError(f"no samples in {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else self.repo_root / p

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        prompt = s.get("text", "")

        sketch_rel = _sketch_field(s)
        if sketch_rel is None:
            raise KeyError(
                f"sample {idx} (frame {s.get('frame','?')}) is missing both "
                "'sketch_path' and 'conditioning_image' — run Stage 2 "
                "(generate_sketches.py) first."
            )
        sketch = _load_rgb(self._resolve(sketch_rel))

        if self.variant == "a":
            albedo = _load_rgb(self._resolve(s["pbr_albedo"]))
            roughness = _load_grayscale(self._resolve(s["pbr_roughness"]))
            if "lighting_sh" not in s or s["lighting_sh"] is None:
                raise KeyError(
                    f"sample {idx} (frame {s.get('frame','?')}) is missing "
                    "'lighting_sh' — run pbr_model/preprocess_lighting_sh.py "
                    "first (issue #18 / PR #25). Or use --variant b for "
                    "the ablation track."
                )
            lighting_sh = torch.tensor(s["lighting_sh"], dtype=torch.float32)
            return {
                "sketch": sketch,
                "prompt": prompt,
                "albedo": albedo,
                "roughness": roughness,
                "lighting_sh": lighting_sh,
            }

        # variant b
        render = _load_rgb(self._resolve(s["file_name"]))
        return {
            "sketch": sketch,
            "prompt": prompt,
            "render": render,
        }


def _collate(batch: list[dict]) -> dict:
    """Stack tensors; leave strings (prompts) as a list."""
    out: dict = {}
    for k in batch[0]:
        if isinstance(batch[0][k], str):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def make_dataloader(
    metadata_path: str | Path = "dataset/metadata.jsonl",
    variant: Literal["a", "b"] = "a",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    repo_root: str | Path | None = None,
) -> DataLoader:
    """Factory: builds ``ClothDataset`` and wraps it in a ``DataLoader``.

    ``batch_size=4`` is a placeholder for the scaffold; tune it once #20 (model)
    and #21 (training loop) land and we know GPU memory headroom.
    """
    ds = ClothDataset(metadata_path=metadata_path, variant=variant, repo_root=repo_root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )


def _smoke_test(args: argparse.Namespace) -> None:
    loader = make_dataloader(
        metadata_path=args.metadata,
        variant=args.variant,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(
        f"dataset: {len(loader.dataset)} samples | variant: {args.variant} | "
        f"batch_size: {args.batch_size}"
    )
    batch = next(iter(loader))
    print("first batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>12}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:>12}: list[{type(v[0]).__name__}] len={len(v)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", default="dataset/metadata.jsonl",
        help="path to metadata.jsonl (default: dataset/metadata.jsonl)",
    )
    parser.add_argument(
        "--variant", choices=["a", "b"], default="a",
        help="A: sketch+prompt → albedo+roughness+lighting_sh; B: sketch+prompt → render",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    _smoke_test(args)


if __name__ == "__main__":
    # Allow `python pbr_model/dataset.py` from repo root (without -m).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
