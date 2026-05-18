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
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


Split = Literal["train", "val", "test", "all"]


def _mesh_split_assignment(
    mesh_id: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Split:
    """Deterministic mesh → split. Stable across runs for a given seed.

    Holds out whole meshes (issue #43): the network never sees any sample
    from a held-out mesh during training. Random per-sample splits would
    leak — same garment under different lighting/camera would appear in
    both train and val.
    """
    h = hashlib.md5(f"{mesh_id}:{seed}".encode()).hexdigest()
    fraction = int(h[:8], 16) / 0xFFFFFFFF
    if fraction < val_fraction:
        return "val"
    if fraction < val_fraction + test_fraction:
        return "test"
    return "train"


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
        split: Split = "all",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        if variant not in ("a", "b"):
            raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train/val/test/all, got {split!r}")
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction={val_fraction} + test_fraction={test_fraction} "
                "must be in [0, 1) and sum to < 1.0"
            )
        self.variant: Literal["a", "b"] = variant
        self.split: Split = split

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
            all_samples = [json.loads(line) for line in f if line.strip()]
        if not all_samples:
            raise RuntimeError(f"no samples in {self.metadata_path}")

        # Split filtering (issue #43): hold out whole meshes, not random samples.
        # Identifier = "mesh_file" (the .obj path relative to repo root).
        if split == "all":
            self.samples = all_samples
        else:
            self.samples = [
                s for s in all_samples
                if _mesh_split_assignment(
                    s.get("mesh_file", s.get("frame", "")),
                    val_fraction,
                    test_fraction,
                    split_seed,
                ) == split
            ]
        if not self.samples:
            raise RuntimeError(
                f"split={split!r} has no samples — {len(all_samples)} total samples "
                f"in metadata, but none assigned to this split with "
                f"val_fraction={val_fraction}, test_fraction={test_fraction}, "
                f"split_seed={split_seed}"
            )

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
    split: Split = "all",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_seed: int = 42,
) -> DataLoader:
    """Factory: builds ``ClothDataset`` and wraps it in a ``DataLoader``.

    ``batch_size=4`` is a placeholder for the scaffold; tune it once #20 (model)
    and #21 (training loop) land and we know GPU memory headroom.

    Issue #43: pass ``split="train" | "val" | "test"`` to hold out whole meshes
    deterministically. Use the same ``split_seed`` across train/val loaders for
    a non-overlapping mesh-id partition.
    """
    ds = ClothDataset(
        metadata_path=metadata_path,
        variant=variant,
        repo_root=repo_root,
        split=split,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
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
        split=args.split,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
    )
    print(
        f"dataset: {len(loader.dataset)} samples | variant: {args.variant} | "
        f"split: {args.split} | batch_size: {args.batch_size}"
    )
    batch = next(iter(loader))
    print("first batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>12}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:>12}: list[{type(v[0]).__name__}] len={len(v)}")

    if args.verify_split:
        # Sanity check: assert train/val/test mesh sets are disjoint with the same seed.
        train_ds = ClothDataset(
            metadata_path=args.metadata, variant=args.variant, split="train",
            val_fraction=args.val_fraction, test_fraction=args.test_fraction,
            split_seed=args.split_seed,
        )
        val_ds = ClothDataset(
            metadata_path=args.metadata, variant=args.variant, split="val",
            val_fraction=args.val_fraction, test_fraction=args.test_fraction,
            split_seed=args.split_seed,
        )
        test_ds = ClothDataset(
            metadata_path=args.metadata, variant=args.variant, split="test",
            val_fraction=args.val_fraction, test_fraction=args.test_fraction,
            split_seed=args.split_seed,
        )
        train_meshes = {s.get("mesh_file", "") for s in train_ds.samples}
        val_meshes = {s.get("mesh_file", "") for s in val_ds.samples}
        test_meshes = {s.get("mesh_file", "") for s in test_ds.samples}
        assert not (train_meshes & val_meshes), "train ∩ val is non-empty"
        assert not (train_meshes & test_meshes), "train ∩ test is non-empty"
        assert not (val_meshes & test_meshes), "val ∩ test is non-empty"
        print(
            f"split sanity: train={len(train_meshes)} meshes / {len(train_ds)} samples, "
            f"val={len(val_meshes)} meshes / {len(val_ds)} samples, "
            f"test={len(test_meshes)} meshes / {len(test_ds)} samples — all disjoint ✓"
        )


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
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all",
        help="Mesh-level split (issue #43). Holds out whole meshes, not random samples.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--verify-split", action="store_true",
        help="After loading, build train/val/test datasets and assert their mesh sets are disjoint.",
    )
    args = parser.parse_args()
    _smoke_test(args)


if __name__ == "__main__":
    # Allow `python pbr_model/dataset.py` from repo root (without -m).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
