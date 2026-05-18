"""Per-sample Qwen2-VL captions for rendered DF3D samples (issue #38).

Walks ``dataset/<bucket>/.../sample_*/render.png``, runs Qwen2-VL-7B on
each render, and writes a 1–2 sentence caption capturing color, pattern,
fabric type, lighting direction, and camera angle.

Each sample's per-sample ``metadata.json`` is updated in place:
- ``text`` field rewritten to the new caption + a photogrammetry tag
- ``vlm_caption_at`` timestamp added (used for idempotency)
- ``prompt.txt`` rewritten to match

Why this replaces PR #39's BLIP captions:
PR #39 captioned the **flat UV unwrap** PNGs, not the rendered scene.
Audit of the 1,212 produced captions found ~13 % hallucinations —
"holes", "torn", "broken" — coming from BLIP misinterpreting UV seams.
This script captions the **rendered image** instead, so the VLM sees
the cloth the way a viewer would.

Why per-sample, not per-garment:
Per-garment captions miss the per-sample variation in lighting, camera
angle, and visible folds. A single garment renders multiple samples,
each with different illumination — per-sample captures that signal.

Designed for a GPU node (VALAR). ~14 GB model, ~1 sec/image on A100.

Usage::

    # caption all DF3D samples (default)
    python -m pbr_model.postprocess_df3d_captions

    # also caption manual + procedural samples
    python -m pbr_model.postprocess_df3d_captions --include df3d,manual,procedural

    # recaption everything (override idempotency)
    python -m pbr_model.postprocess_df3d_captions --force

    # smoke test on a handful of samples first
    python -m pbr_model.postprocess_df3d_captions --limit 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path


PHOTOGRAMMETRY_TAG = "photogrammetry-scanned fabric"
CAPTION_PROMPT = (
    "Describe this rendered cloth garment in 1-2 short sentences. Include "
    "color, dominant pattern, fabric type if visible, key design features "
    "(buttons, seams, neckline), lighting direction, and the camera angle. "
    "Do not say it is a render or mention the background."
)


def _load_model(device: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"[qwen] loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Avoid device_map (needs `accelerate`); do .to(device) explicitly so the
    # script works with a minimal transformers + torch install.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return processor, model


def _caption_one(image, processor, model, device: str) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    out_text = processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]
    return out_text.strip()


def _iter_sample_dirs(repo_root: Path, buckets: list[str]) -> list[Path]:
    """Find every sample_*/ directory across the chosen buckets."""
    dirs: list[Path] = []
    for bucket in buckets:
        pattern = str(repo_root / "dataset" / bucket / "**" / "sample_*")
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if p.is_dir() and (p / "render.png").exists() and (p / "metadata.json").exists():
                dirs.append(p)
    return sorted(dirs)


def _build_text_field(caption: str, bucket: str) -> str:
    """Compose the metadata['text'] string.

    For df3d we append the photogrammetry tag so CLIP can still
    distinguish DF3D samples from procedural/manual ones. For other
    buckets the caption stands alone.
    """
    caption = caption.rstrip(".") + "."
    if bucket == "df3d":
        return f"{caption} {PHOTOGRAMMETRY_TAG}"
    return caption


def _bucket_for_sample(sample_dir: Path) -> str:
    """Infer bucket from sample path: dataset/<bucket>/..."""
    parts = sample_dir.parts
    if "dataset" in parts:
        i = parts.index("dataset")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include", default="df3d",
        help="comma-separated buckets to caption (default: df3d). "
             "Example: --include df3d,manual,procedural",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recaption samples that already have a 'vlm_caption_at' timestamp",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="cap the number of NEW captions written this run (0 = no cap). "
             "Idempotent skips don't count against the limit.",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="repo root override (default: cwd)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="how often to print progress (default: every 10 captions)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()
    buckets = [b.strip() for b in args.include.split(",") if b.strip()]

    print(f"[qwen] repo_root = {repo_root}")
    print(f"[qwen] buckets   = {buckets}")
    print(f"[qwen] force     = {args.force}")
    print(f"[qwen] limit     = {args.limit if args.limit else 'unlimited'}")

    sample_dirs = _iter_sample_dirs(repo_root, buckets)
    if not sample_dirs:
        sys.exit(f"[qwen] no sample directories found under dataset/{{{','.join(buckets)}}}/.../sample_*")
    print(f"[qwen] found {len(sample_dirs)} candidate sample dirs")

    try:
        import torch
    except ImportError:
        sys.exit("[qwen] torch not installed — pip install torch transformers Pillow")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[qwen] device = {device}")
    if device == "cpu":
        print("[qwen] WARNING: no CUDA. Qwen2-VL on CPU is impractically slow.")

    from PIL import Image
    processor, model = _load_model(device)

    new_count = 0
    skipped_idempotent = 0
    errors = 0

    for i, sample_dir in enumerate(sample_dirs):
        meta_path = sample_dir / "metadata.json"
        render_path = sample_dir / "render.png"

        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[qwen] skip (bad metadata.json) {sample_dir}: {e}")
            errors += 1
            continue

        if metadata.get("vlm_caption_at") and not args.force:
            skipped_idempotent += 1
            continue

        try:
            image = Image.open(render_path).convert("RGB")
            caption = _caption_one(image, processor, model, device)
        except Exception as e:
            print(f"[qwen] error captioning {sample_dir}: {e}")
            errors += 1
            continue

        bucket = _bucket_for_sample(sample_dir)
        text = _build_text_field(caption, bucket)

        metadata["text"] = text
        metadata["vlm_caption"] = caption
        metadata["vlm_caption_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        metadata["vlm_caption_model"] = "Qwen/Qwen2-VL-7B-Instruct"

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        with open(sample_dir / "prompt.txt", "w") as f:
            f.write(text + "\n")

        new_count += 1
        if new_count % args.checkpoint_every == 0 or new_count <= 3:
            print(f"[qwen] {new_count} captioned ({skipped_idempotent} skipped, {errors} errors)")
            print(f"       {sample_dir.relative_to(repo_root)}")
            print(f"       -> {caption[:140]}")

        if args.limit and new_count >= args.limit:
            print(f"[qwen] reached --limit {args.limit}, stopping")
            break

    print(
        f"[qwen] done. {new_count} new captions, "
        f"{skipped_idempotent} skipped (idempotent), {errors} errors."
    )


if __name__ == "__main__":
    # Allow `python pbr_model/postprocess_df3d_captions.py` from repo root (without -m).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
