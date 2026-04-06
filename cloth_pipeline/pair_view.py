"""
Side-by-side render + conditioning preview image.

Only reads existing PNGs (``cv2.imread``) and stacks them — no Mitsuba render
and no ``generate_sketch`` / conditioning regeneration.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from cloth_pipeline.paths import DATASET_DIR, PAIR_VIEWS_DIR

_LABEL_BGR = (40, 40, 40)
_BAND_BGR = (245, 245, 245)


def _normalize_frame_id(frame: str | int) -> str:
    s = str(frame).strip()
    if s.isdigit():
        return f"{int(s):04d}"
    return s


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == height:
        return img
    w = max(1, int(round(iw * (height / float(ih)))))
    return cv2.resize(
        img,
        (w, height),
        interpolation=cv2.INTER_AREA if height < ih else cv2.INTER_CUBIC,
    )


class RenderConditioningPairView:
    """
    Compose a single BGR image: **render | gap | conditioning** at a common height.

    Use :meth:`from_paths` or :meth:`from_frame`, then :meth:`compose` to get a
    ``uint8`` array suitable for ``cv2.imwrite``.
    """

    def __init__(
        self,
        render_bgr: np.ndarray,
        conditioning_bgr: np.ndarray,
        *,
        gap_px: int = 8,
        target_height: int | None = None,
    ) -> None:
        if render_bgr is None or render_bgr.ndim != 3 or render_bgr.shape[2] != 3:
            raise ValueError("render_bgr must be HxWx3 BGR uint8")
        if conditioning_bgr is None or conditioning_bgr.ndim != 3 or conditioning_bgr.shape[2] != 3:
            raise ValueError("conditioning_bgr must be HxWx3 BGR uint8")
        self._render = render_bgr
        self._cond = conditioning_bgr
        self._gap_px = max(0, int(gap_px))
        self._target_height = int(target_height) if target_height is not None else None

    @classmethod
    def from_paths(
        cls,
        render_path: str,
        conditioning_path: str,
        **kwargs: Any,
    ) -> RenderConditioningPairView:
        r = cv2.imread(render_path, cv2.IMREAD_COLOR)
        c = cv2.imread(conditioning_path, cv2.IMREAD_COLOR)
        if r is None:
            raise FileNotFoundError(f"Could not read render: {render_path}")
        if c is None:
            raise FileNotFoundError(f"Could not read conditioning: {conditioning_path}")
        return cls(r, c, **kwargs)

    @classmethod
    def from_frame(
        cls,
        frame: str | int,
        *,
        dataset_dir: str | None = None,
        **kwargs: Any,
    ) -> RenderConditioningPairView:
        fs = _normalize_frame_id(frame)
        root = dataset_dir or DATASET_DIR
        render_path = os.path.join(root, "renders", f"render_{fs}.png")
        conditioning_path = os.path.join(root, "conditioning", f"conditioning_{fs}.png")
        return cls.from_paths(render_path, conditioning_path, **kwargs)

    def compose(
        self,
        *,
        label_render: str = "Render",
        label_conditioning: str = "Conditioning",
        label_band_px: int = 36,
        font_scale: float = 0.65,
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Returns a single BGR ``uint8`` image (optional top band with labels).
        ``label_band_px`` ≤ 0 disables the band (images only).
        """
        h_tgt = self._target_height
        if h_tgt is None:
            h_tgt = max(self._render.shape[0], self._cond.shape[0])
        h_tgt = max(1, h_tgt)

        left = _resize_to_height(self._render, h_tgt)
        right = _resize_to_height(self._cond, h_tgt)
        gap_w = self._gap_px
        gap = np.full((h_tgt, gap_w, 3), 255, dtype=np.uint8) if gap_w > 0 else None

        parts = [left]
        if gap is not None:
            parts.append(gap)
        parts.append(right)
        row = np.hstack(parts)
        total_w = row.shape[1]

        if label_band_px <= 0:
            return row

        band_h = int(label_band_px)
        band = np.full((band_h, total_w, 3), _BAND_BGR, dtype=np.uint8)
        mid_l = left.shape[1] // 2
        mid_r = left.shape[1] + gap_w + right.shape[1] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        for text, cx in ((label_render, mid_l), (label_conditioning, mid_r)):
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = int(cx - tw / 2)
            y = int(band_h / 2 + th / 3)
            cv2.putText(
                band,
                text,
                (max(4, x), max(th + 2, y)),
                font,
                font_scale,
                _LABEL_BGR,
                thickness,
                cv2.LINE_AA,
            )

        return np.vstack([band, row])

    def save(
        self,
        out_path: str,
        *,
        label_render: str = "Render",
        label_conditioning: str = "Conditioning",
        label_band_px: int = 36,
    ) -> str:
        """Composes and writes PNG (path returned for convenience)."""
        img = self.compose(
            label_render=label_render,
            label_conditioning=label_conditioning,
            label_band_px=label_band_px,
        )
        cv2.imwrite(out_path, img)
        return out_path


def write_all_pairs(
    *,
    dataset_dir: str | None = None,
    out_dir: str | None = None,
    gap_px: int = 8,
    target_height: int | None = None,
    label_band_px: int = 36,
) -> list[str]:
    """
    For each ``renders/render_XXXX.png`` with a matching ``conditioning/conditioning_XXXX.png``,
    write ``pair_views/pair_XXXX.png``. Does not render or regenerate anything.
    """
    import glob

    root = dataset_dir or DATASET_DIR
    dest = out_dir or PAIR_VIEWS_DIR
    os.makedirs(dest, exist_ok=True)
    pattern = os.path.join(root, "renders", "render_*.png")
    written: list[str] = []
    for render_path in sorted(glob.glob(pattern)):
        base = os.path.basename(render_path)
        fs = base.replace("render_", "").replace(".png", "")
        cond_path = os.path.join(root, "conditioning", f"conditioning_{fs}.png")
        if not os.path.isfile(cond_path):
            continue
        pv = RenderConditioningPairView.from_paths(
            render_path,
            cond_path,
            gap_px=gap_px,
            target_height=target_height,
        )
        out_path = os.path.join(dest, f"pair_{fs}.png")
        pv.save(out_path, label_band_px=label_band_px)
        written.append(out_path)
    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Compose render|conditioning side-by-side from existing PNGs only (no re-render)."
        )
    )
    p.add_argument(
        "frame",
        nargs="?",
        default=None,
        help='Frame id, e.g. "6" or "0006" (omit with --all)',
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=f"Write every pair to dataset pair_views/ (default: {PAIR_VIEWS_DIR})",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Output path for single-frame mode (default: pair_<frame>.png in cwd)",
    )
    p.add_argument("--out-dir", default="", help="With --all: output directory (default: dataset/pair_views)")
    p.add_argument("--dataset", default="", help="Override dataset directory")
    p.add_argument("--height", type=int, default=0, help="Target row height (0 = max of inputs)")
    p.add_argument("--gap", type=int, default=8, help="Gap between panels in pixels")
    p.add_argument("--no-labels", action="store_true", help="Omit top label band")
    args = p.parse_args()

    ds = args.dataset or None
    th = args.height if args.height > 0 else None
    lbl = 0 if args.no_labels else 36

    if args.all:
        od = args.out_dir or None
        paths = write_all_pairs(
            dataset_dir=ds,
            out_dir=od,
            gap_px=args.gap,
            target_height=th,
            label_band_px=lbl,
        )
        for path in paths:
            print(path)
    else:
        if not args.frame:
            p.error("frame is required unless --all is set")
        pv = RenderConditioningPairView.from_frame(
            args.frame,
            dataset_dir=ds,
            gap_px=args.gap,
            target_height=th,
        )
        fs = _normalize_frame_id(args.frame)
        out = args.output or os.path.join(os.getcwd(), f"pair_{fs}.png")
        pv.save(out, label_band_px=lbl)
        print(out)
