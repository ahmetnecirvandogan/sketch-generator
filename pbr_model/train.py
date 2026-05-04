"""Training loop scaffold (issue #21).

Standard PyTorch training boilerplate connecting the data loader (#19) and the
model (#20). This is a *scaffold* — it proves the wiring is correct and is
intentionally tuned for laptop CPU smoke-testing, not for real training runs.

**Variant A loss:**  ``L_albedo + λ₁·L_roughness + λ₂·L_lighting``  (all MSE)

**Variant B loss:**  ``L_render``  (MSE)

Acceptance criteria from #21:

- Runs end-to-end on a tiny batch (2 samples, CPU) without errors
- Loss decreases over 10 steps (proves backward pass connects)
- Saves a checkpoint ``.pt`` file
- Both variants runnable via ``--variant`` flag

CLI smoke test::

    python -m pbr_model.train --variant b --batch-size 2 --steps 10
    python -m pbr_model.train --variant a --batch-size 2 --steps 10  # needs lighting_sh

NOT chasing accuracy — just proves the loop works.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from pbr_model.dataset import make_dataloader
from pbr_model.model import make_model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    outputs: dict,
    batch: dict,
    variant: Literal["a", "b"],
    lambda_roughness: float = 1.0,
    lambda_lighting: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Variant A: weighted-MSE on albedo + roughness + lighting_sh.
    Variant B: MSE on the combined render.

    Returns (total_loss, components_dict) so we can log per-term scalars.
    """
    if variant == "a":
        l_albedo = F.mse_loss(outputs["albedo"], batch["albedo"])
        l_rough = F.mse_loss(outputs["roughness"], batch["roughness"])
        l_light = F.mse_loss(outputs["lighting_sh"], batch["lighting_sh"])
        total = l_albedo + lambda_roughness * l_rough + lambda_lighting * l_light
        return total, {
            "L_albedo": l_albedo.item(),
            "L_roughness": l_rough.item(),
            "L_lighting": l_light.item(),
            "total": total.item(),
        }
    # variant b
    l_render = F.mse_loss(outputs["render"], batch["render"])
    return l_render, {"L_render": l_render.item(), "total": l_render.item()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    metadata_path: str | Path = "dataset/metadata.jsonl",
    variant: Literal["a", "b"] = "b",
    batch_size: int = 2,
    steps: int = 10,
    lr: float = 1e-3,
    base_channels: int = 16,
    lambda_roughness: float = 1.0,
    lambda_lighting: float = 0.1,
    checkpoint_path: str | Path = "checkpoints/scaffold.pt",
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Run a tiny training loop. Returns a dict with first/last loss + checkpoint path."""
    if verbose:
        print(f"=== training scaffold | variant={variant} | bs={batch_size} | steps={steps} | device={device} ===")

    loader = make_dataloader(
        metadata_path=metadata_path, variant=variant, batch_size=batch_size, shuffle=True,
    )
    model = make_model(variant=variant, base_channels=base_channels).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    data_iter = iter(loader)

    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Single epoch may not cover ``steps`` iterations on tiny datasets — restart.
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move tensors to device; leave list[str] (prompt) on CPU.
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(batch["sketch"], batch["prompt"])
        loss, components = compute_loss(
            outputs, batch, variant=variant,
            lambda_roughness=lambda_roughness, lambda_lighting=lambda_lighting,
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose:
            comp_str = "  ".join(f"{k}={v:.4f}" for k, v in components.items() if k != "total")
            print(f"  step {step:>3}/{steps} | loss = {loss.item():.4f}  {comp_str}")

    elapsed = time.time() - t0

    # Save checkpoint (state_dict + a few key settings).
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "variant": variant,
            "base_channels": base_channels,
            "step": steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        },
        checkpoint_path,
    )
    if verbose:
        print(f"\ncheckpoint → {checkpoint_path}  (size: {checkpoint_path.stat().st_size / 1024:.1f} KB)")
        print(f"first loss: {losses[0]:.4f} | last loss: {losses[-1]:.4f} | elapsed: {elapsed:.1f}s")
        if losses[-1] < losses[0]:
            print("✓ loss decreased — backward pass is connected end-to-end.")
        else:
            print("⚠ loss did not decrease over the run; may need more steps or a higher LR.")

    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/metadata.jsonl")
    parser.add_argument("--variant", choices=["a", "b"], default="b",
                        help="A: separated PBR maps + lighting; B: combined render. Default B (no lighting_sh prereq).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lambda-roughness", type=float, default=1.0)
    parser.add_argument("--lambda-lighting", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="checkpoints/scaffold.pt")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    train(
        metadata_path=args.metadata,
        variant=args.variant,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        base_channels=args.base_channels,
        lambda_roughness=args.lambda_roughness,
        lambda_lighting=args.lambda_lighting,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
