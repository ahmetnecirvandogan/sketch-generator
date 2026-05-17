"""Training loop for the Sketch-to-PBR model.

Supports both Variant A (Separated PBR maps) and Variant B (Combined render)
with full epoch-based training and checkpoint saving.
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

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbr_model.dataset import make_dataloader
from pbr_model.model import make_model


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


def train(
    metadata_path: str | Path = "dataset/metadata.jsonl",
    variant: Literal["a", "b"] = "a",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-4,
    base_channels: int = 16,
    use_clip: bool = False,
    lambda_roughness: float = 1.0,
    lambda_lighting: float = 0.1,
    checkpoint_dir: str | Path = "checkpoints",
    save_every_epochs: int = 5,
) -> None:
    # Auto-detect best device (CUDA for VALAR, MPS for Mac, CPU as fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    text_kind = "CLIP" if use_clip else "stub"
    print(
        f"=== Starting Training | Variant: {variant.upper()} | Batch: {batch_size} | "
        f"Epochs: {epochs} | LR: {lr} | Text: {text_kind} | Device: {device} ==="
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loader = make_dataloader(
        metadata_path=metadata_path, 
        variant=variant, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    model = make_model(variant=variant, base_channels=base_channels, use_clip=use_clip).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(loader)
    print(f"Dataset loaded. Total batches per epoch: {total_batches}")

    t0_global = time.time()
    
    for epoch in range(1, epochs + 1):
        t0_epoch = time.time()
        epoch_losses = []
        epoch_components = {}
        
        for batch_idx, batch in enumerate(loader):
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

            epoch_losses.append(loss.item())
            
            # Accumulate components for logging
            for k, v in components.items():
                epoch_components[k] = epoch_components.get(k, 0.0) + v
                
            if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                print(f"  Epoch [{epoch}/{epochs}] Step [{batch_idx + 1}/{total_batches}] Loss: {loss.item():.4f}")

        # Average losses for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_comps = {k: v / len(epoch_losses) for k, v in epoch_components.items()}
        
        elapsed_epoch = time.time() - t0_epoch
        comp_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_comps.items() if k != "total")
        
        print(f"-> Epoch {epoch} complete in {elapsed_epoch:.1f}s | Avg Loss: {avg_loss:.4f} | {comp_str}")

        # Save latest checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "variant": variant,
            "base_channels": base_channels,
            "avg_loss": avg_loss,
        }
        
        latest_path = checkpoint_dir / f"latest_variant_{variant}.pt"
        torch.save(checkpoint_state, latest_path)
        
        # Save numbered checkpoints periodically
        if epoch % save_every_epochs == 0:
            epoch_path = checkpoint_dir / f"epoch_{epoch}_variant_{variant}.pt"
            torch.save(checkpoint_state, epoch_path)

    total_time = time.time() - t0_global
    print(f"\n=== Training Complete in {total_time/60:.1f} minutes ===")
    print(f"Latest checkpoint saved to {latest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/metadata.jsonl")
    parser.add_argument(
        "--variant", choices=["a", "b"], default="a",
        help="A: separated PBR maps + lighting; B: combined render. Default A (Primary).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--use-clip", action="store_true",
        help="Use frozen CLIP text encoder instead of the stub (real-training quality, ~250 MB download).",
    )
    parser.add_argument("--lambda-roughness", type=float, default=1.0)
    parser.add_argument("--lambda-lighting", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=5, help="Save a checkpoint every N epochs")
    
    args = parser.parse_args()

    train(
        metadata_path=args.metadata,
        variant=args.variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        use_clip=args.use_clip,
        lambda_roughness=args.lambda_roughness,
        lambda_lighting=args.lambda_lighting,
        checkpoint_dir=args.checkpoint_dir,
        save_every_epochs=args.save_every,
    )


if __name__ == "__main__":
    main()
