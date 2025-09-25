#!/usr/bin/env python3
"""
LoRA fine-tuning skeleton for HunyuanVideo-11B on Apple Silicon (MPS).

Notes
- This is a minimal, educational scaffold designed for tiny batches and very
  short clips. Full fine-tuning of an 11B video model on M2 Max is likely
  impractical; expect to use very small ranks, resolutions, and few steps.
- Requires: PyTorch with MPS, diffusers with HunyuanVideo pipeline, transformers.
- Data: run scripts/prepare_dataset.py first to create ./data/frames and dataset.jsonl.

What it does
- Loads HunyuanVideo pipeline.
- Injects LoRA adapters into Linear layers of the video transformer.
- Trains only LoRA params on a simple noise-prediction loss over short frame windows.
- Saves LoRA weights only (does not modify base model).
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def device_select() -> torch.device:
    if torch.backends.mps.is_available():
        # Maximize math perf on Apple GPUs where applicable
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return torch.device("mps")
    return torch.device("cpu")


class FrameWindowDataset(Dataset):
    def __init__(self, manifest_path: str, num_frames: int = 8):
        self.items: List[Dict] = []
        with open(manifest_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                frames_dir = Path(obj["frames_dir"]) if "frames_dir" in obj else None
                if not frames_dir or not frames_dir.exists():
                    continue
                frames = sorted(list(frames_dir.glob("*.jpg")))
                if len(frames) < num_frames:
                    continue
                self.items.append({
                    "prompt": obj.get("prompt", "a video"),
                    "frames": frames,
                })
        self.num_frames = num_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = item["frames"]
        # take the first window (simple; could randomize)
        window = frames[: self.num_frames]
        imgs = []
        for p in window:
            import PIL.Image as Image
            import numpy as np
            im = Image.open(p).convert("RGB")
            arr = np.array(im, dtype=np.uint8)  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            imgs.append(t)
        video = torch.stack(imgs, dim=1)  # C, T, H, W
        prompt = item["prompt"]
        return {"video": video, "prompt": prompt}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(module: nn.Module, r: int, alpha: int, target_substrings: Tuple[str, ...]) -> int:
    """Replace Linear layers whose qualified name contains any of target_substrings."""
    count = 0
    for name, child in list(module.named_children()):
        qname = name
        if isinstance(child, nn.Linear) and any(s in qname for s in target_substrings):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, r, alpha, target_substrings)
    return count


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def get_text_conditioning(pipe, prompts, device: torch.device):
    """Return a dict with prompt_embeds, attention_mask, pooled_prompt_embeds.

    Tries several encode_prompt invocation styles and parses common return shapes.
    """
    call_args = dict(device=device)
    res = None
    if hasattr(pipe, "encode_prompt"):
        # Try to ask for pooled and mask explicitly if supported
        for kwargs in (
            {"prompt": prompts, "return_pooled": True, "return_attention_mask": True},
            {"prompt": prompts, "return_pooled": True},
            {"prompt": prompts},
            {"prompt": prompts, "device": device},  # explicit device already in call_args
        ):
            try:
                res = pipe.encode_prompt(**kwargs)
                break
            except TypeError:
                continue
        if res is None:
            # Positional fallback
            try:
                res = pipe.encode_prompt(prompts, device=device)
            except Exception:
                pass

    out = {"prompt_embeds": None, "attention_mask": None, "pooled_prompt_embeds": None}

    def assign_from_dict(d):
        keys_map = {
            "prompt_embeds": ("prompt_embeds", "text_embeds", "encoder_hidden_states", "embeds", "text_embeddings"),
            "pooled_prompt_embeds": ("pooled_prompt_embeds", "pooled_projections", "pooled_embeds", "pooled"),
            "attention_mask": ("attention_mask", "encoder_attention_mask", "text_attention_mask"),
        }
        for tgt, candidates in keys_map.items():
            for k in candidates:
                v = d.get(k, None)
                if isinstance(v, torch.Tensor):
                    out[tgt] = v
                    break

    if isinstance(res, dict):
        assign_from_dict(res)
    elif isinstance(res, (list, tuple)):
        # Heuristic: find 3D tensor as prompt_embeds, 2D as pooled, int/bool tensor as mask
        for x in res:
            if isinstance(x, torch.Tensor):
                if x.ndim == 3 and out["prompt_embeds"] is None:
                    out["prompt_embeds"] = x
                elif x.ndim == 2 and out["pooled_prompt_embeds"] is None:
                    out["pooled_prompt_embeds"] = x
                elif x.ndim == 2 and x.dtype in (torch.int64, torch.int32, torch.bool) and out["attention_mask"] is None:
                    out["attention_mask"] = x

    # Fallbacks
    if out["prompt_embeds"] is None:
        # As a last resort, if res is a single tensor, treat as embeds
        if isinstance(res, torch.Tensor):
            out["prompt_embeds"] = res
    if out["prompt_embeds"] is None:
        raise SystemExit("encode_prompt did not return recognizable prompt embeddings for this pipeline.")
    if out["attention_mask"] is None:
        # Create an all-ones mask of shape (B, L)
        B, L, _ = out["prompt_embeds"].shape
        out["attention_mask"] = torch.ones((B, L), dtype=torch.long, device=out["prompt_embeds"].device)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF model id (e.g., org/name) or local path")
    ap.add_argument("--dataset", default="data/dataset.jsonl", help="JSONL manifest from prepare_dataset.py")
    ap.add_argument("--output_dir", default="outputs/lora", help="Where to save LoRA weights")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--gradient_accumulation", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true", help="Trust custom pipeline code from model repo")
    ap.add_argument("--subfolder", default=None, help="Optional subfolder within the model repo/folder that contains model_index.json (e.g., 'diffusers')")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = device_select()
    print(f"Using device: {device}")

    # Lazy imports to allow repository to be cloned without env ready
    try:
        try:
            from diffusers import HunyuanVideoPipeline  # type: ignore
            PipeCls = HunyuanVideoPipeline
        except Exception:
            from diffusers import AutoPipelineForText2Video  # type: ignore
            PipeCls = AutoPipelineForText2Video
    except Exception:
        raise SystemExit("diffusers is required. Try: pip install 'diffusers>=0.30' 'transformers>=4.43' 'accelerate>=0.30' 'safetensors' 'Pillow'")

    # Resolve local folder layouts where the diffusers checkpoint may live in a subfolder.
    model_id = args.model_id
    local_root = Path(model_id)
    searched_subdir = None
    if local_root.exists() and local_root.is_dir() and not args.subfolder:
        root_has_model_index = (local_root / "model_index.json").exists()
        if not root_has_model_index:
            # Look for nested diffusers directory
            for sub in local_root.iterdir():
                if sub.is_dir() and (sub / "model_index.json").exists():
                    model_id = str(sub)
                    searched_subdir = sub
                    break
        # Helpful hint if we still don't see a diffusers layout
        if not (Path(model_id) / "model_index.json").exists():
            looks_transformers = (local_root / "config.json").exists()
            hint = (
                "No model_index.json found. If this is a Transformers-style checkpoint, "
                "you need a Diffusers-formatted pipeline (with model_index.json). If the repo contains a 'diffusers' subfolder, "
                "point --model_id to that subfolder."
            )
            raise SystemExit(hint)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        pipe = PipeCls.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=args.trust_remote_code,
            use_safetensors=True,
            subfolder=args.subfolder,
        )
    except Exception as e:
        msg = (
            "Failed to load model. Verify the model id exists on Hugging Face, that you've accepted the license (if required), "
            "and that you are logged in with a token that has access. Try: huggingface-cli login. Original error: " + str(e)
        )
        raise SystemExit(msg)
    pipe.to(device)
    pipe.enable_model_cpu_offload = False  # keep on-device if possible
    pipe.set_progress_bar_config(disable=True)

    # Use a DDPM scheduler locally for training noise addition (pipe may use FlowMatch without add_noise)
    try:
        from diffusers import DDPMScheduler  # type: ignore
    except Exception as e:
        raise SystemExit(f"diffusers DDPMScheduler not available: {e}")
    num_train_ts = getattr(getattr(pipe, "scheduler", None), "config", None)
    num_train_ts = getattr(num_train_ts, "num_train_timesteps", 1000)
    train_scheduler = DDPMScheduler(num_train_timesteps=num_train_ts)

    # Identify main video transform block. Depending on version it may be `transformer` or `unet`.
    backbone = getattr(pipe, "transformer", None)
    if backbone is None:
        backbone = getattr(pipe, "unet", None)
    if backbone is None:
        raise SystemExit("Could not locate video backbone (transformer/unet) in pipeline.")

    # Inject LoRA into common attention projections by name substring
    # You may adjust the list below if your model uses different names.
    targets = ("to_q", "to_k", "to_v", "to_out", "proj", "out_proj")
    replaced = inject_lora(backbone, r=args.rank, alpha=args.alpha, target_substrings=targets)
    print(f"Injected LoRA into {replaced} Linear layers")
    # Ensure newly added LoRA params are placed on the active device
    backbone.to(device)

    # Freeze everything else: iterate over pipeline components that are nn.Modules
    for comp_name, comp in getattr(pipe, "components", {}).items():
        if isinstance(comp, nn.Module):
            for p in comp.parameters():
                p.requires_grad = False
    for p in lora_parameters(backbone):
        p.requires_grad = True

    ds = FrameWindowDataset(args.dataset, num_frames=args.num_frames)
    if len(ds) == 0:
        raise SystemExit("Dataset is empty. Run scripts/prepare_dataset.py first.")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(lora_parameters(backbone), lr=args.lr)

    # Simple training loop with per-step accumulation
    accum = args.gradient_accumulation
    step = 0
    # Set train/eval modes appropriately
    backbone.train()
    # Keep non-trainable encoders/VAEs in eval to save memory
    if hasattr(pipe, "vae") and isinstance(pipe.vae, nn.Module):
        pipe.vae.eval()
    if hasattr(pipe, "text_encoder") and isinstance(getattr(pipe, "text_encoder"), nn.Module):
        pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and isinstance(getattr(pipe, "text_encoder_2"), nn.Module):
        pipe.text_encoder_2.eval()
    while step < args.max_steps:
        for batch in dl:
            videos = batch["video"].to(device)  # (B, C, T, H, W) after permute below
            prompts = batch["prompt"]

            # Resize and normalize to match pipe expected preprocessing
            videos = videos  # Already in [0,1]
            videos = videos * 2 - 1  # to [-1, 1]
            videos = videos.to(torch.float32)
            # Ensure spatial size matches desired resolution (no temporal resampling)
            B, C, T, H, W = videos.shape
            if (H != args.resolution) or (W != args.resolution):
                import torch.nn.functional as F
                v = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
                v = F.interpolate(v, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False)
                videos = v.reshape(B, T, C, args.resolution, args.resolution).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

            with torch.no_grad():
                # Encode text via public API
                cond = get_text_conditioning(pipe, prompts, device=device)
                # Encode video frames to latents (expects (B, C, T, H, W))
                latents = pipe.vae.encode(videos).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Diffusion training step (DDPM-style noise prediction)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, train_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

            # Forward through video transformer/unet
            guidance = torch.full((latents.shape[0],), 1.0, device=device, dtype=latents.dtype)
            pred = backbone(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond["prompt_embeds"],
                encoder_attention_mask=cond["attention_mask"],
                pooled_projections=cond.get("pooled_prompt_embeds", None),
                guidance=guidance,
            ).sample
            loss = torch.nn.functional.mse_loss(pred, noise) / accum
            loss.backward()

            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0:
                print(f"step {step}/{args.max_steps} loss={loss.item()*accum:.4f}")
            if step >= args.max_steps:
                break

    # Save LoRA weights only
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lora_state = {}
    for name, module in backbone.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    torch.save({
        "rank": args.rank,
        "alpha": args.alpha,
        "target_keys": list(lora_state.keys()),
        "state": lora_state,
    }, out / "lora_weights.pt")
    print(f"Saved LoRA weights to {out/'lora_weights.pt'}")


if __name__ == "__main__":
    main()
