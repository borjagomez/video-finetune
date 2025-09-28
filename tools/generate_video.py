#!/usr/bin/env python3
"""
Generate a video using a HunyuanVideo/AutoPipelineForText2Video model with LoRA weights.

Features
- Loads base model pipeline from a local folder or HF id.
- Injects LoRA adapters into Linear layers (matching training script) and loads weights.
- Generates text-to-video with configurable frames, steps, guidance.
- Optionally integrates screenshots at specific timestamps via a beats JSON, or appends all screenshots at the end.

Usage (example)
  python tools/generate_video.py \
    --model_dir "/content/drive/My Drive/HunyuanVideo-diffusers" \
    --lora_weights "/content/drive/My Drive/outputs/lora/lora_weights.pt" \
    --prompt "A calm product walkthrough of the dashboard" \
    --screenshots "/content/drive/My Drive/screenshots/*.png" \
    --beats "/content/drive/My Drive/outputs/inference/beats.json" \
    --out "/content/drive/My Drive/outputs/generated.mp4" \
    --num_frames 48 --fps 24 --steps 30 --guidance 3.5

Requirements: diffusers, transformers, safetensors, Pillow, imageio-ffmpeg or ffmpeg CLI.
"""

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

# Ensure PyTorch allocator uses expandable segments unless already set
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --- LoRA injection utilities (mirrors train_lora_hunyuanvideo.py) ---

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
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(module: nn.Module, r: int, alpha: int, target_substrings: Tuple[str, ...]) -> int:
    count = 0
    for name, child in list(module.named_children()):
        qname = name
        if isinstance(child, nn.Linear) and any(s in qname for s in target_substrings):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, r, alpha, target_substrings)
    return count


def load_lora_weights(backbone: nn.Module, lora_ckpt: Path) -> Tuple[int, int]:
    ckpt = torch.load(lora_ckpt, map_location="cpu")
    state = ckpt.get("state", {})
    assigned = 0
    missing = 0
    name_to_module = {name: mod for name, mod in backbone.named_modules() if isinstance(mod, LoRALinear)}
    for k, v in state.items():
        # k example: "<layer>.lora_A.weight" or ".lora_B.weight"
        if ".lora_A.weight" in k:
            base_name = k.replace(".lora_A.weight", "")
            mod = name_to_module.get(base_name)
            if mod is not None:
                mod.lora_A.weight.data.copy_(v)
                assigned += 1
            else:
                missing += 1
        elif ".lora_B.weight" in k:
            base_name = k.replace(".lora_B.weight", "")
            mod = name_to_module.get(base_name)
            if mod is not None:
                mod.lora_B.weight.data.copy_(v)
                assigned += 1
            else:
                missing += 1
    return assigned, missing


# --- Helpers ---

def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(shlex.split(cmd), check=True)


def save_video_frames_ffmpeg(frames: List[Image.Image], out_path: Path, fps: int, crf: int = 18) -> None:
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        for i, im in enumerate(frames, 1):
            im = im.convert("RGB")
            im.save(temp / f"{i:06d}.png")
        vf = f"-framerate {fps} -i {shlex.quote(str(temp))}/%06d.png -c:v libx264 -pix_fmt yuv420p -crf {crf} -movflags +faststart"
        run(f"ffmpeg -y -hide_banner -loglevel error {vf} {shlex.quote(str(out_path))}")


def append_screenshots(out_video: Path, screenshots: List[Path], fps: int, duration_per: float, size: Tuple[int, int]) -> Path:
    """Create still segments from screenshots and concat to end of generated video."""
    w, h = size
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        parts: List[Path] = []
        # First segment is the generated video
        gen_mp4 = out_video
        parts.append(gen_mp4)
        # Build still segments
        for i, img in enumerate(screenshots, 1):
            seg = td_path / f"shot_{i:02d}.mp4"
            # Scale to fit, pad if needed
            vf = f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            run(
                f"ffmpeg -y -hide_banner -loglevel error -loop 1 -t {duration_per} -i {shlex.quote(str(img))} "
                f"-vf {shlex.quote(vf)} -r {fps} -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart {shlex.quote(str(seg))}"
            )
            parts.append(seg)
        # Concat
        concat_txt = td_path / "concat.txt"
        with concat_txt.open("w") as f:
            for p in parts:
                f.write(f"file '{p.as_posix()}'\n")
        final_out = out_video.with_name(out_video.stem + "_with_shots.mp4")
        run(
            f"ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i {shlex.quote(str(concat_txt))} -c copy {shlex.quote(str(final_out))}"
        )
        return final_out


def ffprobe_float(path: Path, key: str) -> Optional[float]:
    try:
        cmd = f"ffprobe -v error -print_format json -show_format {shlex.quote(str(path))}"
        res = subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
        info = json.loads(res.stdout)
        val = info.get("format", {}).get(key)
        return float(val) if val is not None else None
    except Exception:
        return None


def make_trim(in_path: Path, out_path: Path, ss: float, to: float, fps: int, size: Tuple[int, int]) -> None:
    w, h = size
    vf = f"scale={w}:{h}:flags=lanczos"
    run(
        f"ffmpeg -y -hide_banner -loglevel error -ss {ss} -to {to} -i {shlex.quote(str(in_path))} "
        f"-vf {shlex.quote(vf)} -r {fps} -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart {shlex.quote(str(out_path))}"
    )


def make_still_segment(img: Path, out_path: Path, duration: float, fps: int, size: Tuple[int, int]) -> None:
    w, h = size
    vf = f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    run(
        f"ffmpeg -y -hide_banner -loglevel error -loop 1 -t {duration} -i {shlex.quote(str(img))} "
        f"-vf {shlex.quote(vf)} -r {fps} -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart {shlex.quote(str(out_path))}"
    )


def integrate_with_beats(base_video: Path, beats: List[dict], fps: int, size: Tuple[int, int]) -> Path:
    """Insert screenshot still segments into the base video at specific timestamps.

    beats: list of {"time": seconds, "duration": seconds, "screenshot": "/path/img.png"}
    Returns a new MP4 path with segments inserted.
    """
    dur = ffprobe_float(base_video, "duration") or 0.0
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        beats_sorted = sorted(beats, key=lambda x: float(x.get("time", 0)))
        # sanitize
        events = []
        for b in beats_sorted:
            t = max(0.0, float(b.get("time", 0.0)))
            d = max(0.0, float(b.get("duration", 0.0)))
            img = Path(str(b.get("screenshot", "")))
            if d <= 0 or not img.exists() or t >= dur:
                continue
            events.append({"time": t, "duration": d, "screenshot": img})

        parts: List[Path] = []
        cur = 0.0
        idx = 0
        for e in events:
            t = float(e["time"])
            d = float(e["duration"])
            img = Path(e["screenshot"]) 
            if t > cur:
                seg = td_path / f"part_{idx:03d}.mp4"; idx += 1
                make_trim(base_video, seg, cur, t, fps=fps, size=size)
                parts.append(seg)
                cur = t
            shot = td_path / f"shot_{idx:03d}.mp4"; idx += 1
            make_still_segment(img, shot, duration=d, fps=fps, size=size)
            parts.append(shot)
            cur = t + d
        if cur < dur:
            tail = td_path / f"part_{idx:03d}.mp4"; idx += 1
            make_trim(base_video, tail, cur, dur, fps=fps, size=size)
            parts.append(tail)

        concat_txt = td_path / "concat.txt"
        with concat_txt.open("w") as f:
            for p in parts:
                f.write(f"file '{p.as_posix()}'\n")
        final_out = base_video.with_name(base_video.stem + "_with_beats.mp4")
        run(
            f"ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i {shlex.quote(str(concat_txt))} "
            f"-c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart {shlex.quote(str(final_out))}"
        )
        return final_out


@dataclass
class GenArgs:
    model_dir: Path
    lora_weights: Path
    prompt: str
    out: Path
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    guidance: float
    seed: Optional[int]
    screenshots_glob: Optional[str]
    screenshot_duration: float
    beats_json: Optional[Path]
    beats_from_prompt: bool
    shots_base_dir: Optional[Path]
    dtype: Optional[str]
    enable_cpu_offload: bool
    enable_attention_slicing: bool
    enable_vae_slicing: bool
    enable_vae_tiling: bool


def load_pipeline(model_dir: Path, torch_dtype: Optional[torch.dtype] = None, move_to_device: bool = True):
    try:
        from diffusers import HunyuanVideoPipeline  # type: ignore
        Pipe = HunyuanVideoPipeline
    except Exception:
        from diffusers import AutoPipelineForText2Video  # type: ignore
        Pipe = AutoPipelineForText2Video

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    # subfolder auto-detect if model_index.json not at root
    model_id = model_dir
    if not (model_dir / "model_index.json").exists():
        for sub in model_dir.iterdir():
            if sub.is_dir() and (sub / "model_index.json").exists():
                model_id = sub
                break
    kwargs = dict(trust_remote_code=True, use_safetensors=True, token=token)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    pipe = Pipe.from_pretrained(str(model_id), **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # Only move to device here when requested. If CPU offload is desired later,
    # we should avoid moving the full model to GPU up front.
    if move_to_device:
        if torch_dtype is not None:
            pipe.to(device, dtype=torch_dtype)
        else:
            pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe, device


def apply_lora(pipe, lora_path: Path) -> None:
    # find backbone
    backbone = getattr(pipe, "transformer", None)
    if backbone is None:
        backbone = getattr(pipe, "unet", None)
    if backbone is None:
        raise SystemExit("Could not locate video backbone (transformer/unet) in pipeline.")

    # Capture current device of backbone before modification so we can restore
    try:
        current_device = next(backbone.parameters()).device
    except Exception:
        current_device = getattr(pipe, "device", torch.device("cpu"))

    # Use default r/alpha from saved ckpt if present
    ck = torch.load(lora_path, map_location="cpu")
    r = int(ck.get("rank", 4))
    alpha = int(ck.get("alpha", 8))
    replaced = inject_lora(backbone, r=r, alpha=alpha, target_substrings=("to_q", "to_k", "to_v", "to_out", "proj", "out_proj"))
    assigned, missing = load_lora_weights(backbone, lora_path)
    print(f"Injected {replaced} LoRA layers; loaded {assigned} tensors (missing: {missing})")

    # Ensure newly added LoRA modules are on the same device as the backbone
    try:
        backbone.to(current_device)
    except Exception:
        pass


def generate(pipe, prompt: str, width: int, height: int, num_frames: int, steps: int, guidance: float, seed: Optional[int], autocast_dtype: Optional[torch.dtype] = None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    # Try robust call styles
    tried = []
    def attempt(kwargs):
        tried.append(list(kwargs.keys()))
        return pipe(**kwargs)
    out = None
    errors = []
    for kwargs in [
        dict(prompt=prompt, num_frames=num_frames, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, generator=generator),
        dict(prompt=prompt, num_frames=num_frames, height=height, width=width, num_inference_steps=steps, generator=generator),
        dict(prompt=prompt, num_frames=num_frames, height=height, width=width, guidance_scale=guidance, generator=generator),
        dict(prompt=prompt, num_frames=num_frames, height=height, width=width, generator=generator),
        dict(prompt=prompt, height=height, width=width, generator=generator),
        dict(prompt=prompt, generator=generator),
    ]:
        try:
            if autocast_dtype is not None:
                device_type = pipe.device.type if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
                with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    out = attempt(kwargs)
            else:
                with torch.inference_mode():
                    out = attempt(kwargs)
            break
        except TypeError as e:
            errors.append(str(e))
            continue
    if out is None:
        # final positional fallback
        try:
            if autocast_dtype is not None:
                device_type = pipe.device.type if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
                with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    out = pipe(prompt, height=height, width=width, num_frames=num_frames)
            else:
                with torch.inference_mode():
                    out = pipe(prompt, height=height, width=width, num_frames=num_frames)
        except Exception as e:
            msg = "All call patterns failed. Errors: " + " | ".join(errors) + f" Last: {e}"
            raise SystemExit(msg)

    # Diffusers typically returns an object with .frames or .images
    frames = None
    if hasattr(out, "frames") and out.frames:
        frames = out.frames[0]
    elif hasattr(out, "images") and out.images:
        frames = out.images
    if frames is None:
        raise SystemExit("Pipeline did not return frames/images; update diffusers or pipeline class.")
    # Ensure PIL Image list
    frames = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames]
    return frames


# --- Prompt tag parser ---

def _parse_seconds(val: str) -> Optional[float]:
    try:
        val = val.strip().lower()
        if val.endswith("s"):
            val = val[:-1]
        # support mm:ss
        if ":" in val:
            parts = val.split(":")
            if len(parts) == 2:
                m, s = parts
                return float(m) * 60.0 + float(s)
        return float(val)
    except Exception:
        return None


def parse_prompt_beats(prompt: str, shots_base_dir: Optional[Path]) -> Tuple[str, List[dict]]:
    """Parse [SHOT ...] tags from the prompt and return (clean_prompt, beats).

    Supported forms within a tag (key aliases):
      - time | at = seconds (e.g., 6.5, 6s, 1:23)
      - duration | dur = seconds
      - screenshot | file | path = path/to/image (quoted or unquoted)

    Example tag: [SHOT at=6.5 file="dashboard.png" dur=2.0]
    """
    beats: List[dict] = []

    def resolve_path(p: str) -> Path:
        pp = Path(p)
        if not pp.is_absolute() and shots_base_dir is not None:
            pp = shots_base_dir / pp
        return pp

    tag_re = re.compile(r"\[SHOT\s+([^\]]+)\]", re.IGNORECASE)

    def parse_attrs(attr_str: str) -> Optional[dict]:
        # First, try robust regex extraction that tolerates spaces in unquoted values
        def rx(key_aliases: str):
            return re.search(rf"(?i)(?:{key_aliases})\s*=\s*(.+?)(?=(?:\s+[a-zA-Z_]+\s*=)|$)", attr_str)

        def clean(v: str) -> str:
            return v.strip().strip('"').strip("'")

        t_m = rx("time|at")
        d_m = rx("duration|dur")
        f_m = rx("screenshot|file|path")
        t = clean(t_m.group(1)) if t_m else None
        d = clean(d_m.group(1)) if d_m else None
        f = clean(f_m.group(1)) if f_m else None

        # Fallback to tokenization for simple cases
        if not (t and d and f):
            try:
                tokens = shlex.split(attr_str)
            except Exception:
                tokens = attr_str.split()
            data = {}
            for tok in tokens:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                k = k.strip().lower()
                v = clean(v)
                data[k] = v
            t = t or data.get("time") or data.get("at")
            d = d or data.get("duration") or data.get("dur")
            f = f or data.get("screenshot") or data.get("file") or data.get("path")

        t_sec = _parse_seconds(t) if t is not None else None
        d_sec = _parse_seconds(d) if d is not None else None
        if t_sec is None or d_sec is None or not f:
            return None
        p = resolve_path(f)
        return {"time": float(t_sec), "duration": float(d_sec), "screenshot": str(p)}

    clean_prompt = prompt
    for m in tag_re.finditer(prompt):
        attrs = m.group(1)
        beat = parse_attrs(attrs)
        if beat is not None:
            beats.append(beat)
        # remove the tag from the clean prompt
        clean_prompt = clean_prompt.replace(m.group(0), " ")

    # normalize spacing
    clean_prompt = re.sub(r"\s+", " ", clean_prompt).strip()
    return clean_prompt, beats


def main():
    ap = argparse.ArgumentParser(description="Generate a video with LoRA weights and optional screenshot segments")
    ap.add_argument("--model_dir", required=True, help="Path to base model folder (diffusers format)")
    ap.add_argument("--lora_weights", required=True, help="Path to lora_weights.pt")
    ap.add_argument("--prompt", required=True, help="Text prompt for generation")
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--screenshots", dest="screenshots_glob", default=None, help="Glob for screenshot images to append (used if --beats not provided)")
    ap.add_argument("--screenshot_duration", type=float, default=1.5, help="Seconds per screenshot segment (when appending or creating stills for beats)")
    ap.add_argument("--beats", dest="beats_json", default=None, help="JSON file describing insertion beats: [{time, duration, screenshot}]")
    ap.add_argument("--dtype", default="auto", choices=["auto","float16","float32","bfloat16"], help="Inference dtype to reduce memory (float16/bfloat16 recommended on CUDA)")
    ap.add_argument("--enable_cpu_offload", action="store_true", help="Enable model CPU offload to save VRAM")
    ap.add_argument("--enable_attention_slicing", action="store_true", help="Enable attention slicing to save memory")
    ap.add_argument("--enable_vae_slicing", action="store_true", help="Enable VAE slicing if supported")
    ap.add_argument("--enable_vae_tiling", action="store_true", help="Enable VAE tiling if supported")
    ap.add_argument("--beats_from_prompt", action="store_true", help="Parse [SHOT ...] tags from the prompt to build beats if --beats not provided")
    ap.add_argument("--shots_base_dir", default=None, help="Base directory to resolve relative screenshot paths in prompt tags")
    args = ap.parse_args()

    g = GenArgs(
        model_dir=Path(args.model_dir),
        lora_weights=Path(args.lora_weights),
        prompt=args.prompt,
        out=Path(args.out),
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        screenshots_glob=args.screenshots_glob,
        screenshot_duration=args.screenshot_duration,
        beats_json=Path(args.beats_json) if args.beats_json else None,
        beats_from_prompt=bool(args.beats_from_prompt),
        shots_base_dir=Path(args.shots_base_dir) if args.shots_base_dir else None,
        dtype=args.dtype if args.dtype and args.dtype != "auto" else None,
        enable_cpu_offload=bool(args.enable_cpu_offload),
        enable_attention_slicing=bool(args.enable_attention_slicing),
        enable_vae_slicing=bool(args.enable_vae_slicing),
        enable_vae_tiling=bool(args.enable_vae_tiling),
    )

    torch_dtype = None
    autocast_dtype = None
    if g.dtype == "float16":
        torch_dtype = torch.float16
        autocast_dtype = torch.float16
    elif g.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif g.dtype == "float32":
        torch_dtype = torch.float32

    # If CPU offload is requested, avoid moving to GPU in load_pipeline
    pipe, device = load_pipeline(g.model_dir, torch_dtype=torch_dtype, move_to_device=not g.enable_cpu_offload)

    # Memory-saving options
    try:
        if g.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if g.enable_cpu_offload:
            if hasattr(pipe, "enable_model_cpu_offload"):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    if hasattr(pipe, "enable_sequential_cpu_offload"):
                        pipe.enable_sequential_cpu_offload()
        else:
            # If not offloading, ensure the pipe is on the intended device
            if torch_dtype is not None:
                pipe.to(device, dtype=torch_dtype)
            else:
                pipe.to(device)
        if g.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if g.enable_vae_tiling and hasattr(getattr(pipe, "vae", None), "enable_tiling"):
            pipe.vae.enable_tiling()
        # Try xformers if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    except Exception:
        pass
    apply_lora(pipe, g.lora_weights)

    # Parse beats from prompt if requested and not using explicit beats JSON
    parsed_beats: List[dict] = []
    clean_prompt = g.prompt
    if g.beats_json is None and g.beats_from_prompt:
        clean_prompt, parsed_beats = parse_prompt_beats(g.prompt, g.shots_base_dir)
        if parsed_beats:
            print(f"Parsed {len(parsed_beats)} beats from prompt tags.")

    frames = generate(pipe, clean_prompt, g.width, g.height, g.num_frames, g.steps, g.guidance, g.seed, autocast_dtype=autocast_dtype)
    g.out.parent.mkdir(parents=True, exist_ok=True)
    save_video_frames_ffmpeg(frames, g.out, fps=g.fps)
    print(f"Saved generated video to {g.out}")

    if g.beats_json and g.beats_json.exists():
        try:
            beats = json.loads(Path(g.beats_json).read_text())
            if isinstance(beats, list) and beats:
                final = integrate_with_beats(g.out, beats, fps=g.fps, size=(g.width, g.height))
                print(f"Saved video with integrated beats to {final}")
            else:
                print("Beats JSON is empty or invalid list; skipping integration.")
        except Exception as e:
            print(f"Failed to apply beats: {e}")
    elif parsed_beats:
        try:
            beats = []
            for b in parsed_beats:
                # Keep only events with existing screenshot files
                p = Path(b["screenshot"]) if isinstance(b.get("screenshot"), str) else None
                if p and p.exists():
                    beats.append({"time": float(b["time"]), "duration": float(b["duration"]), "screenshot": str(p)})
            if beats:
                final = integrate_with_beats(g.out, beats, fps=g.fps, size=(g.width, g.height))
                print(f"Saved video with integrated beats (from prompt) to {final}")
            else:
                print("No valid screenshot files found from prompt tags; skipping integration.")
        except Exception as e:
            print(f"Failed to apply prompt-derived beats: {e}")
    elif g.screenshots_glob:
        shots = [Path(p) for p in glob.glob(g.screenshots_glob)]
        shots = [p for p in shots if p.exists()]
        if shots:
            final = append_screenshots(g.out, shots, fps=g.fps, duration_per=g.screenshot_duration, size=(g.width, g.height))
            print(f"Saved combined video with screenshots to {final}")
        else:
            print("No screenshots matched the glob; skipping append.")


if __name__ == "__main__":
    main()
