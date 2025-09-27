#!/usr/bin/env python3
"""
Preprocess MP4 videos to a common baseline suitable for fine-tuning.

Features
- Validates decode with ffprobe and optional full read.
- Transcodes to H.264 (libx264) + AAC, YUV420p, faststart.
- Normalizes resolution to a target short-side while preserving aspect ratio.
- Optional FPS normalization.
- Writes a JSONL manifest with key metadata and checksums.

Usage
  python tools/preprocess_videos.py \
    --input_dir videos_finetune \
    --output_dir data/videos_baseline \
    --target_height 720 \
    --fps 30 \
    --compute_checksums

Requirements: ffmpeg/ffprobe available on PATH.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    return subprocess.run(shlex.split(cmd), check=check, capture_output=True, text=True)


def ffprobe_json(path: Path) -> Dict[str, Any]:
    cmd = (
        f"ffprobe -v error -hide_banner -print_format json "
        f"-show_streams -show_format {shlex.quote(str(path))}"
    )
    res = run(cmd)
    return json.loads(res.stdout or "{}")


def has_audio(probe: Dict[str, Any]) -> bool:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            return True
    return False


def first_stream(probe: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
    for s in probe.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return None


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def calc_scaled_dims(width: int, height: int, target_short: int) -> (int, int):
    if width <= 0 or height <= 0:
        return width, height
    if width < height:
        # width is short side
        scale = target_short / width
        w = target_short
        h = int(round(height * scale))
    else:
        scale = target_short / height
        h = target_short
        w = int(round(width * scale))
    return even(w), even(h)


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class Options:
    input_dir: Path
    output_dir: Path
    target_height: int
    fps: Optional[float]
    crf: int
    preset: str
    audio_bitrate: str
    validate_full_decode: bool
    compute_checksums: bool
    manifest_path: Optional[Path]


def preprocess_one(src: Path, dst: Path, opt: Options) -> Optional[Dict[str, Any]]:
    probe = ffprobe_json(src)
    v = first_stream(probe, "video")
    if not v:
        print(f"! Skipping (no video stream): {src}")
        return None

    # Calculate output dimensions
    in_w = int(v.get("width", 0))
    in_h = int(v.get("height", 0))
    out_w, out_h = calc_scaled_dims(in_w, in_h, opt.target_height)

    # Ensure output directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    vf_parts = [f"scale={out_w}:{out_h}:flags=lanczos"]
    if opt.fps and opt.fps > 0:
        vf_parts.append(f"fps={opt.fps}")
    vf = ",".join(vf_parts)

    audio_present = has_audio(probe)
    audio_args = "-c:a aac -b:a {ab} -ac 2".format(ab=opt.audio_bitrate) if audio_present else "-an"

    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(src))} "
        f"-map 0:v:0 {'-map 0:a:0' if audio_present else ''} "
        f"-vf {shlex.quote(vf)} "
        f"-c:v libx264 -preset {opt.preset} -crf {opt.crf} -pix_fmt yuv420p -movflags +faststart "
        f"{audio_args} "
        f"{shlex.quote(str(dst))}"
    )

    run(cmd)

    # Optional full-decode validation
    if opt.validate_full_decode:
        try:
            run(f"ffmpeg -v error -i {shlex.quote(str(dst))} -f null -")
        except subprocess.CalledProcessError as e:
            print(f"! Validation decode failed for {dst}: {e}")
            return None

    out_probe = ffprobe_json(dst)
    out_v = first_stream(out_probe, "video") or {}
    out_a = first_stream(out_probe, "audio") or {}

    size = dst.stat().st_size if dst.exists() else 0
    checksum = sha256sum(dst) if opt.compute_checksums and dst.exists() else None

    meta = {
        "original_path": str(src),
        "output_path": str(dst),
        "vcodec": out_v.get("codec_name"),
        "acodec": out_a.get("codec_name"),
        "width": int(out_v.get("width", 0)),
        "height": int(out_v.get("height", 0)),
        "fps": float(out_v.get("avg_frame_rate", "0/1").split("/")[0])
        if out_v.get("avg_frame_rate") and "/" in out_v["avg_frame_rate"] and out_v["avg_frame_rate"].split("/")[1] != "0"
        else None,
        "duration_sec": float(out_probe.get("format", {}).get("duration", 0.0)),
        "has_audio": audio_present,
        "size_bytes": size,
        "sha256": checksum,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Standardize videos for fine-tuning")
    ap.add_argument("--input_dir", default="videos_finetune", help="Directory with input .mp4 files")
    ap.add_argument("--output_dir", default="data/videos_baseline", help="Directory for processed videos")
    ap.add_argument("--target_height", type=int, default=1080, help="Short-side target height (e.g., 720 or 1080)")
    ap.add_argument("--fps", type=float, default=30.0, help="Normalize FPS (set <=0 to keep original)")
    ap.add_argument("--crf", type=int, default=20, help="x264 CRF quality (lower=better/bigger)")
    ap.add_argument("--preset", default="medium", help="x264 preset (e.g., slow, medium, fast)")
    ap.add_argument("--audio_bitrate", default="128k", help="AAC bitrate")
    ap.add_argument("--validate_full_decode", action="store_true", help="Run an extra pass to fully decode outputs")
    ap.add_argument("--compute_checksums", action="store_true", help="Compute SHA-256 for outputs")
    ap.add_argument("--manifest", default="data/videos_manifest.jsonl", help="Path to write JSONL manifest")
    ap.add_argument("--limit", type=int, default=0, help="Optionally limit number of files processed")
    args = ap.parse_args()

    opt = Options(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        target_height=int(args.target_height),
        fps=float(args.fps) if args.fps and args.fps > 0 else None,
        crf=int(args.crf),
        preset=str(args.preset),
        audio_bitrate=str(args.audio_bitrate),
        validate_full_decode=bool(args.validate_full_decode),
        compute_checksums=bool(args.compute_checksums),
        manifest_path=Path(args.manifest) if args.manifest else None,
    )

    in_dir = opt.input_dir
    out_dir = opt.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted([p for p in in_dir.glob("*.mp4") if p.is_file()])
    if args.limit:
        vids = vids[: args.limit]
    if not vids:
        print(f"No MP4 files found in {in_dir}")
        return

    manifest_f = None
    if opt.manifest_path:
        opt.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = opt.manifest_path.open("w")

    try:
        for i, src in enumerate(vids, 1):
            dst = out_dir / src.name
            print(f"[{i}/{len(vids)}] Processing {src.name}")
            meta = preprocess_one(src, dst, opt)
            if meta and manifest_f:
                manifest_f.write(json.dumps(meta) + "\n")
    finally:
        if manifest_f:
            manifest_f.close()

    if opt.manifest_path and opt.manifest_path.exists():
        print(f"Manifest written to: {opt.manifest_path}")
    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
