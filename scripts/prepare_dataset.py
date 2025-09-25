#!/usr/bin/env python3
"""
Prepare a lightweight frame dataset from the MP4 videos in ./videos_finetune.

For each input video, extracts frames at a fixed FPS and scales the short side
to the target size while preserving aspect ratio. Outputs frames under
./data/frames/<video_stem>/*.jpg and writes a dataset manifest JSONL at
./data/dataset.jsonl with a default prompt derived from the filename.

Requires: ffmpeg available on PATH.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(shlex.split(cmd), check=True)


def normalize_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[\s_]+", " ", stem).strip()
    return stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", default="videos_finetune", help="Input videos directory")
    p.add_argument("--out_dir", default="data/frames", help="Output frames root dir")
    p.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    p.add_argument("--size", type=int, default=384, help="Target short-side size")
    p.add_argument("--limit", type=int, default=0, help="Optional limit of videos to process")
    args = p.parse_args()

    videos_dir = Path(args.videos_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("No MP4 files found in videos_finetune/")
        return

    manifest_path = out_root.parent / "dataset.jsonl"
    with manifest_path.open("w") as mf:
        for vid in videos:
            stem = vid.stem
            out_dir = out_root / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            # Scale short side to size, keep aspect ratio, ensure even dims
            vf = f"fps={args.fps},scale='if(lte(iw,ih),{args.size},-2)':'if(gt(iw,ih),{args.size},-2)':flags=lanczos"
            cmd = (
                f"ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(str(vid))} "
                f"-vf {shlex.quote(vf)} {shlex.quote(str(out_dir))}/%06d.jpg"
            )
            run(cmd)

            # Add to manifest with a simple, filename-derived prompt
            prompt = f"a video of {normalize_title(stem)}"
            entry = {
                "video_id": stem,
                "frames_dir": str(out_dir),
                "prompt": prompt,
            }
            mf.write(json.dumps(entry) + "\n")
            print(f"Wrote frames for {vid.name} -> {out_dir}")

    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()

