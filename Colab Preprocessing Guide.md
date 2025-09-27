# Colab Preprocessing Guide

Use these cells in Google Colab to standardize your videos and (optionally) extract frames for training.

## Cell 1 — Setup
```bash
!apt-get -y update >/dev/null 2>&1 && apt-get -y install ffmpeg >/dev/null 2>&1 || true
!ffmpeg -version | head -n 1
```

## Cell 2 — Mount Drive (optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3 — Prepare workspace paths
```python
from pathlib import Path

WORKDIR = Path('/content')
INPUT_DIR = WORKDIR / 'videos_finetune'      # put your .mp4 files here
OUTPUT_DIR = WORKDIR / 'data' / 'videos_baseline'
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Input dir:', INPUT_DIR)
print('Output dir:', OUTPUT_DIR)
```

You can upload files via the Colab file browser or copy from Drive, e.g.:
```bash
!cp -v \
  /content/drive/MyDrive/path/to/your_videos/*.mp4 \
  /content/videos_finetune/
```

## Cell 4 — Fetch repo files (script only)
If you haven’t synced this repo into Colab, create the preprocessing script cell from the repo version (optional if you uploaded the whole repo):
```bash
%%bash
set -euo pipefail
mkdir -p /content/tools
cat > /content/tools/preprocess_videos.py << 'PY'
"""
Lightweight wrapper: if you’re using this project repo directly,
prefer running tools/preprocess_videos.py from the repo checkout.
"""
import sys, shutil
from pathlib import Path

repo_script = Path('/content/tools_repo/preprocess_videos.py')
local_script = Path('/content/tools/preprocess_videos.py')
if repo_script.exists():
  shutil.copy2(repo_script, local_script)
print(local_script)
PY
python /content/tools/preprocess_videos.py --help || true
```

Alternatively, if you have the repo in Colab already, just use:
```bash
!python tools/preprocess_videos.py --help
```

## Cell 5 — Run preprocessing
```bash
!python tools/preprocess_videos.py \
  --input_dir "/content/videos_finetune" \
  --output_dir "/content/data/videos_baseline" \
  --target_height 1080 \
  --fps 30 \
  --compute_checksums
```

Outputs are in `/content/data/videos_baseline` and a JSONL manifest at `/content/data/videos_manifest.jsonl`.

## Cell 6 — Verify one file
```bash
VID="$(ls -1 /content/data/videos_baseline/*.mp4 | head -n1)"
echo "$VID"
ffprobe -v error -show_streams -select_streams v:0 "$VID" | sed -n '1,40p'
```

## Cell 7 — Optional: extract frames for training
This uses the repo’s frame extractor to create a frames dataset and a dataset manifest.
```bash
!python scripts/prepare_dataset.py \
  --videos_dir "/content/data/videos_baseline" \
  --out_dir "/content/data/frames" \
  --fps 2 --size 384
```

The frames manifest will be at `/content/data/dataset.jsonl`.
