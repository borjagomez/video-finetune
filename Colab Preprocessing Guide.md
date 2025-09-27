# Colab Preprocessing Guide

Use these cells in Google Colab to standardize your videos and (optionally) extract frames for training.

## Cell 1 — Setup
```bash
!apt-get -y update >/dev/null 2>&1 && apt-get -y install ffmpeg >/dev/null 2>&1 || true
!ffmpeg -version | head -n 1
```

## Cell 2 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 3 — Prepare workspace paths
```python
from pathlib import Path

MOUNT = Path('/content/drive') / 'My Drive'
INPUT_DIR = MOUNT / 'videos_finetune'

# Persist outputs on Drive
OUTPUT_DIR = MOUNT / 'data' / 'videos_baseline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Input dir:', INPUT_DIR)
print('Output dir:', OUTPUT_DIR)
```

List your Drive video folder:
```bash
ls -lh "/content/drive/My Drive/videos_finetune" || true
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
```python
import subprocess
args = [
  'python','tools/preprocess_videos.py',
  '--input_dir', str(INPUT_DIR),
  '--output_dir', str(OUTPUT_DIR),
  '--target_height','1080',
  '--fps','30',
  '--compute_checksums',
]
print('Running:', ' '.join(args))
subprocess.run(args, check=True)
```

Outputs are in `/content/drive/My Drive/data/videos_baseline` and a JSONL manifest at `/content/drive/My Drive/data/videos_manifest.jsonl`.

## Cell 6 — Verify one file
```python
from pathlib import Path
import subprocess
vids = sorted(Path(OUTPUT_DIR).glob('*.mp4'))
print('Found outputs:', len(vids))
if vids:
    first = str(vids[0])
    print('Video:', first)
    subprocess.run(['ffprobe','-v','error','-show_streams','-select_streams','v:0', first])
```

## Cell 7 — Optional: extract frames for training
This uses the repo’s frame extractor to create a frames dataset and a dataset manifest.
```python
from pathlib import Path
import subprocess
FRAMES_DIR = MOUNT / 'data' / 'frames'
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
args = [
  'python','scripts/prepare_dataset.py',
  '--videos_dir', str(OUTPUT_DIR),
  '--out_dir', str(FRAMES_DIR),
  '--fps','2','--size','384'
]
print('Running:', ' '.join(args))
subprocess.run(args, check=True)
```

The frames manifest will be at `/content/drive/My Drive/data/dataset.jsonl`.
