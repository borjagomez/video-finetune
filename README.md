# video-finetune

Quick start (Apple Silicon, M2 Max)

- Prepare frames: `python scripts/prepare_dataset.py --fps 2 --size 384`
- Install deps: `pip install -r requirements.txt`
- Train LoRA (very small run):
  1) Pick the correct Hugging Face model id for HunyuanVideo (ensure access/license accepted), e.g. `org/model`.
  2) Run: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_lora_hunyuanvideo.py --model_id <org/model> --trust_remote_code --rank 4 --alpha 8 --batch_size 1 --max_steps 200`

Notes

- Training an 11B video model locally is constrained. Keep resolution, frames, and rank small.
- The training script uses MPS if available and saves only LoRA weights to `outputs/lora/`.
- If loading fails with 404, confirm the model id on Hugging Face, accept the model license, and run `huggingface-cli login`. You can also use a local path in `--model_id`.

Local model checkpoints

- You can download the model locally (e.g., `models/tencent-HunyuanVideo`). Some repos place the Diffusers pipeline in a subfolder (commonly `diffusers/`).
- If you see `no file named model_index.json`, point `--model_id` to the subfolder that contains `model_index.json`, for example:
  `--model_id models/tencent-HunyuanVideo/diffusers`

Preprocess videos (standardize) and Colab

- Standardize videos to H.264/AAC and 1080p/30fps: `python tools/preprocess_videos.py --input_dir videos_finetune --output_dir data/videos_baseline --target_height 1080 --fps 30 --compute_checksums`
- A copy‑paste Colab workflow is in `Colab Preprocessing Guide.md`.
- A ready-to-run Colab notebook is included: `Colab Preprocessing Guide.ipynb`.
