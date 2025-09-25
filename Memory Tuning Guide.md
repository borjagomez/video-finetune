Controlling Training Memory Usage (CPU/GPU)

- Goal: reduce RAM/VRAM usage even if training slows down. Apply the smallest set of changes that stabilize runs, then iterate.

**Quick Wins (Try First)**
- Smaller micro‑batch: set `batch_size=1` (or lower) and use gradient accumulation to keep the effective batch size. Example effective batch 8 → `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.
- Mixed precision: enable `fp16` or `bf16` (prefer `bf16` on newer GPUs). Cuts activation memory 2× with minimal accuracy loss.
- Gradient checkpointing: turn on to trade compute for ~25–40% activation memory savings. Disable model cache: `model.config.use_cache = False`.
- Downscale the video: train at 720p (or 480p) and/or reduce FPS. Example: `ffmpeg -i in.mp4 -vf scale=-1:720,fps=8 -c:v libx264 -crf 20 -c:a aac -b:a 128k out.mp4`.
- Reduce clip length: fewer frames per training sample (e.g., 8–16 instead of 32–64) significantly lowers memory.
- Stream data: avoid loading whole datasets into RAM; decode on the fly.

**Video‑Specific Memory Levers**
- Resolution: 1080p → 720p (or 480p) typically halves (or quarters) memory per clip.
- Frames per clip: e.g., 8–16 frames at 8–12 FPS for prototyping; only increase after stabilizing.
- Decode on demand: use lazy decoding (e.g., decord, torchvision.io) instead of preloading frames.
- Fewer decode threads: too many workers/threads duplicates buffers. Start with 2 workers, 1–2 decode threads.

**PyTorch / Trainer Knobs**
- Batch/accumulation:
  - `per_device_train_batch_size=1` (or 2) + `gradient_accumulation_steps` to reach target effective batch.
- Precision:
  - Enable AMP: `fp16=True` or `bf16=True` (Trainer) or `torch.autocast` (manual loop). Prefer `bf16` if supported.
- Activations:
  - `gradient_checkpointing=True` and `model.config.use_cache=False`.
- Optimizer memory:
  - Use `adamw_bnb_8bit` (bitsandbytes) or `adafactor` to shrink optimizer state. For LoRA/QLoRA, train fewer params.
- DataLoader memory:
  - `dataloader_pin_memory=False` to drop host pinned buffers.
  - `dataloader_num_workers=2` (start low) and `dataloader_prefetch_factor=2`.
  - Avoid `keep_in_memory=True` with Hugging Face datasets. Prefer `streaming=True`.
- CPU RAM footprint:
  - Limit threads: set env `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
  - `torch.multiprocessing.set_sharing_strategy('file_system')` to reduce shared‑memory pressure.
- Cleanup:
  - Delete large refs between steps: `del batch, loss`; then `torch.cuda.empty_cache()` (helps allocator reuse, not OS RAM).

**Hugging Face Trainer Example**

Use these to cut memory quickly (adjust to taste):

```
TrainingArguments(
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,
  fp16=True,                       # or bf16=True if supported
  gradient_checkpointing=True,
  dataloader_num_workers=2,
  dataloader_prefetch_factor=2,
  dataloader_pin_memory=False,
  optim="adamw_bnb_8bit",         # requires bitsandbytes; else use "adafactor"
  max_grad_norm=1.0,
)

# Before training
model.config.use_cache = False
```

For QLoRA fine‑tuning (optionally) set 4‑bit base weights and train LoRA adapters to greatly reduce VRAM.

**DeepSpeed (Optional) Minimal Config**

Offload optimizer state and shard parameters to cut GPU memory. Example `ds_config.json`:

```
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": false }
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false }
}
```

Use stage 2 for simplicity. Stage 3 further reduces memory by partitioning parameters but adds complexity.

**Accelerate (Optional) Offload Sketch**

- Configure: `accelerate config` → choose `cpu` offload for optimizer and/or params.
- Keep micro‑batch small, rely on accumulation. Enable `mixed_precision=bf16` or `fp16`.

**Data Streaming Patterns**
- Hugging Face Datasets: `load_dataset(..., streaming=True)`; map a transform that decodes only the frames you need per sample.
- WebDataset/Shard files: iterate tar shards sequentially; avoid materializing entire shards in memory.
- Avoid global caches of decoded frames. If caching is required, use small, bounded LRU caches.

**Preprocessing to Shrink Data (ffmpeg)**
- Downscale to 720p and cap FPS:
  - `ffmpeg -i input.mp4 -vf scale=-1:720,fps=8 -c:v libx264 -preset slow -crf 20 -c:a aac -b:a 128k output.mp4`
- Crop/pad consistently if your model expects fixed sizes: add `-vf crop=...` or `scale` then `pad`.

**Debugging Memory**
- GPU: watch `nvidia-smi -l 1` while ramping batch size; find the largest stable micro‑batch.
- CPU RAM: monitor with `htop` and DataLoader worker counts; if RAM climbs with workers, reduce `num_workers` and disable `pin_memory`.
- PyTorch summaries: `torch.cuda.memory_summary()` after a step to see allocator view.
- Identify phase: if OOM occurs during forward → activations too large (lower frames/resolution/precision). If during optimizer step → optimizer state too large (switch optimizer or LoRA/QLoRA).

**Suggested Order of Changes**
- Step 1: `batch_size=1`, `gradient_accumulation_steps=8`, mixed precision on.
- Step 2: enable gradient checkpointing and `use_cache=False`.
- Step 3: reduce video resolution/FPS and clip length.
- Step 4: lower DataLoader workers, disable pin_memory, enable streaming.
- Step 5: switch to 8‑bit optimizers or LoRA/QLoRA.
- Step 6: consider DeepSpeed ZeRO‑2 offload if still constrained.

If you share your exact framework (PyTorch/Trainer/Accelerate/DeepSpeed) and current batch/clip settings, we can tailor a drop‑in config.

