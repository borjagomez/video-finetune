# Repository Guidelines

## Project Structure & Module Organization
- `videos_finetune/`: Source assets (MP4 videos) used for fine‑tuning workflows.
- `README.md`: Short project intro. No code modules or tests exist yet.
- Keep new non-video docs at the repo root (e.g., usage notes, manifests).

## Build, Test, and Development Commands
This repo stores binary assets; there is no build system. Use these local tools to inspect and standardize files:
- Inspect metadata: `ffprobe -v error -show_streams -select_streams v:0 "videos_finetune/Your Video.mp4"`
- Quick listing: `ls -lh videos_finetune/` and `rg -n "\.mp4$" videos_finetune`
- Transcode to a common baseline (example):
  `ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 20 -c:a aac -b:a 128k output.mp4`

## Coding Style & Naming Conventions
There is no application code. For assets and docs:
- Filenames: preserve existing names. For new files, prefer human‑readable Title Case, spaces allowed, `.mp4` extension. Example: `videos_finetune/Project Planning Overview.mp4`.
- Avoid non‑ASCII punctuation when possible (use `-` instead of em dashes) and keep names stable after merge.
- If adding helper scripts later, use 2‑space indentation for YAML, 4‑space for Python; name scripts descriptively (e.g., `tools/validate_videos.py`).

## Testing Guidelines (Asset Validation)
- Each added video should: decode without errors, use H.264 video and AAC audio, and target 1080p or 720p.
- Verify with `ffprobe` and include duration, resolution, codecs in the PR description.
- Optional: include a checksum file (e.g., `sha256sum "videos_finetune/…"`).

## Commit & Pull Request Guidelines
- Use Conventional Commits for clarity: `feat: add Spaces walkthrough video`, `chore: rename asset for clarity`.
- PRs must include: purpose, list of added/changed files, basic metadata (duration, resolution), and any processing steps (e.g., ffmpeg command used).
- Do not rename or delete existing assets unless necessary; explain impact and downstream consumers.
- Large binaries are expected. If introducing many or very large files, consider proposing Git LFS (`git lfs track "*.mp4"`) in a separate PR.

## Security & Privacy
- Ensure videos contain no sensitive data (credentials, PII beyond intended demos) and that you have rights to include the media.
- Do not commit secrets, API keys, or private URLs in filenames or docs.

