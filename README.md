# LTX-2.5 Runpod Serverless Endpoint

> [**Lightricks/LTX-2.5**](https://huggingface.co/Lightricks/LTX-2.5) — a 22B DiT
> that generates video **and** synchronised audio in one pass — deployed as a
> cost-efficient **Runpod Serverless** endpoint with network-volume weight
> caching. Pay nothing while idle; skip the download on every cold start after
> the first.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Developer Workflow                             │
│                                                                         │
│  git push → main                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────┐     build & push      ┌──────────────────────┐        │
│  │  GitHub     │ ──────────────────▶   │  ghcr.io             │        │
│  │  Actions    │                       │  (container registry)│        │
│  └─────────────┘                       └──────────┬───────────┘        │
└────────────────────────────────────────────────────┼───────────────────┘
                                                     │ image pull (on deploy)
                                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           Runpod Platform                              │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Serverless Endpoint  (min=0 workers, max=3 workers)             │  │
│  │                                                                  │  │
│  │  Cold start sequence (once per new worker):                      │  │
│  │    1. Mount network volume at /runpod-volume                     │  │
│  │    2. ensure_weights_present()                                   │  │
│  │       ├─ [FIRST TIME ONLY] download ~66 GB from Hugging Face     │  │
│  │       └─ [SUBSEQUENT]      files already on volume → skip        │  │
│  │    3. load_pipeline()  → load bfloat16 model into GPU VRAM       │  │
│  │                                                                  │  │
│  │  Warm request:                                                   │  │
│  │    1. Validate JSON input (Pydantic)                             │  │
│  │    2. Run LTX-2.5 inference (video + audio)                       │  │
│  │    3. Encode MP4 (libx264 + audio mux) → upload to S3/R2          │  │
│  │    4. Return {status, video_url, generation_time_seconds, …}     │  │
│  │                                                                  │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  Network Volume  (/runpod-volume)                        │    │  │
│  │  │   /models/ltx-2.5/   ← weights live here permanently   │    │  │
│  │  │   /.cache/huggingface/ ← HF metadata cache             │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ltx25-runpod-serverless/
├── .github/
│   └── workflows/
│       └── docker-build-push.yml   # CI/CD: build + push to ghcr.io
├── src/
│   ├── handler.py        # Runpod serverless entrypoint
│   ├── model_loader.py   # volume weight check + HF download + pipeline load
│   ├── inference.py      # text2video / image2video / flf2video logic
│   └── schema.py         # Pydantic input/output schemas
├── docker/
│   └── Dockerfile        # single-stage, CUDA 12.8.1 + PyTorch 2.8.0, no weights
├── scripts/
│   └── download_weights.py   # standalone pre-population script
├── tests/
│   ├── test_schema_standalone.py  # schema-only, needs just pydantic
│   └── test_handler_local.py      # handler unit tests + opt-in integration test
├── .dockerignore
├── .gitignore
├── .env.example          # copy to .env, fill in real values
├── requirements.txt
├── runpod.toml           # endpoint config reference (not parsed by Runpod)
└── README.md
```

---

## Quick Start (Zero to Live Endpoint)

### Prerequisites

- GitHub account (for GHCR)
- Runpod account with billing set up
- Hugging Face account with a **read token** that has **"read access to gated repos"** scope
- Access granted to [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) on Hugging Face

---

### Step 1 — Fork / clone this repo

```bash
git clone https://github.com/<YOUR_ORG>/ltx25-runpod-serverless.git
cd ltx25-runpod-serverless
cp .env.example .env
# fill in HF_TOKEN and S3 credentials in .env
```

---

### Step 2 — Create a Runpod Network Volume

1. Go to **Runpod Console → Storage → Network Volumes → Create**.
2. **Name:** `ltx25-weights` (or any name).
3. **Size:** ≥ 100 GB. The six files this endpoint actually needs total **~66 GB**:
   the 22B distilled transformer (39.1 GB), the Gemma-4-12B text encoder with
   projections (24.5 GB), the video VAE (1.4 GB), the audio VAE (0.34 GB), the
   ×2 latent spatial upscaler (0.93 GB) and the duration head (3.7 MB).
   `download_weights.py` fetches only those — the full `Lightricks/LTX-2.5` repo
   is ~186 GB because it also carries the dev, nvfp4 and ComfyUI-int8 variants,
   the 450 LoRA and the temporal upscaler, none of which this handler loads.
   Size the volume ≥ 100 GB so a future checkpoint revision still fits.
4. **Region:** same region you'll deploy your endpoint (latency matters for cold start).
5. Note the **Volume ID** — you'll need it in Step 4.

> **Tip:** Pre-populate the volume cheaply by spinning up a CPU Pod (or cheap GPU Pod), attaching the volume, and running:
> ```bash
> HF_TOKEN=hf_xxx RUNPOD_VOLUME_PATH=/runpod-volume python scripts/download_weights.py
> ```
> This avoids paying Serverless GPU time just for the download.

---

### Step 3 — Set Runpod Secrets

In **Runpod Console → Settings → Secrets**, create:

| Secret Name           | Value                  | Notes                               |
|-----------------------|------------------------|-------------------------------------|
| `HF_TOKEN`            | `hf_xxxxxxxxxxxx`      | Must have gated-repo read access    |
| `S3_ACCESS_KEY_ID`    | your R2/S3 key id      | Optional — for persistent video URLs|
| `S3_SECRET_ACCESS_KEY`| your R2/S3 secret      | Optional                            |

---

### Step 4 — Configure GitHub Actions & Docker Hub Secrets

1. In your GitHub repository, go to **Settings → Secrets and variables → Actions → New repository secret**.
2. Add the following secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username or organization
   - `DOCKERHUB_TOKEN`: Your Docker Hub Personal Access Token (PAT)
3. Push to `main` (or run manually via **Actions → Build & Push Docker Image → Run workflow**):

```bash
git add .
git commit -m "Configure Docker Hub CI/CD workflow"
git push origin main
```

After the workflow completes, the image is published to Docker Hub and GHCR:
```
<YOUR_DOCKERHUB_USERNAME>/ltx2.5:latest
<YOUR_DOCKERHUB_USERNAME>/ltx2.5:sha-abc1234
ghcr.io/<YOUR_ORG>/ltx2.5:latest
```

---

### Step 5 — Create the Runpod Serverless Endpoint

Go to **Runpod Console → Serverless → Endpoints → New Endpoint**.

| Setting                | Value                                                    |
|------------------------|----------------------------------------------------------|
| **Name**               | `ltx25-video-gen`                                        |
| **Container Image**    | `ghcr.io/<YOUR_ORG>/ltx25-runpod-serverless:latest`     |
| **GPU Type**           | NVIDIA L40S (48 GB) — see [GPU Notes](#gpu-notes)        |
| **Min Workers**        | `0`                                                      |
| **Max Workers**        | `3`                                                      |
| **Idle Timeout**       | `60` seconds                                             |
| **Execution Timeout**  | `600` seconds                                            |
| **Container Disk**     | `50` GB                                                  |
| **Network Volume**     | Select `ltx25-weights`, mount at `/runpod-volume`        |

Under **Environment Variables**, add:

```
HF_TOKEN             = (select secret: HF_TOKEN)
S3_ENDPOINT_URL      = https://YOUR_ACCOUNT.r2.cloudflarestorage.com
S3_BUCKET            = your-bucket-name
S3_REGION            = auto
S3_KEY_PREFIX        = ltx-2.5
S3_ACCESS_KEY_ID     = (select secret: S3_ACCESS_KEY_ID)
S3_SECRET_ACCESS_KEY = (select secret: S3_SECRET_ACCESS_KEY)
PRESIGNED_URL_TTL_SECONDS = 86400
```

---

## API Reference

### Run (async)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "A serene mountain lake at golden hour, camera slowly panning right",
      "mode": "text2video",
      "resolution": "720p",
      "num_frames": 97,
      "fps": 24,
      "seed": 42
    }
  }'
```

Returns a job ID; poll `/status/<JOB_ID>` for completion.

### RunSync (blocking, up to execution_timeout)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "A lone astronaut walking on the moon surface",
      "resolution": "480p",
      "num_frames": 49,
      "seed": 1234
    }
  }'
```

Returns the full result JSON when inference completes:

```json
{
  "status": "success",
  "mode": "text2video",
  "video_url": "https://your-r2-bucket.r2.cloudflarestorage.com/ltx-2.5/abc123.mp4?...",
  "duration_seconds": 2.04,
  "generation_time_seconds": 47.3,
  "seed_used": 1234,
  "resolution": "896x512",
  "num_frames": 49,
  "fps": 24,
  "has_audio": true
}
```

`has_audio` reports whether LTX-2.5's joint audio head produced a track — the
model generates video and audio together, and the MP4 has both muxed in. Results
under 5 MB come back as `video_base64` instead of `video_url`; anything larger
needs object storage configured (see [Step 3](#step-3--set-runpod-secrets)),
since Runpod caps job results at 20 MB.

### Image-to-Video

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "The person slowly raises their hand and waves",
      "mode": "image2video",
      "first_frame_image": "https://example.com/my-first-frame.jpg",
      "resolution": "720p",
      "num_frames": 97
    }
  }'
```

### First-Last-Frame (FLF)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "prompt": "A flower blooms from bud to full bloom",
      "mode": "flf2video",
      "first_frame_image": "https://example.com/bud.jpg",
      "last_frame_image": "https://example.com/full-bloom.jpg",
      "resolution": "720p",
      "num_frames": 97
    }
  }'
```

### Input Schema Reference

| Field                  | Type            | Default      | Description                                               |
|------------------------|-----------------|--------------|-----------------------------------------------------------|
| `prompt`               | string          | **required** | Text description (1–2000 chars)                           |
| `mode`                 | enum            | `text2video` | `text2video` \| `image2video` \| `flf2video`. Auto-promoted when images are supplied |
| `first_frame_image`    | string          | `null`       | Base64 or HTTPS URL. Required for `image2video`/`flf2video`|
| `last_frame_image`     | string          | `null`       | Base64 or HTTPS URL. Required for `flf2video`             |
| `resolution`           | enum            | `450p`       | `450p` (768×448) \| `480p` (896×512) \| `576p` (1024×576) \| `720p` (1280×704) \| `1080p` (1920×1088) |
| `num_frames`           | int             | `241`        | Must satisfy `(N−1) % 8 == 0`. Range: 9–257               |
| `fps`                  | int             | `24`         | Output frame rate. Range: 8–30                            |
| `conditioning_strength`| float           | `1.0`        | How tightly the conditioning image(s) constrain the result. Range: >0–1.0 |
| `seed`                 | int \| null     | `null`       | RNG seed for reproducibility. `null` = random             |
| `negative_prompt`      | string          | (see schema) | **Accepted but ignored** — see below                      |
| `num_inference_steps`  | int             | `40`         | **Accepted but ignored** — see below                      |
| `guidance_scale`       | float           | `3.5`        | **Accepted but ignored** — see below                      |

#### Why three fields are ignored

This endpoint runs the **distilled** LTX-2.5 checkpoint, which is
guidance-distilled with a baked-in sigma schedule: 8 steps in stage 1, then a
2× latent upsample and 3 refinement steps in stage 2. It has no
classifier-free-guidance branch and no negative-conditioning path, so
`ltx_pipelines`' `DistilledPipeline.__call__` accepts no `negative_prompt`,
`num_inference_steps` or `guidance_scale` argument at all. The schema keeps the
three fields so existing callers don't break, but they have no effect on output.
Set `LTX_TRANSFORMER=dev` on the endpoint to run the non-distilled checkpoint
instead (slower, and still driven by the pipeline's own schedule here).

#### Why the resolution values look unusual

`DistilledPipeline` is two-stage, so upstream `assert_resolution(..., is_two_stage=True)`
requires the **final** width and height to be divisible by **64** — not 32. The
conventional 848×480, 1280×720 and 1920×1080 all fail that check and raise a
`ValueError` before a single denoising step runs, which is why the tokens map to
896×512, 1280×704 and 1920×1088 instead.

---

## Performance & Cost Notes

### Timing Estimates (L40S 48 GB, `fp8-cast`)

Every warm figure below is for the same fixed schedule — the distilled
checkpoint always runs 8 stage-1 steps, a ×2 latent upsample, then 3 stage-2
refinement steps. `num_inference_steps` in the request does not change it.

| Phase                            | Duration         | Notes                                        |
|----------------------------------|------------------|----------------------------------------------|
| First-ever cold start (download) | 20–40 min        | Downloads ~66 GB; paid once, volume-cached   |
| Subsequent cold starts           | 60–150 s         | Volume-load: weight check + fp8 cast into VRAM |
| Warm inference — 450p / 121f     | ~40–70 s         | Includes chunked VAE decode + libx264 encode |
| Warm inference — 450p / 241f     | ~70–130 s        | The schema default                           |
| Warm inference — 720p / 121f     | ~110–190 s       | Decode dominates at this size                |
| Warm inference — 1080p / 121f    | ~240–420 s       | Needs `LTX_OFFLOAD_MODE=cpu` on 48 GB        |

These are order-of-magnitude estimates, not measurements from this endpoint —
they have not been benchmarked on the deployed worker. Treat them as a guide for
sizing `execution_timeout`, and measure with the `generation_time_seconds` field
the handler returns.

### Understanding Cold Starts vs. Warm Requests

**Common misconception:** "Every request after idle re-downloads the weights."

**Reality:**
- The **network volume** persists between scale-to-zero events. Weights downloaded on the first cold start stay on the volume forever.
- Subsequent cold starts only pay the cost of **re-loading weights from disk into GPU VRAM** (~60–150 s) — not re-downloading.
- Warm workers (still running, in their idle window) serve requests with **no loading cost** at all.

```
                    [First cold start on fresh volume]
  Worker starts → download 66 GB from HF (20–40 min) → load VRAM → serve

  [Scale to zero after idle_timeout]

                    [Any subsequent cold start]
  Worker starts → check volume (files exist!) → skip download → load VRAM (~2 min) → serve
                             ^^^^^^^^^^^^
                          This is the key insight
```

### Cost Optimisation Tips

- **`min_workers = 0`** — zero idle billing. Perfect for on-demand workloads.  
- **`min_workers = 1`** — one worker always warm; eliminates cold starts for the first concurrent request. Costs ~\$0.80–1.20/hr depending on GPU. Use for latency-critical production apps.
- **`idle_timeout = 60s`** — balances warm-start coverage for bursty traffic against idle billing.
- **Pre-populate the volume** using a cheap CPU pod + `scripts/download_weights.py` to avoid paying GPU time for the initial download.

---

## Local Development & Testing

### Unit Tests (no GPU required)

Two suites run without a GPU, CUDA, or the LTX packages installed — both mock
the pipeline and the encoder:

```bash
python tests/test_schema_standalone.py
```

```bash
python tests/test_handler_local.py
```

`test_schema_standalone.py` needs only `pydantic`; it checks every mode,
the `(N−1) % 8 == 0` frame rule, and that all five resolutions are ÷64 legal.
`test_handler_local.py` needs `pydantic` and `loguru`; it covers the handler's
response contract, conditioning temp-file cleanup, and the upload/base64 split.

### Integration Test (GPU + volume required)

```bash
# Set env vars
export HF_TOKEN=hf_xxx
export RUNPOD_VOLUME_PATH=/path/to/your/volume

python tests/test_handler_local.py --integration
```

### Running the Handler Locally (with `test_input.json`)

Create `test_input.json`:
```json
{
  "id": "local-test-001",
  "input": {
    "prompt": "a golden retriever playing fetch on a sunny beach",
    "resolution": "450p",
    "num_frames": 9,
    "seed": 42
  }
}
```

`num_frames: 9` is the smallest legal value and the cheapest way to smoke-test a
real worker — the step count is fixed regardless, so frames are the only knob
that shortens the run.

```bash
cd src
RUNPOD_VOLUME_PATH=/runpod-volume HF_TOKEN=hf_xxx \
  python handler.py  # Runpod SDK reads test_input.json automatically in local mode
```

---

## Troubleshooting

### `EnvironmentError: Runpod network volume not found at '/runpod-volume'`

**Cause:** The network volume isn't attached to the endpoint, or the mount path is wrong.

**Fix:**
1. In the Runpod console, open the endpoint → Edit → confirm the network volume is selected and mount path is `/runpod-volume`.
2. Confirm the volume ID is correct (it's visible in Runpod Console → Storage).

---

### `RuntimeError: HF_TOKEN environment variable is not set`

**Cause:** The Hugging Face token isn't available in the worker environment.

**Fix:**
1. Create a secret named `HF_TOKEN` in Runpod Console → Settings → Secrets.
2. In the endpoint's Environment Variables, add `HF_TOKEN` referencing that secret.
3. Verify the token itself has "read access to gated repositories" scope (check on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

---

### `401 / 403` errors from Hugging Face during download

**Cause:** The HF token is valid but either:
- Doesn't have gated-repo scope, OR
- You haven't accepted the model license on Hugging Face

**Fix:**
1. Visit [huggingface.co/Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), click **"Agree and access repository"**.
2. Re-generate your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **"Read access to gated repos"** scope checked.
3. Update the `HF_TOKEN` secret in Runpod.

---

### `{"error": "out_of_memory", ...}` on inference

**Cause:** The requested resolution/frames exceeds available VRAM.

> **Already fixed once — autograd.** An earlier build OOMed at 450p/241f on a
> 48 GB L40S with **43.37 GiB allocated** right after the log line
> `Text encoder done, building embeddings processor`. That was not a capacity
> problem: nothing in `ltx_pipelines.utils.blocks` disables gradient tracking
> (the only guard upstream ships is `@torch.inference_mode()` on the reference
> CLI's `distilled.main()`, which this repo's handler replaces), so the graph
> hanging off Gemma's hidden states kept the encoder's ~24 GB alive past
> `gpu_model()`'s `dispose()`. `src/inference.py` now wraps both the pipeline
> call and the lazy VAE chunk iterator in `torch.no_grad()`, and `handler()`
> disables grad for the whole request. If you see a comparable OOM, check the
> `[inference] VRAM ...` line in the logs first — allocated ≫ expected means a
> live-tensor leak, not fragmentation, and no
> `PYTORCH_CUDA_ALLOC_CONF` setting will help.

The 22B transformer dominates VRAM, and how much it needs depends entirely on
`LTX_QUANTIZATION`:

| Checkpoint / policy       | Transformer VRAM | Requires        |
|---------------------------|------------------|-----------------|
| bf16, `LTX_QUANTIZATION` unset | ~39 GB      | 80 GB+ card     |
| `fp8-cast` (**default**)  | ~19.6 GB         | sm_89+ (L40S ✅) |
| `nvfp4-*`                 | ~17.4 GB         | sm_100+ (Blackwell only — **not** L40S) |

**Fix:**

| GPU                  | Safe config with the default `fp8-cast`     |
|----------------------|---------------------------------------------|
| RTX PRO 6000 (96 GB) | Up to 1080p / 241f                          |
| H100 80 GB           | Up to 1080p / 121f                          |
| L40S 48 GB           | 450p / 241f, 720p / 121f; 1080p needs `LTX_OFFLOAD_MODE=cpu` |
| 24 GB cards          | Not recommended — `LTX_OFFLOAD_MODE=auto` streams weights for you, but every pass re-transfers them |

`LTX_OFFLOAD_MODE` defaults to `auto`: weights stay resident on a card with at
least 44 GiB of VRAM (an L40S reports 44.39 GiB) and stream layer-by-layer from
pinned host RAM below that, so a smaller GPU gets slower rather than failing.
Streaming only accepts `bf16` and `fp8-cast` quantization — pairing it with
`fp8-scaled-mm` or `nvfp4-*` is rejected at load time with an explicit message.

Reduce `resolution` or `num_frames`. Note that `num_inference_steps` has **no
effect** — the distilled checkpoint runs a fixed 8-step + 3-step schedule — so
lowering it will not relieve memory pressure. The error response includes
guidance text.

---

### Download stalls or partial weights on volume

If `download_weights.py` was interrupted mid-run, re-run it. The sentinel-file check in `model_loader.py` detects missing/zero-size files and triggers a re-download automatically. `snapshot_download` from `huggingface_hub` handles partial-download resumption.

---

### Container fails to pull from `ghcr.io`

**Cause:** The GHCR package is private by default.

**Fix:**
1. Go to your GitHub profile → Packages → `ltx25-runpod-serverless` → Package Settings → Change Visibility → **Public**.  
   OR
2. Configure Runpod to use a registry credential (Runpod Console → Endpoint → Registry Credentials) with a GitHub personal access token that has `read:packages` scope.

---

## GPU Notes

> **Primary:** NVIDIA L40S (48 GB GDDR6, sm_89)
>
> The image defaults to `LTX_QUANTIZATION=fp8-cast` because the bf16 transformer
> alone (~39 GB) leaves no room for the Gemma-4-12B text encoder and the video +
> audio VAEs on a 48 GB card. sm_89 supports fp8 but **not** nvfp4 — the nvfp4
> checkpoints need Blackwell (sm_100+) and `model_loader.py` rejects them below
> that rather than failing cryptically at load time.
>
> If workers sit in **THROTTLED** with 0 idle and 0 unhealthy, that is a capacity
> problem, not a code problem: the selected data centers have no free L40S. Widen
> the GPU pool (`ADA_48_PRO`) or add data centers in the endpoint config — jobs
> will queue indefinitely otherwise.

---

## License

This deployment scaffold is MIT licensed. The LTX-2.5 model weights are subject to the [LTX-2.5 Open Weights License](https://huggingface.co/Lightricks/LTX-2.5/blob/main/LICENSE.md).
