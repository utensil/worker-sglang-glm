# worker-sglang-glm

A thin fork of [`runpod-workers/worker-sglang`](https://github.com/runpod-workers/worker-sglang)
(v2.0.2) that serves **GLM-5.2-W4AFP8** on RunPod Serverless: it adds the launch flags the
model needs, **bakes the DeepGEMM JIT cache into the image** for fast warm inference, and
uses a **register-first background-load handler** so the worker survives RunPod's ~500s
recycle window on this large (368 GB) model.

## Delta vs upstream (`engine.py`)
Upstream builds the `sglang.launch_server` command from a fixed env→flag mapping and has
no escape hatch for flags it doesn't map. This fork adds:
- `KV_CACHE_DTYPE` → `--kv-cache-dtype` (e.g. `fp8_e4m3`)
- `DISABLE_SHARED_EXPERTS_FUSION` (boolean) → `--disable-shared-experts-fusion`
- **`EXTRA_ARGS`** — a generic shell-split passthrough for any other raw flag, so a new
  flag never needs another fork.
- a `[engine] launch: …` log line for observability

`handler.py` is restructured to **start the model load in a background thread and call
`runpod.serverless.start()` immediately** — the worker registers within seconds (beating
RunPod's ~500s recycle) and the first job waits for the load. Raise the per-request
Execution Timeout to cover the first (cold) request.

## What's baked in
- Base `lmsysorg/sglang:v0.5.14-cu129` (has GLM-5.2's `GlmMoeDsaForCausalLM` DSA arch + w4afp8).
- **`dg_cache/deep_gemm/` → `/root/.cache/deep_gemm`** — the DeepGEMM JIT kernels (SM90/Hopper
  FP8 GEMM + GLM DSA indexer kernels) precompiled on a live 4×H200 pod, so a cold worker skips
  the JIT recompile (which otherwise crawls 10-20 min on the container disk / a volume).
  **Note:** sglang v0.5.14's DeepGEMM ignores `DG_JIT_CACHE_DIR` and reads the default
  `~/.cache/deep_gemm` — that is why the cache is baked there, and why you should NOT set
  `DG_JIT_CACHE_DIR`.

## Build / publish
GitHub Actions (`.github/workflows/build.yml`) builds `linux/amd64` and pushes
`ghcr.io/<owner>/worker-sglang-glm:latest` on every push to `main` (built-in `GITHUB_TOKEN`,
no secrets). Make the GHCR package **public** so RunPod can pull it without credentials.

## Deploy as a serverless endpoint (GLM-5.2-W4AFP8, 4×H200)

### A) RunPod Hub (`.runpod/hub.json` + `.runpod/tests.json`)
This repo ships a Hub manifest. To list it: create a **GitHub Release** (the Hub indexes
releases, not commits), then in the RunPod console → **Hub → Add your repo** → paste this
repo URL. RunPod scans/builds/tests, then a human review lists it.
**Caveat for this model:** the automated Hub test spins a *fresh* worker with **no volume**,
so it re-downloads the 368 GB checkpoint before it can answer — a **~35-40 min cold start**.
The bundled `tests.json` sets a long timeout accordingly; if the Hub pipeline caps test time
below that, publish via (B) instead, or request a longer test window.

### B) New Endpoint from the image (works today, no Hub review)
Console → **Serverless → New Endpoint** → import the public image
`ghcr.io/utensil/worker-sglang-glm:latest`, set **4 GPUs (H200 141 GB) per worker**,
**container disk 420 GB**, and the env below. Or create it programmatically (see `sless.py`
in the companion `runpod-llm` skill). The endpoint is **private to your own RunPod API key**;
sharing the template grants no cross-access — each instantiator runs under their own key.

**Env (no secrets; `HF_TOKEN` only for a gated `MODEL_NAME` — the default is public):**
```
MODEL_NAME=PhalaCloud/GLM-5.2-W4AFP8
QUANTIZATION=w4afp8
TENSOR_PARALLEL_SIZE=4
CONTEXT_LENGTH=32768
MEM_FRACTION_STATIC=0.85
REASONING_PARSER=glm45
TOOL_CALL_PARSER=glm47
KV_CACHE_DTYPE=fp8_e4m3
TRUST_REMOTE_CODE=true
DISABLE_SHARED_EXPERTS_FUSION=true
EXTRA_ARGS=--cuda-graph-max-bs 16
NCCL_SHM_DISABLE=1
```

**Do NOT set** `LOAD_FORMAT=fastsafetensors` (crashes the sglang v0.5.14 scheduler) or
`DG_JIT_CACHE_DIR` (ignored by DeepGEMM here; the cache is already baked at the default path).

**Warm inference** is ~2-3 s (the baked cache skips the JIT). Measured on 4×H200:
**~68 tok/s single, ~456 peak, TTFT ~1-3 s** (near-flat TTFT up to ~18 k-token prompts).
Cold-start numbers depend on the load mode — see **[Load modes](#load-modes)** below.

## Environment variables

Every env the worker reads. `engine.py` maps `MODEL_NAME→--model-path` etc. onto the
`sglang.launch_server` command; booleans are appended only when set to `true`/`1`/`yes`.
The values below are the tested GLM-5.2-W4AFP8 defaults (also shown in the env block above).

| Variable | Maps to / effect | Value (GLM-5.2-W4AFP8) | Notes |
|---|---|---|---|
| `MODEL_NAME` | `--model-path` | `PhalaCloud/GLM-5.2-W4AFP8` **or** `/runpod-volume/GLM-5.2-W4AFP8` | HF id (no-volume) or on-volume local path (volume mode). |
| `QUANTIZATION` | `--quantization` | `w4afp8` | The model's quant scheme. |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | `4` | Must equal the GPU count per worker (4×H200). |
| `CONTEXT_LENGTH` | `--context-length` | `32768` | Max sequence length. |
| `MEM_FRACTION_STATIC` | `--mem-fraction-static` | `0.85` | Fraction of VRAM for weights + KV. |
| `REASONING_PARSER` | `--reasoning-parser` | `glm45` | Parses GLM reasoning output. |
| `TOOL_CALL_PARSER` | `--tool-call-parser` | `glm47` | Parses GLM tool calls. |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `fp8_e4m3` | Fork addition (upstream lacks it). |
| `TRUST_REMOTE_CODE` | `--trust-remote-code` (bool) | `true` | Needed for the GLM custom arch. |
| `DISABLE_SHARED_EXPERTS_FUSION` | `--disable-shared-experts-fusion` (bool) | `true` | Fork addition; required for this MoE. |
| `EXTRA_ARGS` | raw shell-split passthrough | `--cuda-graph-max-bs 16` | Any flag not otherwise mapped (fork escape hatch). |
| `NCCL_SHM_DISABLE` | NCCL env (not a flag) | `1` | Multi-GPU: disables NCCL's /dev/shm transport (RunPod's tiny 64 MB shm otherwise kills a TP worker at init). |
| `HF_TOKEN` | Hugging Face auth | *(unset)* | Only needed for a **gated** `MODEL_NAME`; the default is public. `stage_volume.py` forwards it too. |
| `HF_HUB_OFFLINE` | transformers/sglang: no hub calls | `1` (**volume mode only**) | Set **only** when `MODEL_NAME` is a local `/runpod-volume/...` path so the loader never contacts HF. |

Infra defaults you normally leave alone: `HOST` (`0.0.0.0`) and `PORT` (`30000`) are read by
`engine.py`; `SERVED_MODEL_NAME`, `LOAD_FORMAT`, `DTYPE`, and the many other upstream
`sglang.launch_server` flags are all still available via the same env→flag mapping if needed.

## Load modes

Two ways to get the 368 GB of weights in front of the worker. Pick per your idle pattern.

### No-volume (default)
- `MODEL_NAME=PhalaCloud/GLM-5.2-W4AFP8` (do **not** set `HF_HUB_OFFLINE`), container disk **420 GB**.
- Cold start **≈ 36 min** — dominated by the 368 GB Hugging Face download on each fresh worker.
- **$0 idle.** Best for weeks-idle / bursty use where you never pay for standing storage.

### Volume (pre-staged weights — deterministic, not necessarily faster)
- Run **`stage_volume.py`** once to pre-stage the weights onto a region-pinned RunPod network
  volume (see below), attach that volume to the endpoint, then set:
  ```
  MODEL_NAME=/runpod-volume/GLM-5.2-W4AFP8
  HF_HUB_OFFLINE=1
  ```
  (keep every other env from the table above).
- **Measured cold start ≈ 19 min** on 4×H200 — the **network-volume weight READ (~12 min)**
  dominates, + baked DeepGEMM ~2 min + graph ~4 min. **Reality check:** this is *not*
  dramatically faster than no-volume — a **direct HF download to local disk on a fast H200 node
  was ~12.6 min total** (R4), so on a fast-network node **no-volume can actually beat a volume.**
  The 368 GB weight I/O is slow either way; pre-staging doesn't fix that.
- **So a volume's real value is determinism, not speed:** it **caps** weight I/O at the volume
  read time (~12 min) regardless of node network — a **hedge** against slow-download workers
  (a no-volume *serverless* cold start was ~36 min on a slow worker), plus no re-download egress
  on repeat cold starts and no Hugging Face dependency. Pick it for *predictability*, not to go fast.
- Small **standing $/mo** for the volume storage.
- **Co-location constraint:** a RunPod network volume is datacenter-local, so the endpoint
  **must run in the volume's datacenter** (an `M22`-style region, e.g. the `stage_volume.py`
  default `US-GA-2`), and that datacenter **must have 4×H200 available**. If it doesn't, you
  can't attach the volume there — fall back to no-volume mode or stage into a DC that has both.

#### Staging the volume — `stage_volume.py`
A self-contained helper (stdlib + `requests` only) that stages the weights once:

```
export RUNPOD_API_KEY=...
python3 stage_volume.py --dry-run          # $0 — print the exact plan first
python3 stage_volume.py                     # create-or-reuse volume, download, verify, sweep pod
# uv run --with requests python stage_volume.py --dry-run   # if requests isn't in the base env
```

It (1) create-or-reuses a network volume (idempotent by name) in `--datacenter US-GA-2`
(`--size 480` GB default), (2) spins a **cheap** staging pod in that same datacenter with the
volume mounted at `/runpod-volume` that runs `huggingface-cli download … --local-dir
/runpod-volume/GLM-5.2-W4AFP8` (via `hf_transfer`), (3) polls for completion, then in a
`try/finally` **always terminates the staging pod** (a leaked pod bills) while **keeping the
volume**. It prints the volume id, the on-volume model path, and the volume-mode env above.
`--no-wait` creates the pod and prints a watch URL instead of blocking; `--delete-volume <id>`
sweeps a volume you no longer need. `HF_TOKEN` (env) is forwarded for gated repos but the
default model is public.

**Staging is slow — budget ~1-2 h (and it's a one-time cost).** Writing 368 GB to a *network*
volume is bound by the volume's **write throughput** (~45-150 MB/s), *not* the GPU node's
download speed — a fast GPU does not make it much faster, so the cheap default (L4) is the right
pick. `huggingface-cli download` is **resumable**: if a run hits `--timeout-min`, just re-run
(it skips completed shards and continues; expect it to re-verify existing files on the volume
first, which itself takes a while). Raise `--timeout-min` (default 120) for fewer re-runs.

**GPU/capacity tips (verified the hard way):**
- Community capacity is unreliable in the H200 datacenters this targets — if `--compute GPU`
  can't place, pass **`--cloud SECURE`**; a **SECURE L4 (~$0.43/hr)** is a reliable cheap staging box.
- `--gpus` values are RunPod **REST gpu-type ids**, which differ from console *display* names
  (e.g. "H100 SXM" → `NVIDIA H100 80GB HBM3`, and L4 → `NVIDIA L4`). A `400 … items/enum` error
  means the id string is wrong; a `500 … could not find any pods` means no capacity for it there.

## License
Inherits upstream's license (see `LICENSE`). Derived from `runpod-workers/worker-sglang`.
