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

**Cold vs warm:** no-volume cold start ≈ 35-40 min (dominated by the 368 GB download; $0 idle,
best for weeks-idle use). For a faster cold start, stage the weights on a **region-pinned
network volume**, then set `MODEL_NAME` to the volume path + `HF_HUB_OFFLINE=1`. Warm inference
is ~2-3 s (the baked cache skips the JIT). Measured on 4×H200: **~68 tok/s single, ~456 peak,
TTFT ~1-3 s** (near-flat TTFT up to ~18 k-token prompts).

## License
Inherits upstream's license (see `LICENSE`). Derived from `runpod-workers/worker-sglang`.
