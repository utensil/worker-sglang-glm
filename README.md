# worker-sglang-glm

A thin fork of [`runpod-workers/worker-sglang`](https://github.com/runpod-workers/worker-sglang)
(v2.0.2) that adds the launch flags GLM-5.2-W4AFP8 needs on RunPod Serverless.

## Delta vs upstream (`engine.py`)
Upstream builds the `sglang.launch_server` command from a fixed env→flag mapping and has
no escape hatch for flags it doesn't map. This fork adds:
- `KV_CACHE_DTYPE` → `--kv-cache-dtype` (e.g. `fp8_e4m3`)
- `DISABLE_SHARED_EXPERTS_FUSION` (boolean) → `--disable-shared-experts-fusion`
- **`EXTRA_ARGS`** — a generic shell-split passthrough for any other raw flag, so a new
  flag never needs another fork:
  `EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --disable-shared-experts-fusion"`
- a `[engine] launch: …` log line for observability

Everything else is upstream. `QUANTIZATION=w4afp8` already works upstream (the value is
forwarded to `--quantization` unchecked).

## Build / publish
GitHub Actions (`.github/workflows/build.yml`) builds `linux/amd64` and pushes
`ghcr.io/<owner>/worker-sglang-glm:latest` on every push to `main` (uses the built-in
`GITHUB_TOKEN`; no secrets). Make the GHCR package **public** so RunPod can pull it
without registry credentials.

## Use on RunPod Serverless (GLM-5.2-W4AFP8, 4×H200)
Attach a network volume with weights + a precompiled DeepGEMM cache, set GPUs/worker = 4, env:
```
MODEL_NAME=/runpod-volume/models/GLM-5.2-W4AFP8
QUANTIZATION=w4afp8
TENSOR_PARALLEL_SIZE=4
CONTEXT_LENGTH=32768
MEM_FRACTION_STATIC=0.85
REASONING_PARSER=glm45
TOOL_CALL_PARSER=glm47
TRUST_REMOTE_CODE=true
LOAD_FORMAT=fastsafetensors
EXTRA_ARGS=--kv-cache-dtype fp8_e4m3 --disable-shared-experts-fusion
HF_HUB_OFFLINE=1
DG_JIT_CACHE_DIR=/runpod-volume/deep_gemm_cache
VLLM_WORKER_MULTIPROC_METHOD=spawn
```
See the full deployment recipe in `skills-land/ours/runpod-llm/serverless/README.md`.

## License
Inherits upstream's license (see `LICENSE`). Derived from `runpod-workers/worker-sglang`.
