# worker-sglang-glm

[![Deploy the GLM-5.2 pod template on RunPod](https://img.shields.io/badge/Deploy-GLM--5.2%20Pod-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=uud9e7iajt&ref=km0th85l)
[![Deploy the GLM-5.2 serverless template on RunPod](https://img.shields.io/badge/Deploy-GLM--5.2%20Serverless-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=3zla5fv3z9&ref=km0th85l)

Serve **GLM-5.2-W4AFP8** (a 753B-parameter / ~40B-active MoE with DSA sparse attention) as
your own private, **OpenAI-compatible serverless endpoint** on RunPod. The image bakes in
everything the model needs, so you deploy it, wait out one cold start, and then call it like
any OpenAI endpoint.

## Quick start (serverless)

**Fastest:** click the **Deploy GLM-5.2 Serverless** badge above, which opens the RunPod
console with this template preselected. Or set it up by hand:

1. RunPod console -> **Serverless -> New Endpoint**.
2. Select this template, or import the public image `ghcr.io/utensil/worker-sglang-glm:latest`.
3. **GPU: 4 x H200 (141 GB) per worker.** All four are required - the model is tensor-parallel
   across them. It will not run on fewer or smaller GPUs.
4. **Container disk: 420 GB** (the checkpoint is ~368 GB).
5. **Execution Timeout: at least 2400 s (40 min)**, so the first cold request can finish loading
   the model. See [What to expect](#what-to-expect) for why.
6. **Env: none required.** The worker defaults to the full, tested GLM-5.2-W4AFP8 config out of
   the box. Every setting is still overridable by env if you want - see
   [Environment variables](#environment-variables).
7. Create the endpoint, then call it with the OpenAI API:

```bash
curl https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/openai/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PhalaCloud/GLM-5.2-W4AFP8",
    "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
    "max_tokens": 32
  }'
```

Any OpenAI SDK works too - set `base_url` to `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`
and `api_key` to your RunPod key. Your endpoint is **private to your own RunPod API key**; no
secrets are baked into the image, and the default model is public so **no Hugging Face token is
required** - though setting one is recommended, since it speeds and steadies the 368 GB download
(see [What to expect](#what-to-expect)).

## What to expect

**Hardware.** 4 x H200 (141 GB) per worker, ~368 GB of weights, W4AFP8 (4-bit experts + FP8
activations). It fits on 4 x H200 where a plain FP8 build would need 8, which roughly halves the
GPU bill at parity.

**First cold start: about 36 minutes.** A fresh serverless worker with no attached volume must
download the full 368 GB checkpoint before it can answer. The worker *registers* within seconds
(so RunPod won't recycle it), but the **first job waits** for the download and model load - that
is why the Execution Timeout must be high. Budget for one slow first request per cold worker;
everything after is warm.

**Tip: set `HF_TOKEN` to speed the download.** Even though the default model is public,
authenticating lifts Hugging Face's **anonymous rate limits**, so the 368 GB multi-shard pull is
less likely to hit 429 throttling and tends to download faster and more consistently. It costs
nothing (a read token) and helps both the no-volume cold start and volume staging. Only strictly
*required* if you point `MODEL_NAME` at a gated repo.

**Warm inference is fast.** The DeepGEMM kernel cache is baked into the image, so a warm worker
skips the 10-20 min JIT recompile. Measured on 4 x H200:
- Response starts in ~2-3 s (TTFT stays near-flat ~3 s up to ~18k-token prompts).
- ~68 tokens/s single stream, ~456 tokens/s aggregate across 16 concurrent requests.

**Cost.** About **$17.56/hr** per live 4 x H200 worker, roughly **$11-16 per million output
tokens**. A flex endpoint (`workers_min = 0`) costs **$0 while idle** - only a running worker
bills - and RunPod's FlashBoot caches worker state for faster later starts, so **keep the
endpoint** between uses rather than deleting and recreating it.

**Shrinking the cold start.** If the ~36 min first request is a problem for your pattern, you can
pre-stage the weights on a network volume (a deterministic ~19 min - see [Load modes](#load-modes)).
Note that on a fast node a plain no-volume download can be about as quick, so a volume mainly buys
*predictability*, not raw speed.

## HiCache (long-context KV offload)

**Off by default. Turn it on for long-context, reuse-heavy workloads** - load a large document
or session once, then ask many questions against it. Set `ENABLE_HICACHE=true`.

**What it does.** It tiers the KV cache from GPU to host RAM. When a large reused context no
longer fits in the GPU KV pool, HiCache **restores it from CPU** instead of recomputing the whole
prefill from scratch.

**When it helps (measured on 4 x H200).** The payoff scales with context length, because it only
saves you the recompute you would otherwise pay:
- At **18k** tokens: no benefit - recomputing an evicted prefix is only ~4 s, nothing to save.
- At **43k** tokens: an evicted context re-read in **2.5 s vs 7.7 s recompute (~3x faster)**.
- At **100k-500k**: a recompute is tens of seconds to minutes, so restoring from CPU saves
  proportionally more. Rule of thumb: **turn it on above ~50k reuse; leave it off below.**

**Enable and tune:**
```
ENABLE_HICACHE=true
CONTEXT_LENGTH=131072       # room for your long contexts (more KV VRAM)
MAX_TOTAL_TOKENS=200000     # GPU KV pool; larger = fewer evictions
HICACHE_RATIO=2             # CPU tier size = ratio x GPU pool; raise it to hold more
```
The actively-reused context stays hot in the CPU tier and keeps restoring cheaply; only the
*oldest* contexts age out (LRU), so raise `HICACHE_RATIO` if you juggle several big contexts. The
remaining knobs have sensible defaults: `HICACHE_IO_BACKEND=kernel`, `HICACHE_MEM_LAYOUT=page_first`,
`HICACHE_WRITE_POLICY=write_through`, `PAGE_SIZE=64`.

**Image:** this capability ships on the **`hicache` branch / `:hicache` image tag**
(`ghcr.io/utensil/worker-sglang-glm:hicache`), kept separate from the default `:latest` so the
published template stays frozen. Deploy that tag to use it.

## Deploy as a pod (published pod template)

Besides the serverless template, a **public RunPod pod template** ships this same baked image +
serve command for launching GLM-5.2 as a persistent GPU pod (bills every second, so kill it when
done). Click the **Deploy GLM-5.2 Pod** badge above, find **"GLM-5.2-W4AFP8 (SGLang 4xH200)"** in
the console **Explore** section, or recreate it under your own account with
`python3 create_pod_template.py` (self-contained; `--dry-run` prints the payload). Deploy on
**4 x H200**, port 8000 = OpenAI API. It is env-driven, so the same template serves **both load
modes** with no change - keep the default `MODEL_NAME` for no-volume, or set it to the
`/runpod-volume/...` path + `HF_HUB_OFFLINE=1` and attach a staged volume (see below).

## Environment variables

The worker serves GLM-5.2-W4AFP8 with **zero env config** - the values below are applied as
defaults. Set any of them as an endpoint env var only to **override** a default. `engine.py` maps
`MODEL_NAME -> --model-path` etc. onto the `sglang.launch_server` command; booleans are appended
when set to `true`/`1`/`yes`.

| Variable | Maps to / effect | Default (GLM-5.2-W4AFP8) | Notes |
|---|---|---|---|
| `MODEL_NAME` | `--model-path` | `PhalaCloud/GLM-5.2-W4AFP8` **or** `/runpod-volume/GLM-5.2-W4AFP8` | HF id (no-volume) or on-volume local path (volume mode). |
| `QUANTIZATION` | `--quantization` | `w4afp8` | The model's quant scheme. |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | `4` | Must equal the GPU count per worker (4 x H200). |
| `CONTEXT_LENGTH` | `--context-length` | `32768` | Max sequence length. Raise for longer contexts (needs more KV VRAM). |
| `MEM_FRACTION_STATIC` | `--mem-fraction-static` | `0.85` | Fraction of VRAM for weights + KV. |
| `REASONING_PARSER` | `--reasoning-parser` | `glm45` | Parses GLM reasoning output. |
| `TOOL_CALL_PARSER` | `--tool-call-parser` | `glm47` | Parses GLM tool calls. |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `fp8_e4m3` | Fork addition (upstream lacks it). |
| `TRUST_REMOTE_CODE` | `--trust-remote-code` (bool) | `true` | Needed for the GLM custom arch. |
| `DISABLE_SHARED_EXPERTS_FUSION` | `--disable-shared-experts-fusion` (bool) | `true` | Fork addition; required for this MoE. |
| `EXTRA_ARGS` | raw shell-split passthrough | *(unset)* | Any flag not otherwise mapped, e.g. `--cuda-graph-max-bs 16` (fork escape hatch). |
| `ENABLE_HICACHE` | `--enable-hierarchical-cache` + tier knobs (bool) | *(unset = off)* | KV offload to host RAM for long-context reuse. See [HiCache](#hicache-long-context-kv-offload) for `HICACHE_RATIO` etc. |
| `PAGE_SIZE` | `--page-size` | *(unset; 64 under HiCache)* | KV page size. HiCache `page_first` layout needs it, defaulted to 64. |
| `MAX_TOTAL_TOKENS` | `--max-total-tokens` | *(unset)* | GPU KV pool size. Larger = fewer evictions (relevant with HiCache). |
| `NCCL_SHM_DISABLE` | NCCL env (not a flag) | set `1` if TP init hangs | Multi-GPU: disables NCCL's /dev/shm transport (RunPod's tiny 64 MB shm can otherwise kill a TP worker at init). |
| `HF_TOKEN` | Hugging Face auth | *(unset, recommended)* | **Required** for a gated `MODEL_NAME`. Even for the public default, a read token authenticates the download and lifts anonymous HF rate limits, reducing 429 throttling on the 368 GB pull - faster, steadier cold start and staging. |
| `HF_HUB_OFFLINE` | transformers/sglang: no hub calls | *(unset)* | Set `1` **only** in volume mode (local `/runpod-volume/...` path) so the loader never contacts HF. |

**Do NOT set** `LOAD_FORMAT=fastsafetensors` (crashes the sglang v0.5.14 scheduler) or
`DG_JIT_CACHE_DIR` (ignored by DeepGEMM here; the cache is already baked at the default path).

Infra defaults you normally leave alone: `HOST` (`0.0.0.0`) and `PORT` (`30000`). Other upstream
`sglang.launch_server` flags remain available via the same env->flag mapping if needed.

## Load modes

Two ways to get the 368 GB of weights in front of the worker. Pick per your idle pattern.

### No-volume (default)
- `MODEL_NAME=PhalaCloud/GLM-5.2-W4AFP8` (do **not** set `HF_HUB_OFFLINE`), container disk **420 GB**.
- Cold start **~36 min** - dominated by the 368 GB Hugging Face download on each fresh worker.
- **$0 idle.** Best for weeks-idle / bursty use where you never pay for standing storage.

### Volume (pre-staged weights - deterministic, not necessarily faster)
- Run **`stage_volume.py`** once to pre-stage the weights onto a region-pinned RunPod network
  volume (see below), attach that volume to the endpoint, then set:
  ```
  MODEL_NAME=/runpod-volume/GLM-5.2-W4AFP8
  HF_HUB_OFFLINE=1
  ```
  (keep every other default from the table above).
- **Measured cold start ~19 min** on 4 x H200 - the **network-volume weight READ (~12 min)**
  dominates, + baked DeepGEMM ~2 min + graph ~4 min. **Reality check:** this is *not*
  dramatically faster than no-volume - a direct HF download to local disk on a fast H200 node was
  ~12.6 min total, so on a fast-network node **no-volume can actually beat a volume.** The 368 GB
  weight I/O is slow either way; pre-staging doesn't fix that.
- **So a volume's real value is determinism, not speed:** it **caps** weight I/O at the volume read
  time (~12 min) regardless of node network - a hedge against slow-download workers - plus no
  re-download egress on repeat cold starts and no Hugging Face dependency. Pick it for
  *predictability*, not to go fast.
- Small **standing $/mo** for the volume storage.
- **Co-location constraint:** a RunPod network volume is datacenter-local, so the endpoint **must
  run in the volume's datacenter** (e.g. the `stage_volume.py` default `US-GA-2`), and that
  datacenter **must have 4 x H200 available**. If it doesn't, fall back to no-volume mode or stage
  into a DC that has both.

#### Staging the volume - `stage_volume.py`
A self-contained helper (stdlib + `requests` only) that stages the weights once:

```
export RUNPOD_API_KEY=...
python3 stage_volume.py --dry-run          # $0 - print the exact plan first
python3 stage_volume.py                     # create-or-reuse volume, download, verify, sweep pod
# uv run --with requests python stage_volume.py --dry-run   # if requests isn't in the base env
```

It (1) create-or-reuses a network volume (idempotent by name) in `--datacenter US-GA-2`
(`--size 480` GB default), (2) spins a **cheap** staging pod in that same datacenter with the
volume mounted at `/runpod-volume` that runs `huggingface-cli download ... --local-dir
/runpod-volume/GLM-5.2-W4AFP8` (via `hf_transfer`), (3) polls for completion, then in a
`try/finally` **always terminates the staging pod** while **keeping the volume**. It prints the
volume id, the on-volume model path, and the volume-mode env above. `--no-wait` creates the pod
and prints a watch URL instead of blocking; `--delete-volume <id>` sweeps a volume you no longer
need. `HF_TOKEN` (env) is forwarded for gated repos but the default model is public.

**Staging is slow - budget ~1-2 h (a one-time cost).** Writing 368 GB to a *network* volume is
bound by the volume's **write throughput** (~45-150 MB/s), *not* the GPU node's download speed - a
fast GPU does not make it much faster, so the cheap default (L4) is the right pick.
`huggingface-cli download` is **resumable**: if a run hits `--timeout-min`, just re-run (it skips
completed shards; expect it to re-verify existing files first, which itself takes a while). Raise
`--timeout-min` (default 120) for fewer re-runs.

**GPU/capacity tips (verified the hard way):**
- Community capacity is unreliable in the H200 datacenters this targets - if `--compute GPU` can't
  place, pass **`--cloud SECURE`**; a **SECURE L4 (~$0.43/hr)** is a reliable cheap staging box.
- `--gpus` values are RunPod **REST gpu-type ids**, which differ from console *display* names (e.g.
  "H100 SXM" -> `NVIDIA H100 80GB HBM3`, and L4 -> `NVIDIA L4`). A `400 ... items/enum` error means
  the id string is wrong; a `500 ... could not find any pods` means no capacity for it there.

## Publish on the RunPod Hub

This repo ships a Hub manifest (`.runpod/hub.json` + `.runpod/tests.json`). To list your fork:
create a **GitHub Release** (the Hub indexes releases, not commits), then in the console ->
**Hub -> Add your repo** -> paste the repo URL. RunPod scans, builds, tests, then a human review
lists it. Two things learned publishing this one:
- **Connect the RunPod GitHub App first** (Settings -> Connections -> GitHub) with access to the
  repo, or the "Add your repo" button never appears and the submit fails with a generic error.
- **Keep `hub.json` `description` <= 191 chars** (a Prisma `VARCHAR(191)` column) and remember the
  Hub sets **no env** during its build/test, so the worker must serve with zero env config (this
  fork does). The Hub renders the listing README from the **release snapshot**, so re-release to
  update it.

**Cold-start caveat for this model:** the automated Hub test spins a *fresh* worker with **no
volume** (~36 min to download 368 GB); `tests.json` sets a 45-min timeout. If the Hub pipeline caps
test time below that, deploy from the image directly (Quick start above) instead.

## How it works (internals)

**Fork delta (`engine.py`).** A thin fork of
[`runpod-workers/worker-sglang`](https://github.com/runpod-workers/worker-sglang) (v2.0.2).
Upstream builds the `sglang.launch_server` command from a fixed env->flag mapping with no escape
hatch. This fork adds `KV_CACHE_DTYPE -> --kv-cache-dtype`,
`DISABLE_SHARED_EXPERTS_FUSION -> --disable-shared-experts-fusion`, an **`EXTRA_ARGS`** shell-split
passthrough for any other raw flag, a `[engine] launch: ...` log line, and **built-in GLM-5.2
defaults** so the worker serves correctly with no env set.

**Register-first handler (`handler.py`).** Restructured to start the model load in a background
thread and call `runpod.serverless.start()` immediately - the worker registers within seconds
(beating RunPod's ~500s recycle window) while the first job waits for the load. This is why you
raise the per-request Execution Timeout.

**What's baked in.** Base `lmsysorg/sglang:v0.5.14-cu129` (has GLM-5.2's `GlmMoeDsaForCausalLM` DSA
arch + w4afp8), plus **`dg_cache/deep_gemm/` -> `/root/.cache/deep_gemm`** - the DeepGEMM JIT
kernels (SM90/Hopper FP8 GEMM + GLM DSA indexer) precompiled on a live 4 x H200 pod, so a cold
worker skips the 10-20 min JIT recompile. sglang v0.5.14's DeepGEMM ignores `DG_JIT_CACHE_DIR` and
reads the default `~/.cache/deep_gemm`, which is why the cache is baked there and why you should
NOT set `DG_JIT_CACHE_DIR`.

**Build / publish.** GitHub Actions (`.github/workflows/build.yml`) builds `linux/amd64` and pushes
`ghcr.io/<owner>/worker-sglang-glm:latest` on every push to `main` (built-in `GITHUB_TOKEN`, no
secrets). Make the GHCR package **public** so RunPod can pull it without credentials.

## License
Inherits upstream's license (see `LICENSE`). Derived from `runpod-workers/worker-sglang`.
