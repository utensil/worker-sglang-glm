#!/usr/bin/env python3
"""create_pod_template — publish (or update) the GLM-5.2-W4AFP8 RunPod POD template.

A pod template is a saved image+env+disk+ports+docker-command config for launching
a persistent GPU pod (distinct from the serverless Hub manifest in .runpod/). This
POSTs it via the REST API with isServerless=false + isPublic=true, so it appears in
the RunPod console **Explore** section for anyone to deploy. Self-contained
(stdlib + requests); reads RUNPOD_API_KEY from env.

The pod runs the SAME baked image + serve command as the serverless worker, and is
env-driven so it supports BOTH load modes with no template change:
  - no-volume (default): MODEL_NAME=PhalaCloud/GLM-5.2-W4AFP8 (downloads once)
  - volume:              MODEL_NAME=/runpod-volume/GLM-5.2-W4AFP8 + HF_HUB_OFFLINE=1
                         (attach a network volume staged via stage_volume.py)

Usage:
  export RUNPOD_API_KEY=...
  python3 create_pod_template.py --dry-run    # print the payload, $0
  python3 create_pod_template.py              # create the public pod template
"""
import os
import sys
import json
import argparse

import requests

REST = "https://rest.runpod.io/v1"
IMAGE = "ghcr.io/utensil/worker-sglang-glm:latest"

SERVE = (
    'python3 -m sglang.launch_server --model-path "$MODEL_NAME" '
    '--quantization "$QUANTIZATION" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" '
    '--disable-shared-experts-fusion --kv-cache-dtype "$KV_CACHE_DTYPE" '
    '--reasoning-parser "$REASONING_PARSER" --tool-call-parser "$TOOL_CALL_PARSER" '
    '--context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC" '
    '--trust-remote-code --cuda-graph-max-bs 16 --host 0.0.0.0 --port 8000'
)

README = (
    "# GLM-5.2-W4AFP8 - SGLang pod (4xH200)\n\n"
    "OpenAI-compatible GLM-5.2 (753B / 40B-active MoE, DSA sparse attention, up to 1M ctx) "
    "on 4xH200, port 8000.\n\n"
    "**Why W4AFP8:** 4-bit experts + FP8 activations roughly **halve the GPU bill** vs FP8 "
    "(fits **4xH200, not 8x**) at benchmark parity - the strongest cost/speed pick for GLM-5.2. "
    "The DeepGEMM JIT cache is baked into the image -> warm inference ~2.7s (no recompile).\n\n"
    "**Why this image:** a thin fork of `runpod-workers/worker-sglang` that adds GLM-5.2's launch "
    "flags upstream lacks (`KV_CACHE_DTYPE`, `DISABLE_SHARED_EXPERTS_FUSION`, `EXTRA_ARGS`) plus the "
    "baked cache + a register-first handler. **Public image - no GitHub token needed to pull** "
    "(RunPod pulls it anonymously from GHCR).\n\n"
    "**Load modes (optional volume):**\n"
    "- **No-volume (default):** `MODEL_NAME=PhalaCloud/GLM-5.2-W4AFP8`, disk 420GB; downloads "
    "368GB on first start (~12-15min), $0 standing.\n"
    "- **Volume:** attach a network volume with weights pre-staged, set "
    "`MODEL_NAME=/runpod-volume/GLM-5.2-W4AFP8` + `HF_HUB_OFFLINE=1`. For determinism / "
    "no re-download (NOT faster, ~19min). Stage via `stage_volume.py` in the source repo.\n\n"
    "**Env** defaults are set (no secrets baked; no registry credentials). Do NOT set "
    "`LOAD_FORMAT=fastsafetensors` (crashes) or "
    "`DG_JIT_CACHE_DIR` (ignored; cache baked). `HF_TOKEN` only for a gated MODEL_NAME.\n\n"
    "Measured 4xH200: ~68 tok/s single, ~456 peak, TTFT ~1-3s, $10.69/Mtok. "
    "Image `ghcr.io/utensil/worker-sglang-glm:latest` - github.com/utensil/worker-sglang-glm"
)

ENV = {
    "MODEL_NAME": "PhalaCloud/GLM-5.2-W4AFP8", "QUANTIZATION": "w4afp8",
    "TENSOR_PARALLEL_SIZE": "4", "CONTEXT_LENGTH": "32768", "MEM_FRACTION_STATIC": "0.85",
    "REASONING_PARSER": "glm45", "TOOL_CALL_PARSER": "glm47", "KV_CACHE_DTYPE": "fp8_e4m3",
    "NCCL_SHM_DISABLE": "1",
}


def payload():
    return {
        "name": "GLM-5.2-W4AFP8 (SGLang 4xH200)", "imageName": IMAGE, "category": "NVIDIA",
        "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [SERVE],
        "containerDiskInGb": 420, "volumeInGb": 0, "volumeMountPath": "/runpod-volume",
        "ports": ["8000/http"], "env": ENV,
        "isPublic": True, "isServerless": False, "readme": README,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the payload, no API call")
    a = ap.parse_args()
    p = payload()
    if a.dry_run:
        print(json.dumps(p, indent=2))
        print("\n[dry-run] would POST the above to", REST + "/templates", "(isPublic, isServerless=false)")
        return
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("set RUNPOD_API_KEY")
    r = requests.post(REST + "/templates", json=p,
                      headers={"Authorization": "Bearer " + key.strip(),
                               "Content-Type": "application/json"}, timeout=40)
    print("status", r.status_code)
    if r.status_code in (200, 201):
        j = r.json()
        print("template id:", j.get("id"), "| public:", j.get("isPublic"),
              "-> RunPod console > Explore")
    else:
        print(r.text[:400])
        sys.exit(1)


if __name__ == "__main__":
    main()
