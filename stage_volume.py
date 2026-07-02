#!/usr/bin/env python3
"""stage_volume — one-shot helper to pre-stage the GLM-5.2-W4AFP8 weights (368 GB)
onto a RunPod NETWORK VOLUME so a serverless worker cold-starts in ~5-9 min
(local read) instead of ~36 min (re-downloading 368 GB from Hugging Face).

SELF-CONTAINED: stdlib + `requests` only. No dependency on the companion
`runpod-llm` skill. Run it once, keep the volume, point the endpoint at
`/runpod-volume/GLM-5.2-W4AFP8` + `HF_HUB_OFFLINE=1`.

What it does
------------
  1. Create-or-reuse (idempotent by name) a RunPod network volume in a chosen
     datacenter (default US-GA-2 — has 4xH200 for serving + a cheap staging GPU
     + storage, so the volume and the eventual endpoint can co-locate).
  2. Spin a CHEAP staging pod in that SAME datacenter with the volume mounted at
     /runpod-volume, whose start command installs hf tooling and downloads the
     model to the volume:
       pip install -U huggingface_hub hf_transfer &&
       HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download <MODEL> \
         --local-dir /runpod-volume/<name>
  3. Poll for completion (a tiny http.server on the pod serves a sentinel file
     through the RunPod proxy), then in a try/finally ALWAYS terminate the
     staging pod (a leaked pod bills) while intentionally KEEPING the volume.
  4. Print the volume id, the on-volume model path, and the exact volume-mode
     env to set on the serverless endpoint.

REST API (https://rest.runpod.io/v1, Authorization: Bearer <RUNPOD_API_KEY>) —
endpoints/payload shapes are copied from the companion skill's proven calls
(stage.py / podctl.py):
  - GET    /networkvolumes                 -> [{id,name,size,dataCenterId}, ...]
  - POST   /networkvolumes  {name,size,dataCenterId}
  - DELETE /networkvolumes/{id}
  - POST   /pods  {imageName,containerDiskInGb,volumeInGb,volumeMountPath,ports,
                   env,supportPublicIp,dockerEntrypoint,dockerStartCmd,
                   networkVolumeId,dataCenterIds, + compute selector}
      GPU compute selector: {computeType:GPU,cloudType,gpuTypeIds:[..],gpuCount}
      CPU compute selector: {computeType:CPU,cpuFlavorIds:[..],cpuFlavorPriority}
  - GET    /pods/{id}
  - DELETE /pods/{id}

Usage
-----
  export RUNPOD_API_KEY=...                 # required (except --dry-run)
  python3 stage_volume.py --dry-run         # $0, prints the plan
  python3 stage_volume.py                    # stage with defaults
  python3 stage_volume.py --datacenter US-GA-2 --size 480 --compute CPU
  python3 stage_volume.py --no-wait          # create pod, print watch cmd, exit
  python3 stage_volume.py --delete-volume <volume_id>   # sweep
"""
import os
import sys
import time
import argparse

import requests

REST = "https://rest.runpod.io/v1"
VOLUME_MOUNT = "/runpod-volume"
STAGE_PORT = 8000  # http.server port on the staging pod (proxied for polling)

DEFAULT_MODEL = "PhalaCloud/GLM-5.2-W4AFP8"
DEFAULT_NAME = "GLM-5.2-W4AFP8"           # on-volume dir + default volume name
DEFAULT_DATACENTER = "US-GA-2"             # 4xH200 + cheap staging GPU + storage
DEFAULT_SIZE_GB = 480                      # 368 GB weights + headroom
DEFAULT_IMAGE = "python:3.11-slim"         # only needs python + pip + network
# Cheap GPU candidates (first that has capacity wins). A GPU pod tends to get
# more network bandwidth than a bare CPU flavor, so it is the default; use
# --compute CPU for the absolute cheapest.
DEFAULT_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"]
DEFAULT_CPU_FLAVORS = ["cpu3c-2-4", "cpu5c-2-4"]


def _hdr(key, json_body=False):
    h = {"Authorization": "Bearer " + key}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


# ---- 1. network volume: create-or-reuse (REST) ----------------------------
def ensure_volume(name, size_gb, datacenter_id, key):
    """Idempotently create-or-reuse a RunPod network volume. Reuse the first
    volume whose name matches (and datacenter, if it also matches). Returns
    (volume_id, datacenter_id)."""
    r = requests.get(REST + "/networkvolumes", headers=_hdr(key), timeout=30)
    if r.status_code == 200:
        for v in (r.json() or []):
            if v.get("name") == name and (
                    not datacenter_id or v.get("dataCenterId") == datacenter_id):
                print(f"[stage] reusing network volume '{name}' -> id {v['id']} "
                      f"({v.get('dataCenterId')}, {v.get('size')}GB)", flush=True)
                return v["id"], v.get("dataCenterId")
    payload = {"name": name, "size": int(size_gb), "dataCenterId": datacenter_id}
    r = requests.post(REST + "/networkvolumes", json=payload,
                      headers=_hdr(key, True), timeout=60)
    if r.status_code not in (200, 201):
        raise SystemExit(f"[stage] volume create failed {r.status_code}: {r.text[:300]}")
    data = r.json()
    vid, dc = data["id"], data.get("dataCenterId", datacenter_id)
    print(f"[stage] CREATED network volume '{name}' -> id {vid} "
          f"({dc}, {size_gb}GB)", flush=True)
    return vid, dc


def list_volumes(key):
    r = requests.get(REST + "/networkvolumes", headers=_hdr(key), timeout=30)
    return r.json() if r.status_code == 200 else []


def delete_volume(vid, key):
    r = requests.delete(REST + f"/networkvolumes/{vid}", headers=_hdr(key), timeout=30)
    return r.status_code


# ---- staging pod start command --------------------------------------------
def build_start_cmd(model, name):
    """bash -c script for the staging pod: bring up a tiny http.server EARLY so
    the orchestrator can poll, install hf tooling, download the model FLAT into
    /runpod-volume/<name>, then drop a .stage_complete sentinel the poller reads
    through the RunPod proxy. `set -e` makes a failed download skip the sentinel
    (so the poll times out rather than falsely reporting success)."""
    local_dir = f"{VOLUME_MOUNT}/{name}"
    return (
        "set -e; "
        f"mkdir -p {local_dir}; "
        # serve the volume early so polling works throughout the download
        f"cd {VOLUME_MOUNT} && python3 -m http.server {STAGE_PORT} "
        f">/tmp/http.log 2>&1 & "
        "pip install -U -q huggingface_hub hf_transfer; "
        "export HF_HUB_ENABLE_HF_TRANSFER=1; "
        f"huggingface-cli download {model} --local-dir {local_dir}; "
        f"touch {local_dir}/.stage_complete; "
        "echo '[stage] download complete'; "
        # keep the container (and thus http.server) up so the poll sees the
        # sentinel; the orchestrator terminates the pod via the API afterwards.
        "sleep infinity"
    )


# ---- 2. staging pod: create (REST) ----------------------------------------
def create_staging_pod(key, image, volume_id, datacenter_id, script, name,
                       compute, gpus, cpu_flavors, container_disk_gb, hf_token):
    """Create the cheapest staging pod in the volume's datacenter with the volume
    mounted at /runpod-volume and :8000 exposed for the completion poll. Tries
    each compute candidate (cheapest first) until one has capacity. Returns the
    pod id."""
    env = {}
    if hf_token:
        env["HF_TOKEN"] = hf_token  # optional; the default model is public
    base = {
        "name": f"stage-{name}"[:60],
        "imageName": image,
        "containerDiskInGb": int(container_disk_gb),
        "volumeInGb": 0,                       # weights land on the NETWORK volume
        "volumeMountPath": VOLUME_MOUNT,
        "ports": [f"{STAGE_PORT}/http"],
        "env": env,
        "supportPublicIp": True,
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [script],
        "networkVolumeId": volume_id,
        "dataCenterIds": [datacenter_id],      # network volumes are DC-local
    }
    if str(compute).upper() == "CPU":
        attempts = [{"computeType": "CPU", "cpuFlavorIds": [f],
                     "cpuFlavorPriority": "availability"} for f in cpu_flavors]
    else:
        attempts = [{"computeType": "GPU", "cloudType": "COMMUNITY",
                     "gpuTypeIds": [g], "gpuCount": 1} for g in gpus]
    tried = []
    for extra in attempts:
        payload = dict(base, **extra)
        try:
            r = requests.post(REST + "/pods", json=payload,
                              headers=_hdr(key, True), timeout=60)
            if r.status_code in (200, 201):
                data = r.json() if r.text else {}
                pid = data.get("id") or (data.get("pod") or {}).get("id")
                if pid:
                    print(f"[stage] staging pod created -> id {pid} "
                          f"({extra.get('gpuTypeIds') or extra.get('cpuFlavorIds')}, "
                          f"{datacenter_id})", flush=True)
                    return pid
                tried.append(f"{extra}: 2xx but no id: {r.text[:80]}")
            else:
                tried.append(f"{extra}: {r.status_code} {r.text[:80]}")
        except Exception as e:
            tried.append(f"{extra}: {str(e)[:80]}")
    raise SystemExit("[stage] staging pod create failed across candidates:\n  "
                     + "\n  ".join(tried))


def terminate_pod(pid, key):
    """Terminate a pod via REST. Best-effort with retries; returns True on success."""
    for attempt in range(3):
        try:
            r = requests.delete(REST + f"/pods/{pid}", headers=_hdr(key), timeout=30)
            if r.status_code in (200, 204):
                return True
            print(f"[stage] terminate attempt {attempt} -> {r.status_code} "
                  f"{r.text[:120]}", flush=True)
        except Exception as e:
            print(f"[stage] terminate attempt {attempt} failed: {e!r}", flush=True)
        time.sleep(2)
    return False


# ---- 3. poll for completion -----------------------------------------------
def poll_complete(pid, name, timeout_s):
    """Poll the staging pod's proxied http.server for the .stage_complete
    sentinel. Returns True when the download finished, False on timeout. The
    RunPod proxy 403s non-browser user-agents, so send a browser-like UA."""
    url = f"https://{pid}-{STAGE_PORT}.proxy.runpod.net/{name}/.stage_complete"
    hdr = {"User-Agent": "Mozilla/5.0 stage_volume"}
    deadline = time.time() + timeout_s
    next_ping = 0.0
    print(f"[stage] polling {url} (up to {timeout_s/60:.0f} min)", flush=True)
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=15, headers=hdr).status_code == 200:
                return True
        except Exception:
            pass
        now = time.time()
        if now >= next_ping:
            next_ping = now + 120
            mins = (now - (deadline - timeout_s)) / 60
            print(f"[stage] still staging... {mins:.1f} min elapsed", flush=True)
        time.sleep(15)
    return False


# ---- volume-mode env banner -----------------------------------------------
def print_volume_env(volume_id, datacenter_id, name):
    local_path = f"{VOLUME_MOUNT}/{name}"
    print("\n" + "=" * 70)
    print("STAGING COMPLETE — volume KEPT (the staging pod was terminated).")
    print(f"  network volume id : {volume_id}")
    print(f"  datacenter        : {datacenter_id}")
    print(f"  on-volume path    : {local_path}")
    print("\nAttach this volume to the serverless endpoint (it MUST run in the")
    print(f"volume's datacenter {datacenter_id}, which must have 4xH200 free), then")
    print("set the volume-mode env (overriding the no-volume defaults):")
    print("-" * 70)
    print(f"MODEL_NAME={local_path}")
    print("HF_HUB_OFFLINE=1")
    print("-" * 70)
    print("(keep the rest of the serving env: QUANTIZATION, TENSOR_PARALLEL_SIZE,")
    print(" CONTEXT_LENGTH, MEM_FRACTION_STATIC, REASONING_PARSER, TOOL_CALL_PARSER,")
    print(" KV_CACHE_DTYPE, TRUST_REMOTE_CODE, DISABLE_SHARED_EXPERTS_FUSION,")
    print(" EXTRA_ARGS, NCCL_SHM_DISABLE — see the README env table.)")
    print("=" * 70)


# ---- dry-run plan ----------------------------------------------------------
def print_dry_run(a):
    local_path = f"{VOLUME_MOUNT}/{a.name}"
    script = build_start_cmd(a.model, a.name)
    print("=" * 70)
    print("DRY RUN — no API calls, $0. Would perform:")
    print("=" * 70)
    print(f"1. ensure_volume: create-or-reuse network volume")
    print(f"     POST {REST}/networkvolumes")
    print(f"     body {{'name': '{a.volume_name}', 'size': {a.size}, "
          f"'dataCenterId': '{a.datacenter}'}}")
    print(f"2. create staging pod ({a.compute}) in {a.datacenter}")
    print(f"     POST {REST}/pods")
    print(f"     imageName        : {a.image}")
    print(f"     containerDiskInGb: {a.container_disk}")
    print(f"     volumeInGb       : 0   (weights land on the network volume)")
    print(f"     volumeMountPath  : {VOLUME_MOUNT}")
    print(f"     ports            : ['{STAGE_PORT}/http']")
    print(f"     networkVolumeId  : <volume id from step 1>")
    print(f"     dataCenterIds    : ['{a.datacenter}']")
    if a.compute.upper() == "CPU":
        print(f"     computeType      : CPU  (cpuFlavorIds {a.cpu_flavors})")
    else:
        print(f"     computeType      : GPU  (COMMUNITY, gpuTypeIds {a.gpus}, gpuCount 1)")
    print(f"     dockerStartCmd   :\n       {script}")
    print(f"3. poll https://<pod>-{STAGE_PORT}.proxy.runpod.net/{a.name}/.stage_complete")
    print(f"4. try/finally: TERMINATE the staging pod (DELETE {REST}/pods/<id>),")
    print(f"   KEEP the volume.")
    print("=" * 70)
    print(f"Result would be: volume path {local_path}")
    print("Volume-mode env to set on the endpoint afterwards:")
    print(f"  MODEL_NAME={local_path}")
    print(f"  HF_HUB_OFFLINE=1")
    print("=" * 70)


def load_key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        raise SystemExit("[stage] set RUNPOD_API_KEY in the environment "
                         "(not needed for --dry-run).")
    return k.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Pre-stage GLM-5.2-W4AFP8 weights onto a RunPod network volume.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id (default {DEFAULT_MODEL})")
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help=f"on-volume dir name (default {DEFAULT_NAME})")
    ap.add_argument("--volume-name", default=None,
                    help="network volume name (default: same as --name)")
    ap.add_argument("--datacenter", default=DEFAULT_DATACENTER,
                    help=f"RunPod datacenter (default {DEFAULT_DATACENTER})")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE_GB,
                    help=f"volume size in GB (default {DEFAULT_SIZE_GB})")
    ap.add_argument("--image", default=DEFAULT_IMAGE,
                    help=f"staging pod image (default {DEFAULT_IMAGE})")
    ap.add_argument("--compute", default="GPU", choices=["GPU", "CPU"],
                    help="staging compute type (default GPU)")
    ap.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS,
                    help="cheap GPU candidates, cheapest first")
    ap.add_argument("--cpu-flavors", nargs="+", default=DEFAULT_CPU_FLAVORS,
                    help="cheap CPU flavor candidates (for --compute CPU)")
    ap.add_argument("--container-disk", type=int, default=20,
                    help="staging pod container disk GB (default 20; weights go "
                         "to the volume, not here)")
    ap.add_argument("--timeout-min", type=int, default=120,
                    help="max minutes to wait for the download (default 120)")
    ap.add_argument("--no-wait", action="store_true",
                    help="create the pod, print the watch URL, and exit WITHOUT "
                         "polling/terminating (you must terminate it yourself)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without any API calls ($0)")
    ap.add_argument("--delete-volume", metavar="VOLUME_ID", default=None,
                    help="delete a network volume by id and exit (sweep)")
    a = ap.parse_args()
    if a.volume_name is None:
        a.volume_name = a.name

    if a.dry_run:
        print_dry_run(a)
        return

    key = load_key()

    if a.delete_volume:
        code = delete_volume(a.delete_volume, key)
        if code in (200, 204):
            print(f"[stage] deleted volume {a.delete_volume} (HTTP {code})")
        else:
            print(f"[stage] delete volume {a.delete_volume} -> HTTP {code}")
        return

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    vid, dc = ensure_volume(a.volume_name, a.size, a.datacenter, key)
    script = build_start_cmd(a.model, a.name)
    pid = create_staging_pod(key, a.image, vid, dc, script, a.name, a.compute,
                             a.gpus, a.cpu_flavors, a.container_disk, hf_token)

    if a.no_wait:
        watch = f"https://{pid}-{STAGE_PORT}.proxy.runpod.net/{a.name}/.stage_complete"
        print(f"\n[stage] --no-wait: staging pod {pid} is downloading.")
        print(f"[stage] watch for a 200 at: {watch}")
        print(f"[stage] when it returns 200, TERMINATE the pod yourself:")
        print(f"[stage]   python3 stage_volume.py  # (or) DELETE {REST}/pods/{pid}")
        print(f"[stage] then set MODEL_NAME={VOLUME_MOUNT}/{a.name} + HF_HUB_OFFLINE=1")
        print(f"[stage] volume id (KEPT): {vid} ({dc})")
        return

    ok = False
    try:
        ok = poll_complete(pid, a.name, a.timeout_min * 60)
    finally:
        # ALWAYS terminate the staging pod (a leaked pod bills); KEEP the volume.
        killed = terminate_pod(pid, key)
        print(f"[stage] staging pod {pid} terminated: {killed} "
              f"(volume {vid} intentionally KEPT)", flush=True)

    if not ok:
        raise SystemExit(f"[stage] staging did not complete within "
                         f"{a.timeout_min} min (pod {pid} terminated). Re-run to "
                         f"resume — huggingface-cli download is resumable.")
    print_volume_env(vid, dc, a.name)


if __name__ == "__main__":
    main()
