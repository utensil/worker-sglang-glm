import asyncio
import requests
from engine import SGlangEngine
from utils import process_response
import runpod
import os

# Initialize the engine
engine = SGlangEngine()
engine.start_server()
engine.wait_for_server(timeout=2400)

# Warm-up: force the DeepGEMM JIT / lazy kernel compile to run NOW, during worker
# init — BEFORE runpod.serverless.start() begins pulling real jobs. GLM-5-class
# MoE models JIT DeepGEMM lazily on the FIRST real inference (~30 min on H200 —
# SGLang #20401); /v1/models returns 200 before that, so wait_for_server() passes
# "ready" prematurely and the first /run request then blocks for the whole JIT and
# looks like a permanent stall. Running one dummy inference here absorbs that cost
# at load time, so real requests are fast (and it doubles as /health_generate).
def _warmup():
    try:
        r = requests.post(
            f"{engine.base_url}/v1/chat/completions",
            json={"model": engine.model or "default",
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 8, "temperature": 0},
            timeout=(10, 2400))
        print(f"[warmup] status={r.status_code} — JIT/kernels compiled at init", flush=True)
    except Exception as e:  # never block startup on a warm-up failure
        print(f"[warmup] skipped/failed: {e!r}", flush=True)


_warmup()


def get_max_concurrency(default=300):
    """
    Returns the maximum concurrency value.
    By default, it uses 50 unless the 'MAX_CONCURRENCY' environment variable is set.

    Args:
        default (int): The default concurrency value if the environment variable is not set.

    Returns:
        int: The maximum concurrency value.
    """
    return int(os.getenv("MAX_CONCURRENCY", default))


async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]

    # Case 1: full OpenAI style payload where caller already specifies the route.
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get(
            "openai_input"
        )

        openai_url = f"{engine.base_url}" + openai_route
        headers = {"Content-Type": "application/json"}

        # timeout=(connect, read): fail fast instead of hanging the whole job if
        # the server stalls (e.g. a lazy first-inference JIT that the warm-up missed).
        response = requests.post(openai_url, headers=headers, json=openai_input,
                                 timeout=(10, 2400))
        # Process the streamed response
        if openai_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    yield decoded_chunk

    # Case 2: payload looks like OpenAI chat/completions but omits the wrapper.
    elif "messages" in job_input:
        openai_url = f"{engine.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Make sure model is set; fall back to default.
        if "model" not in job_input:
            job_input["model"] = engine.model or "default"

        response = requests.post(openai_url, headers=headers, json=job_input, timeout=(10, 2400))

        if job_input.get("stream", False):
            for formated_chunk in process_response(response):
                yield formated_chunk
        else:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    # Case 3: assume user meant the native /generate endpoint.
    else:
        generate_url = f"{engine.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        # Directly pass `job_input` to `json`. Can we tell users the possible fields of `job_input`?
        response = requests.post(generate_url, json=job_input, headers=headers, timeout=(10, 2400))

        if response.status_code == 200:
            yield response.json()
        else:
            yield {
                "error": f"Generate request failed with status code {response.status_code}",
                "details": response.text,
            }


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": get_max_concurrency,
        "return_aggregate_stream": True,
    }
)
