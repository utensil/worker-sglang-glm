import asyncio
import threading
import requests
from engine import SGlangEngine
from utils import process_response
import runpod
import os

# Load the model in a BACKGROUND thread and call runpod.serverless.start()
# IMMEDIATELY (bottom of file). Why: a big model (GLM-5.2 ~556s load) blocking at
# import exceeds RunPod's ~500s worker-recycle window, so the platform kills the
# worker before it registers to pull jobs → infinite restart loop (CONFIRMED
# 2026-07-02 via console logs). Registering first (worker "ready" in seconds)
# beats the recycle; the first job then waits for the model to finish loading
# (raise the endpoint Execution Timeout above the load time).
engine = SGlangEngine()
_state = {"ready": False, "error": None}


def _load_model():
    try:
        engine.start_server()
        engine.wait_for_server(timeout=2400)
        _state["ready"] = True
        print("[load] model ready — accepting jobs", flush=True)
    except Exception as e:
        _state["error"] = repr(e)
        print(f"[load] FAILED: {e!r}", flush=True)


threading.Thread(target=_load_model, daemon=True).start()


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
    # The worker registered before the model finished loading (so it wasn't
    # recycled). The FIRST job waits here for the background load to complete;
    # later jobs find it ready and proceed immediately. Keep the endpoint's
    # Execution Timeout above the model load time so this wait doesn't get killed.
    while not _state["ready"]:
        if _state["error"]:
            yield {"error": f"model failed to load: {_state['error']}"}
            return
        await asyncio.sleep(3)
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
