import subprocess
import time
import requests
import openai
import asyncio
import aiohttp
import os


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_NAME"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 30000)),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = {
            "MODEL_NAME": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "TOKENIZER_MODE": "--tokenizer-mode",
            "LOAD_FORMAT": "--load-format",
            "DTYPE": "--dtype",
            "CONTEXT_LENGTH": "--context-length",
            "QUANTIZATION": "--quantization",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_RUNNING_REQUESTS": "--max-running-requests",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
            "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
            "SCHEDULE_POLICY": "--schedule-policy",
            "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "STREAM_INTERVAL": "--stream-interval",
            "RANDOM_SEED": "--random-seed",
            "LOG_LEVEL": "--log-level",
            "LOG_LEVEL_HTTP": "--log-level-http",
            "API_KEY": "--api-key",
            "FILE_STORAGE_PATH": "--file-storage-path",
            "DATA_PARALLEL_SIZE": "--data-parallel-size",
            "LOAD_BALANCE_METHOD": "--load-balance-method",
            "ATTENTION_BACKEND": "--attention-backend",
            "SAMPLING_BACKEND": "--sampling-backend",
            "TOOL_CALL_PARSER": "--tool-call-parser",
            "REASONING_PARSER": "--reasoning-parser",
            # --- GLM-5.2-W4AFP8 additions (upstream worker-sglang lacks these) ---
            "KV_CACHE_DTYPE": "--kv-cache-dtype",          # e.g. fp8_e4m3
            "PAGE_SIZE": "--page-size",                    # KV page size (HiCache page_first needs it)
        }

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
            "DISABLE_SHARED_EXPERTS_FUSION",   # GLM-5.2-W4AFP8 MoE
        ]

        # Defaults so this worker serves GLM-5.2-W4AFP8 correctly with ZERO env
        # config. The RunPod Hub build/test injects none of hub.json's env
        # defaults, so without these fallbacks the launch command is modelless
        # (just --host/--port) and every fitness check fails. Each value is still
        # overridable by setting the matching env var at deploy time.
        defaults = {
            "MODEL_NAME": "PhalaCloud/GLM-5.2-W4AFP8",
            "QUANTIZATION": "w4afp8",
            "KV_CACHE_DTYPE": "fp8_e4m3",
            "REASONING_PARSER": "glm45",
            "TOOL_CALL_PARSER": "glm47",
            "CONTEXT_LENGTH": "32768",
            "MEM_FRACTION_STATIC": "0.85",
            "TENSOR_PARALLEL_SIZE": "4",
        }
        boolean_defaults = {"DISABLE_SHARED_EXPERTS_FUSION", "TRUST_REMOTE_CODE"}

        # Add options: env var if set, else the GLM default.
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is None or value == "":
                value = defaults.get(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags: respect an explicit env value; else default-on for
        # the flags GLM-5.2 requires (shared-experts-fusion off, trust-remote-code).
        for flag in boolean_flags:
            set_val = os.getenv(flag)
            if set_val is not None and set_val != "":
                on = set_val.lower() in ("true", "1", "yes")
            else:
                on = flag in boolean_defaults
            if on:
                command.append(f"--{flag.lower().replace('_', '-')}")

        # HiCache: tier the KV cache to host RAM (GPU -> CPU) so a large reused
        # context that no longer fits in the GPU pool is RESTORED from CPU instead
        # of recomputed. Off by default (adds write-through overhead and only pays
        # off for long-context REUSE). Measured on 4xH200 at 43k tokens: an evicted
        # context re-read in 2.5s vs 7.7s recompute (~3x); the win grows with
        # context length, so turn it on for 100k+ reuse-heavy workloads. Set a
        # roomy CONTEXT_LENGTH + MAX_TOTAL_TOKENS, and raise HICACHE_RATIO to hold
        # more in the CPU tier. See README "HiCache (long-context KV offload)".
        if os.getenv("ENABLE_HICACHE", "").lower() in ("true", "1", "yes"):
            command.append("--enable-hierarchical-cache")
            hicache = {
                "--hicache-ratio": os.getenv("HICACHE_RATIO", "2"),
                "--hicache-io-backend": os.getenv("HICACHE_IO_BACKEND", "kernel"),
                "--hicache-mem-layout": os.getenv("HICACHE_MEM_LAYOUT", "page_first"),
                "--hicache-write-policy": os.getenv("HICACHE_WRITE_POLICY", "write_through"),
            }
            for flag, val in hicache.items():
                command.extend([flag, val])
            # page_first layout needs an explicit page size; default it if unset.
            if "--page-size" not in command:
                command.extend(["--page-size", "64"])

        # Generic passthrough: any raw sglang.launch_server flags not covered by
        # the mappings above (so a new flag never needs another fork). Parsed
        # shell-style, e.g. EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --foo bar".
        extra = os.getenv("EXTRA_ARGS", "").strip()
        if extra:
            import shlex
            command.extend(shlex.split(extra))

        print(f"[engine] launch: {' '.join(command)}", flush=True)
        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server shut down.")


class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    async def request_chat_completions(
        self,
        model="default",
        messages=None,
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def request_completions(
        self,
        model="default",
        prompt="The capital of France is",
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def get_models(self):
        response = await self.client.models.list()
        return response
