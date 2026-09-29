"""
local_judges.py — run the LLM judges locally from Unsloth checkpoints.

Two backends behind one call, `JudgePool.chat(model, messages, max_tokens)`:
  gguf → llama.cpp `llama-server` (CUDA), OpenAI-compatible HTTP on localhost
  hf   → transformers in-process (bitsandbytes 4-bit checkpoints)

Only one judge is resident on the GPU at a time: switching to another judge
stops the previous server / frees the previous model first.

Judge registry (all from the Unsloth org on Hugging Face):
  pointwise : unsloth/gemma-4-12B-it-qat-GGUF   (UD-Q4_K_XL, the only text file)
              unsloth/gpt-oss-20b-GGUF           (F16)
  pairwise  : unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit  (bnb NF4, transformers)
              unsloth/Llama-3.2-1B-Instruct-GGUF (F16)
"""
import os, re, gc, sys, time, json, glob, shutil, tarfile, subprocess

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
POINTWISE_JUDGES = [
    {"name": "gemma4_12b_qat", "model": "unsloth/gemma-4-12B-it-qat-GGUF",
     "backend": "gguf", "file": "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
     "reasoning_off": True, "ctx": 8192},
    {"name": "gptoss_20b_f16", "model": "unsloth/gpt-oss-20b-GGUF",
     "backend": "gguf", "file": "gpt-oss-20b-F16.gguf",
     # gpt-oss always reasons; keep it short and give it room to finish.
     "template_kwargs": {"reasoning_effort": "low"}, "min_tokens": 1024, "ctx": 8192},
]
# Pairwise referees: the same two models as the pointwise judges (both via llama-server).
PAIR_JUDGES = POINTWISE_JUDGES
ALL_JUDGES = POINTWISE_JUDGES   # unique models

def _tmp_dir():
    """Scratch space for the llama.cpp source, binary and server logs.
    Colab: /content. Cluster: $FRMEDQA_TMP (set by the job scripts), else ~/.cache/frmedqa."""
    d = os.environ.get("FRMEDQA_TMP") or ("/content" if os.path.isdir("/content")
                                          else os.path.join(os.path.expanduser("~"), ".cache", "frmedqa"))
    os.makedirs(d, exist_ok=True)
    return d


LLAMA_DIR = os.path.join(_tmp_dir(), "llama.cpp")
_THINK = re.compile(r"<think>.*?</think>|<think>.*", re.DOTALL)


class JudgeStartError(RuntimeError):
    """A judge model could not be started."""


class JudgeFatalError(RuntimeError):
    """A judge failed to start repeatedly; stop the whole run (do not save scores)."""


_HINTS = [
    ("unknown model architecture", "llama.cpp is too old for this model: delete tools/llama-server_*.tar.gz "
                                    "and the llama.cpp source folder (LLAMA_DIR), then re-run to rebuild from the latest source."),
    ("error while loading shared libraries", "the cached binary misses libraries: delete tools/llama-server_*.tar.gz "
                                             "and the llama.cpp source folder (LLAMA_DIR), then re-run to rebuild."),
    ("out of memory", "not enough GPU memory: restart the runtime so nothing else holds the GPU, or lower 'ctx'."),
    ("cudaMalloc", "not enough GPU memory: restart the runtime so nothing else holds the GPU, or lower 'ctx'."),
    ("couldn't bind", "the port is taken by an old server: run  !pkill -f llama-server  and re-run."),
    ("Address already in use", "the port is taken by an old server: run  !pkill -f llama-server  and re-run."),
    ("invalid argument", "a command-line flag is not supported by this llama.cpp build (see the log line above)."),
    ("error: unknown", "a command-line flag is not supported by this llama.cpp build (see the log line above)."),
]


def _log_tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        return "(no log)", ""
    tail = "\n".join(lines[-n:])
    hint = next((h for k, h in _HINTS if k.lower() in tail.lower()), "")
    return tail, hint


_HELP_CACHE = {}


def _server_supports(binary, flag):
    """True if this llama-server build accepts `flag` (read once from --help).
    llama.cpp renames or drops options over time, so optional ones are only
    passed when the build lists them."""
    if binary not in _HELP_CACHE:
        try:
            out = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60)
            _HELP_CACHE[binary] = (out.stdout or "") + (out.stderr or "")
        except Exception:
            _HELP_CACHE[binary] = ""
    return re.search(rf"(^|[\s,]){re.escape(flag)}([\s,=]|$)", _HELP_CACHE[binary], flags=re.M) is not None


def _log(logger, msg):
    (logger.info if logger else print)(msg)


def _gpu_slug():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        return re.sub(r"[^A-Za-z0-9]+", "_", out).strip("_") or "gpu"
    except Exception:
        return "gpu"


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp build (once per GPU type, cached on Drive)
# ─────────────────────────────────────────────────────────────────────────────
BIN_DIR = os.path.join(_tmp_dir(), "llama-bin")          # restored / built binary lives here (not in the source tree)


def _run_logged(cmd, cwd=None, what=""):
    """Run a build step; on failure raise with the last lines of its output."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip().splitlines()
        raise JudgeStartError(f"{what} failed (exit {r.returncode}):\n" + "\n".join(out[-30:]))
    return r


def _cuda_arch():
    """GPU compute capability as cmake expects it, e.g. '89' for an L4, '80' for an A100."""
    try:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        return cap.replace(".", "")
    except Exception:
        return "native"


def ensure_llama_server(base_dir, logger=None):
    """Return the path of a CUDA llama-server binary.

    Built from source once (~5–10 min), then cached on Drive as a tarball keyed
    by GPU model, so later sessions only unpack it. Static build → single file.
    The cached binary is unpacked into BIN_DIR, never into the source tree, so a
    failed restore can never leave a source folder without source files.
    """
    binary = os.path.join(BIN_DIR, "llama-server")
    def _runs(p):
        return os.path.exists(p) and subprocess.run([p, "--version"], capture_output=True).returncode == 0
    if _runs(binary):
        return binary
    cache = os.path.join(base_dir, "tools", f"llama-server_{_gpu_slug()}.tar.gz")
    if os.path.exists(cache):
        os.makedirs(BIN_DIR, exist_ok=True)
        with tarfile.open(cache) as t:
            t.extractall(BIN_DIR)
        os.chmod(binary, 0o755)
        if _runs(binary):
            _log(logger, f"  llama-server restored from {cache}")
            return binary
        _log(logger, "  cached llama-server did not run on this runtime; rebuilding")
    else:
        _log(logger, f"  no cached llama-server for GPU '{_gpu_slug()}'")

    arch = _cuda_arch()
    _log(logger, f"  building llama.cpp with CUDA (sm_{arch}, first time only, ~5–10 min)…")
    if not os.path.exists(os.path.join(LLAMA_DIR, "CMakeLists.txt")):
        shutil.rmtree(LLAMA_DIR, ignore_errors=True)          # incomplete / stale folder
        _run_logged(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_DIR],
                    what="git clone llama.cpp")
    shutil.rmtree(os.path.join(LLAMA_DIR, "build"), ignore_errors=True)   # no stale cmake cache
    cfg = ["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DBUILD_SHARED_LIBS=OFF",
           f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DCMAKE_BUILD_TYPE=Release"]
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        cfg.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    _run_logged(cfg, cwd=LLAMA_DIR, what="cmake configure")
    _run_logged(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 4),
                 "--target", "llama-server"], cwd=LLAMA_DIR, what="cmake build")
    os.makedirs(BIN_DIR, exist_ok=True)
    shutil.copy2(os.path.join(LLAMA_DIR, "build", "bin", "llama-server"), binary)
    os.chmod(binary, 0o755)
    if not _runs(binary):
        raise JudgeStartError("llama-server was built but does not run (--version failed).")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with tarfile.open(cache, "w:gz") as t:
        t.add(binary, arcname="llama-server")
    _log(logger, f"  ✓ llama-server built and cached → {cache}")
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────
class _GGUFBackend:
    def __init__(self, spec, base_dir, port=8089, logger=None):
        import requests
        from huggingface_hub import hf_hub_download
        self.spec, self.logger, self.port = spec, logger, port
        self.url = f"http://127.0.0.1:{port}"
        binary = ensure_llama_server(base_dir, logger)
        path = hf_hub_download(spec["model"], spec["file"],
                               cache_dir=os.path.join(base_dir, "gguf_cache"),
                               token=os.environ.get("HF_TOKEN") or None)
        # Core options only. Sampling (temperature 0, seed) is sent with every
        # request, so it needs no command-line flag.
        cmd = [binary, "-m", path, "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "999", "-c", str(spec.get("ctx", 8192)), "-np", "1", "--jinja"]
        self.extra_template = {}
        if spec.get("reasoning_off"):
            if _server_supports(binary, "--reasoning-budget"):
                cmd += ["--reasoning-budget", "0"]
            else:   # older/newer builds without the flag: turn thinking off per request
                self.extra_template = {"enable_thinking": False}
        _log(logger, f"  llama-server options: {' '.join(cmd[3:])}")
        self.logf = open(os.path.join(_tmp_dir(), f"llama-server_{spec['name']}.log"), "w")
        _log(logger, f"  starting llama-server: {spec['model']} [{spec['file']}]")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                self.logf.flush()
                tail, hint = _log_tail(self.logf.name)
                raise JudgeStartError(
                    f"llama-server exited (code {self.proc.returncode}) for {spec['name']}.\n"
                    f"--- last lines of {self.logf.name} ---\n{tail}\n---"
                    + (f"\nLikely fix: {hint}" if hint else ""))
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise TimeoutError("llama-server did not become ready in 15 min")
            time.sleep(3)
        _log(logger, f"  ✓ {spec['name']} ready ({time.time()-t0:.0f}s)")
        self._requests = requests

    def chat(self, messages, max_tokens=256, temperature=0.0):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": 42}
        tkw = {**(self.spec.get("template_kwargs") or {}), **getattr(self, "extra_template", {})}
        if tkw:
            payload["chat_template_kwargs"] = tkw
        for attempt in range(3):
            try:
                r = self._requests.post(self.url + "/v1/chat/completions",
                                        json=payload, timeout=600)
                if r.status_code == 200:
                    msg = r.json()["choices"][0]["message"]
                    return _THINK.sub("", msg.get("content") or "").strip()
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                err = str(e)
            time.sleep(2 + 2 * attempt)
        raise RuntimeError(f"{self.spec['name']} failed: {err}")

    def close(self):
        try:
            self.proc.terminate(); self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
        self.logf.close()


class _HFBackend:
    """bitsandbytes-4bit checkpoints via transformers (text-only use)."""
    def __init__(self, spec, base_dir, logger=None):
        import torch, transformers
        from transformers import AutoProcessor
        self.spec, self.logger = spec, logger
        arch = spec.get("arch")
        if arch:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            if arch not in CONFIG_MAPPING:
                raise RuntimeError(
                    f"transformers {transformers.__version__} does not know '{arch}'. "
                    f"Run  !pip install -U transformers  then restart the runtime.")
        tok = os.environ.get("HF_TOKEN") or None
        cache = os.path.join(base_dir, "hf_cache")
        _log(logger, f"  loading {spec['model']} (transformers, bnb 4-bit)…")
        self.processor = AutoProcessor.from_pretrained(spec["model"], token=tok, cache_dir=cache)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        try:
            from transformers import AutoModelForImageTextToText as _Auto
        except ImportError:
            from transformers import AutoModelForCausalLM as _Auto
        self.model = _Auto.from_pretrained(spec["model"], device_map={"": 0},
                                           torch_dtype=torch.bfloat16,
                                           token=tok, cache_dir=cache).eval()
        self._torch = torch
        _log(logger, f"  ✓ {spec['name']} ready")

    def chat(self, messages, max_tokens=256, temperature=0.0):
        torch = self._torch
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            **self.spec.get("template_kwargs", {}))
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(0)
        with torch.inference_mode():
            from transformers.utils import logging as _hf_log
            _prev = _hf_log.get_verbosity(); _hf_log.set_verbosity_error()   # hide the per-call max_length notice
            try:
                out = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
            finally:
                _hf_log.set_verbosity(_prev)
        raw = self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        # Muse Glimmer: reasoning goes to "to=self", the answer to "to=user".
        if "to=user<|message|>" in raw:
            raw = raw.rsplit("to=user<|message|>", 1)[1]
        elif "<|message|>" in raw:
            raw = raw.rsplit("<|message|>", 1)[1]
        raw = re.split(r"<\|eot\|>|<\|eom\|>|<\|start\|>", raw)[0]
        raw = re.sub(r"<\|[^|]*\|>", "", raw)
        return _THINK.sub("", raw).strip()

    def close(self):
        del self.model, self.processor
        gc.collect(); self._torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Pool: one resident judge at a time
# ─────────────────────────────────────────────────────────────────────────────
class JudgePool:
    def __init__(self, specs, base_dir, logger=None):
        self.specs = {s["model"]: s for s in specs}
        self.base_dir, self.logger = base_dir, logger
        self.current, self.backend = None, None
        self._fails, self._last_err, self._timing = {}, {}, {}

    MAX_START_ATTEMPTS = 2

    def _switch(self, model):
        self.close()
        spec = self.specs[model]
        fails = self._fails.get(model, 0)
        if fails >= self.MAX_START_ATTEMPTS:
            raise JudgeFatalError(f"{spec['name']} failed to start {fails}x; not retrying.\n{self._last_err.get(model, '')}")
        backend = _GGUFBackend if spec["backend"] == "gguf" else _HFBackend
        try:
            self.backend = backend(spec, self.base_dir, logger=self.logger)
        except Exception as e:
            self._fails[model] = fails + 1
            self._last_err[model] = str(e)
            _log(self.logger, f"  ✗ could not start {spec['name']} (attempt {fails + 1}/{self.MAX_START_ATTEMPTS}):\n{e}")
            if fails + 1 >= self.MAX_START_ATTEMPTS:
                raise JudgeFatalError(f"{spec['name']} failed to start {fails + 1}x; stopping.\n{e}") from e
            return self._switch(model)          # one immediate retry
        self._fails[model] = 0
        self.current = model

    def chat(self, model, messages, max_tokens=256, temperature=0.0):
        if model not in self.specs:
            raise KeyError(f"unknown judge {model}; known: {list(self.specs)}")
        if model != self.current:
            self._switch(model)
        spec = self.specs[model]
        t0 = time.time()
        out = self.backend.chat(messages, max_tokens=max(max_tokens, spec.get("min_tokens", 0)),
                                temperature=temperature)
        n, tot = self._timing.get(model, (0, 0.0))
        n, tot = n + 1, tot + (time.time() - t0)
        self._timing[model] = (n, tot)
        if n % 20 == 0:
            _log(self.logger, f"  ⏱ {spec['name']}: {tot / n:.1f} s per generation (avg of {n})")
        return out

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend, self.current = None, None
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
