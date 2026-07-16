"""Unified LLM access (controller = strong, solver = weak), all via API.

provider: "openai" -> any OpenAI-compatible endpoint (OpenAI / DeepSeek / vLLM / DashScope-compat ...)
provider: "mock"   -> offline plumbing test; returns canned outputs (NOT real answers),
                      just enough for the whole loop to run without network or keys.
"""
import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Global cap on CONCURRENT API calls across the whole process, regardless of how many nested
# ThreadPools (eval workers × voting candidates) want to fire at once. 60+ simultaneous TLS
# handshakes made the proxy drop connections (SSL UNEXPECTED_EOF). 24 is 2× the known-good 12.
_API_SEM = threading.Semaphore(int(os.environ.get("ASE_MAX_CONCURRENCY", "24")))
# Backstop pool: every API call runs here under a hard wall-clock timeout so a single hung socket
# (proxy stall where the SDK's own timeout never fires) can't pin a semaphore permit forever and
# freeze the whole run. A timed-out call's thread is abandoned (leaked) but the permit is released.
_CALL_POOL = ThreadPoolExecutor(max_workers=128)


@dataclass
class LLMConfig:
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    controller_model: str = "gpt-4o"
    solver_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout: int = 60
    # optional: give the controller/examiner its own key (same base_url). Empty = share api_key_env.
    controller_api_key_env: str = ""
    # HOW this endpoint expresses "thinking", chosen by config not by guessing the model name:
    #   "deepseek" -> send extra_body {"thinking": {"type": "enabled"}, "reasoning_effort": ...}
    #                 (works for deepseek-v* and mimo-v*; both accept the field)
    #   "none"     -> the model has no thinking toggle -> send temperature instead (e.g. plain OpenAI)
    thinking_style: str = "deepseek"
    reasoning_effort: str = "high"


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = None          # solver client (default)
        self._ctrl_client = None     # controller/examiner client (own key if configured)
        if cfg.provider == "openai":
            self._client = self._make_client(cfg.api_key_env)
            self._ctrl_client = (self._make_client(cfg.controller_api_key_env)
                                 if cfg.controller_api_key_env else self._client)

    def _make_client(self, key_env):
        from openai import OpenAI
        key = os.environ.get(key_env, "")
        if not key:
            raise RuntimeError(
                f"env ${key_env} is empty. export it, or set llm.provider=mock for an offline test."
            )
        return OpenAI(base_url=self.cfg.base_url, api_key=key, timeout=self.cfg.request_timeout)

    def chat(self, purpose, system, user, model_role="controller", n=1, temperature=None):
        """Return n string completions via n separate calls.

        Many OpenAI-compatible endpoints (incl. DeepSeek) do NOT support the `n` parameter,
        so we loop instead of asking for n choices in one request. `purpose` is only used by
        the mock provider.
        """
        if self.cfg.provider == "mock":
            return [_mock(purpose, user) for _ in range(n)]
        if model_role == "controller":
            model, client = self.cfg.controller_model, self._ctrl_client
        else:
            model, client = self.cfg.solver_model, self._client
        temp = self.cfg.temperature if temperature is None else temperature
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        n = max(n, 1)
        if n == 1:
            return [self._complete(client, model, messages, temp)]
        # endpoint lacks n>1 -> fire the n candidates CONCURRENTLY instead of looping sequentially
        # (voting was ~5x slower than it needed to be — each voted solve waited on 5 serial calls)
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(lambda _: self._complete(client, model, messages, temp), range(n)))

    def _complete(self, client, model, messages, temp, retries=4):
        last = None
        hard = self.cfg.request_timeout + 30         # wall-clock backstop if the SDK timeout never fires
        for attempt in range(retries + 1):
            _API_SEM.acquire()                       # bound total in-flight calls (anti connection-storm)
            try:
                create_kw = dict(model=model, messages=messages,
                                 max_tokens=int(os.environ.get("ASE_MAX_TOKENS", "32000")))  # generous so
                                 # thinking-on never truncates before the answer
                if self.cfg.thinking_style == "deepseek":    # native THINKING mode (deepseek-v*, mimo-v*)
                    create_kw["extra_body"] = {"thinking": {"type": "enabled"},
                                               "reasoning_effort": self.cfg.reasoning_effort}
                else:                                        # no thinking toggle -> keep temperature
                    create_kw["temperature"] = temp
                fut = _CALL_POOL.submit(client.chat.completions.create, **create_kw)
                resp = fut.result(timeout=hard)      # FutureTimeout if the call hangs past `hard`
                msg = resp.choices[0].message
                # Some endpoints (or a gateway that ignores thinking:disabled) route the whole reply into
                # `reasoning_content`, leaving `content` empty. Fall back to it rather than drop the work.
                return msg.content or getattr(msg, "reasoning_content", "") or ""
            except Exception as e:  # noqa: BLE001 - incl. timeout; retry transient errors with backoff
                last = e
                time.sleep(1.0 * (attempt + 1))
            finally:
                _API_SEM.release()                   # ALWAYS free the permit, even on a hung/abandoned call
        raise last

    def one(self, purpose, system, user, model_role="controller", temperature=None):
        return self.chat(purpose, system, user, model_role, n=1, temperature=temperature)[0]


# ---------- parsing helpers ----------

def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"((?:with|select)\b.*)", text, re.S | re.I)
    return (m.group(1).strip() if m else text.strip())


def extract_json(text):
    m = re.search(r"```json\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    raw = m.group(1) if m else text
    return json.loads(raw)


def extract_code(text):
    """Pull a Python code block out of a model reply. The code itself may contain inner ``` fences
    (e.g. in its own prompt strings like '```sql ... ```'), so match from the OPENING fence to the
    LAST ``` in the reply rather than a non-greedy first-close (which would truncate mid-string)."""
    fence = re.search(r"```[ \t]*(?:python|py)?[ \t]*\r?\n", text, re.I)
    if fence:
        end = text.rfind("```")
        return (text[fence.end():end].strip() if end > fence.end() else text[fence.end():].strip())
    m = re.search(r"```(.*?)```", text, re.S)
    # No fenced block -> NO code. Returning the whole reply here would pass truncated prose
    # (e.g. a reasoning trace cut off before it ever opened a fence) off as "code".
    return m.group(1).strip() if m else ""


# ---------- mock provider (offline plumbing only) ----------

def _mock(purpose, user):
    tables = re.findall(r"Table (\w+)\(", user)
    t0 = tables[0] if tables else "t"
    if purpose == "solve_sql":
        # deliberately "wrong" (a plain select) so the smoke test exercises failures+ladder
        return f"```sql\nSELECT * FROM {t0} LIMIT 5\n```"
    if purpose == "verify_solve":
        # matches propose_sql so probes pass the self-consistency gate offline
        return f"```sql\nSELECT count(*) FROM {t0}\n```"
    if purpose == "propose_sql":
        return json.dumps({"question": "How many rows are in the main table?",
                           "sql": f"SELECT count(*) FROM {t0}"})
    if purpose == "sql_to_nl":
        return "How many rows are in the main table?"
    if purpose == "rephrase":
        return "Roughly how many entries are we talking about here?"
    if purpose == "gen_hint":
        return "Verify every table and column name against the schema before writing SQL."
    if purpose == "gen_fewshot":
        return json.dumps({"question": "How many rows in the first table?", "sql": f"SELECT count(*) FROM {t0}"})
    if purpose == "self_debug":
        return f"```sql\nSELECT * FROM {t0} LIMIT 5\n```"
    return "OK"
