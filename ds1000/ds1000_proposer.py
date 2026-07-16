"""The PROPOSER for the DS-1000 (data-science coding) domain: agentic Claude-Code-on-flash generators + a
judge, each an isolated hard-killed subprocess. Mechanics mirror livecodebench/lcb_proposer.py; prompts are
rewritten for data-science snippet generation. The label-free signal the proposer reads = the SELF-CHECK
execution (does the candidate run on a constructed example input? what does it output?) — the gold hidden test
is never shown. Run as a worker:  python -m ...ds1000_proposer --worker|--picker <task.json>
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import ds1000_bridge as bridge
from text_to_sql import claude_wrapper
from .ds1000_common import PKG, PKG_DIR, AGENTS_DIR, MH_ROOT

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def _check_script(new_name, batch_json):
    """A standalone script the generator runs to VERIFY its harness on the batch (LABEL-FREE self-check only;
    NEVER calls bridge.is_correct / gold)."""
    return (
        "import json\n"
        f"from {PKG} import ds1000_bridge as bridge\n"
        f"from {PKG}.ds1000_common import load_harness\n"
        f"batch = json.load(open({json.dumps(batch_json)}, encoding='utf-8'))\n"
        "pids = [str(it['pid']) for it in batch]\n"
        "probs = {p.pid: p for p in bridge.load_problems(ids=pids)}\n"
        "ran = 0\n"
        "for i, it in enumerate(batch):\n"
        "    p = probs[str(it['pid'])]\n"
        "    try:\n"
        f"        code = load_harness({json.dumps(new_name)}, p).solve()\n"
        "        sc = bridge.selfcheck(code, p)\n"
        "        ran += bool(sc['ran'])\n"
        "        print('Q%d [%s/%s] ran=%s code_len=%d out=%r' % (i, it['pid'], p.library,\n"
        "              sc['ran'], len(code), str(sc['output'])[:120]))\n"
        "    except Exception:\n"
        "        import traceback; print('Q%d HARNESS-CRASH' % i); traceback.print_exc()\n"
        "print('SUMMARY: %d/%d ran clean in self-check' % (ran, len(batch)))\n")


def propose_batch(tk):
    """One branch generator: deep-read ALL peer traces for context, but IMPROVE only its assigned base ->
    write agents/<new_name>.py. Worker subprocess."""
    candidates, trace_dir = tk["candidates"], tk["trace_dir"]
    new_name, gi, tag, run_dir = tk["new_name"], tk["gi"], tk["tag"], tk["run_dir"]
    base_candidate = tk.get("base_candidate", candidates[0])
    if base_candidate not in candidates:
        raise ValueError(f"branch base {base_candidate!r} is not in active candidates")
    cand_path = AGENTS_DIR / f"{new_name}.py"
    base_path = AGENTS_DIR / f"{base_candidate}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    check_path = Path(run_dir) / f"check_{tag}_g{gi}.py"
    cand_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, cand_path)                 # child starts byte-for-byte from its assigned base
    check_path.write_text(_check_script(new_name, tk["batch_json"]))
    cand_rows = []
    for c in candidates:
        relation = "ASSIGNED BASE" if c == base_candidate else "PEER EVIDENCE"
        cand_rows.append(f"  - `{c}` [{relation}]  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)")
    cand_list = "\n".join(cand_rows)
    prompt = (
        f"You are evolving a GENERAL data-science-coding HARNESS (DS-1000): arbitrary Python wrapping a FROZEN "
        f"coder (deepseek-flash) that writes pandas/numpy/scipy/sklearn/torch/tf snippets which set a `result` "
        f"variable. There are {len(candidates)} candidate harnesses; each was run on a BATCH of problems and its "
        f"FULL trace saved. You CANNOT see the hidden gold test — only the prompt, the candidate's SELF-CHECK "
        f"execution (does it run? what does it output on a constructed example input?), and a back-translation.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"BRANCH LINEAGE — this is mandatory:\n"
        f"- You are branch G{gi}. Your assigned base is `{base_candidate}`.\n"
        f"- `{cand_path}` has already been copied byte-for-byte from that base. Edit this copy in place.\n"
        f"- Other candidates and traces are PEER EVIDENCE: learn from and test their mechanisms, but do not "
        f"switch your parent or replace the target wholesale with a peer harness.\n"
        f"- The output must remain a descendant of `{base_candidate}`. If evidence is insufficient, make a "
        f"minimal or no-op refactor rather than adopting another branch.\n\n"
        f"DEEP-READ THE FULL TRACES (most important step). Each `{trace_dir}/<harness>__q<j>.md` has, for ONE "
        f"harness on ONE problem: the PROBLEM, every coder call (its prompt, response, and whether THINKING was "
        f"on/off), the FINAL code, the SELF-CHECK result (ran? error? output?), and a BACK-TRANSLATION. A harness "
        f"whose solutions RUN CLEAN in self-check and whose output matches the problem's stated example is better. "
        f"You may READ THE SOURCE of ANY candidate at `{AGENTS_DIR}/<name>.py` to understand behavioral "
        f"differences, but your implementation base remains `{base_candidate}` — IMPROVE it so it solves MORE of "
        f"the batch; port a peer mechanism only when its trace evidence supports it.\n\n"
        f"  *** #1 PITFALL — INSERTION-ONLY CODE ***: DS-1000 is an INSERTION task. The problem already DEFINES "
        f"the input variables (e.g. df, a, X) in its `<code>` block; the solution must be ONLY the lines that "
        f"compute `result` FROM those given variables. A solution that REDEFINES an input (e.g. `a = np.array(...)`, "
        f"`df = pd.DataFrame(...)`), adds its own example data, or wraps everything in a function + calls it on "
        f"hardcoded values will RUN CLEAN in self-check (its hardcoded data matches the example) but FAIL the "
        f"hidden test, which supplies DIFFERENT inputs. The self-check now reports `redefines` (the input vars a "
        f"solution hardcodes); `redefines` MUST be empty. This is the single biggest cause of bare-beating "
        f"harnesses turning WORSE. The bare harness is correct here — do not regress it.\n"
        f"KEY LEVERS you can write into the harness (target the failures you see in the traces):\n"
        f"  • PROMPT ENGINEERING — sharpen the generation prompt/system: demand the INSERTION-ONLY `result` snippet "
        f"(use the EXACT given context variable names, NEVER redefine them, no example data, no def wrapper, no "
        f"prose), with the right library idioms.\n"
        f"  • SELF-TEST-DRIVEN REPAIR — `code=self.llm(...)`; `sc=self.selfcheck(code)`; if `not sc['ran']` OR "
        f"`sc['redefines']` is non-empty, feed that back to the coder ('do NOT redefine {{redefines}}; use the "
        f"given variables') and retry; keep code that runs clean AND has empty `redefines`.\n"
        f"  • THINKING CONTROL — `self.llm(prompt, system='', thinking=False|'low'|'medium'|'high')`. No preset "
        f"best — decide from the traces: thinking often helps on harder problems but is slower; if you see "
        f"empty/truncated output, lower it.\n"
        f"  • ROBUST EXTRACTION — handle empty/garbled output (re-prompt 'output ONLY the code'); strip prose.\n\n"
        f"Harness API: self.problem; self.prompt; self.library; self.llm(prompt, system='', thinking=False, n=1); "
        f"self.selfcheck(code) -> {{ran,error,output,redefines}}; bridge.extract_code(text). GENERAL — never hardcode a "
        f"problem's answer; no per-problem special-casing. Keep `from ..harness_base import DS1000Harness` and "
        f"`from .. import ds1000_bridge as bridge`. Do NOT write infinite loops / catastrophic regex (they HANG).\n\n"
        f"VERIFY: edit `{cand_path}`, then run  PYTHONPATH={MH_ROOT} python {check_path}  (prints per-problem "
        f"self-check ran-status + a SUMMARY); iterate to RAISE how many run clean without breaking others. Leave "
        f"the final harness at EXACTLY `{cand_path}`."
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return cand_path.exists()


DIVERSITY = [
    (
        "CONSERVATIVE REPAIR ROLE: preserve every parent behavior that has sound trace evidence. "
        "Make the smallest general change that fixes a concrete self-check mismatch or runtime failure. "
        "Do not turn one batch-specific interpretation into a universal result-shape/dtype, library-idiom "
        "(pandas/numpy/scipy/sklearn), context-variable-usage, or insertion-only output-format rule."
    ),
    (
        "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis from the conservative "
        "proposal. Enumerate plausible result shapes/dtypes, library idioms (pandas/numpy/scipy/sklearn), and "
        "ways of computing `result` from the GIVEN context variables (use df/a/X as provided — never redefine "
        "them), then use `self.selfcheck(code)` on a constructed example input to falsify them. Implement the "
        "simplest surviving general mechanism; do not copy the other proposal's mechanism merely because it is "
        "the current default."
    ),
    (
        "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements among the existing candidates. "
        "Compare the problem, the back-translation (`bridge.back_translate(code)`), the self-check output, "
        "result shape/dtype, and insertion-only output format clause by clause. Use `self.selfcheck(code)` "
        "counterexamples on constructed example inputs to falsify unsupported mechanisms, then synthesize only "
        "independently supported strengths. Preserve runtime robustness, but never equate 'runs / non-empty "
        "output' with semantic correctness."
    ),
]


def _spawn_and_wait(procs, timeout):
    hard = time.time() + timeout + 120
    for p in procs:
        try:
            p.wait(timeout=max(1, hard - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def sample_branches(branches, trace_dir, run_dir, tag, run_name, batch, model, timeout):
    """Advance fixed proposer branches once. Every branch sees the same active harness--trace pairs, while
    branch ``gi`` receives ``branches[gi]`` as its mechanically pre-copied edit base. The returned list
    preserves branch order and uses ``None`` for a failed child."""
    active = list(dict.fromkeys(branches))
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(json.dumps([{"pid": p.pid, "library": getattr(p, "library", "?")} for p in batch]))
    names = [f"cand_{run_name}_{tag}_g{gi}" for gi in range(len(branches))]
    procs = []
    for gi, (base_candidate, name) in enumerate(zip(branches, names)):
        (AGENTS_DIR / f"{name}.py").unlink(missing_ok=True)
        task = {"kind": "generate_batch", "candidates": active, "base_candidate": base_candidate,
                "trace_dir": str(trace_dir), "batch_json": str(batch_json), "new_name": name,
                "run_dir": str(run_dir), "tag": tag, "gi": gi, "model": model, "timeout": timeout,
                "diversity": DIVERSITY[gi % len(DIVERSITY)]}
        tpath = Path(run_dir) / f"task_{tag}_g{gi}.json"
        tpath.write_text(json.dumps(task))
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.ds1000_proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    _spawn_and_wait(procs, timeout)
    return [name if (AGENTS_DIR / f"{name}.py").exists() else None for name in names]


def run_picker(tk):
    """ONE JUDGE session: deep-read all candidates' traces, pick the harness whose solutions look most likely
    correct from LABEL-FREE evidence (runs clean in self-check, output matches the prompt's example)."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    cand_list = "\n".join(f"  - `{c}`  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)" for c in cands)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate data-science-coding harnesses (DS-1000); each ran "
        f"on a BATCH of problems with its FULL trace saved. You do NOT write/change harnesses — ONLY pick the "
        f"SINGLE best. You CANNOT see the hidden gold test.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each `{trace_dir}/<harness>__q<j>.md` has the PROBLEM, every coder call (+ thinking on/off), the final "
        f"code, the SELF-CHECK (ran? error? output?), and a BACK-TRANSLATION.\n\n"
        f"HARD RULES:\n"
        f"1. READ the recorded evidence; DO NOT re-generate. Each trace ALREADY contains, per problem, the "
        f"SELF-CHECK (ran? error? output?) and a BACK-TRANSLATION. Base the verdict ONLY on what is recorded. "
        f"Do NOT call `solve()` or re-run the harness — generation is non-deterministic and re-running gives "
        f"different, misleading numbers (and wastes calls). NEVER call b.is_correct.\n"
        f"2. DISQUALIFY HARDCODERS FIRST. Each self-check reports `redefines_input`. If it is NON-EMPTY, the "
        f"solution HARDCODES the problem's input variables instead of using the given ones — it runs clean here "
        f"but is a near-certain FAIL on the hidden test (different inputs). Treat any such solution as WRONG, no "
        f"matter how clean it looks. A harness that hardcodes on problems where `bare` does not is WORSE than bare.\n"
        f"3. Among the rest, the decisive signal is whether each solution DOES WHAT THE PROBLEM ASKS — not mere "
        f"runnability. Use (i) SELF-CHECK ran/error — a solution that CRASHES (ran=False) is wrong; (ii) "
        f"BACK-TRANSLATION — a plain-English description of what each candidate's code computes; compare it to the "
        f"PROBLEM: a candidate whose back-translation matches the requested operation (right columns/axis/grouping/"
        f"transform) is likely correct, one that describes a DIFFERENT computation is wrong. Across the batch, pick "
        f"the harness whose solutions RUN CLEAN, do NOT hardcode (empty redefines), and whose back-translations "
        f"best match the problems. A harness that runs clean on MORE problems than `bare` WITHOUT hardcoding and "
        f"whose back-translations fit is better than `bare` — do not default to "
        f"`bare` out of caution. Ties -> the simpler/more general harness.\n\n"
        f"Write ONLY the chosen harness's exact NAME to `{choice_path}` (run:  echo <name> > {choice_path} ). Then STOP.")
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(tk["run_dir"]) / "claude_sessions"), name=f"judge_{tk['tag']}",
                       timeout_seconds=tk["timeout"], progress=False)


def pick_batch(candidates, trace_dir, run_dir, tag, model, timeout):
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    choice_path = Path(run_dir) / f"judge_{tag}.txt"
    choice_path.unlink(missing_ok=True)
    tpath = Path(run_dir) / f"judgetask_{tag}.json"
    tpath.write_text(json.dumps({"candidates": candidates, "trace_dir": str(trace_dir),
                                 "choice_path": str(choice_path), "run_dir": str(run_dir), "tag": tag,
                                 "model": model, "timeout": timeout}))
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.ds1000_proposer", "--picker", str(tpath)],
                         cwd=str(MH_ROOT), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _spawn_and_wait([p], timeout)
    if choice_path.exists():
        nm = choice_path.read_text().strip().split()
        if nm and nm[0] in candidates:
            return nm[0]
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--picker")
    a = ap.parse_args()
    if a.picker:
        run_picker(json.load(open(a.picker)))
    else:
        propose_batch(json.load(open(a.worker)))
