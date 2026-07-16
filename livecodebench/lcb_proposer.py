"""The PROPOSER for the LiveCodeBench (code) domain: agentic Claude-Code-on-flash generators + a judge,
each an isolated hard-killed subprocess. Mechanics mirror text_to_sql/proposer.py; prompts are rewritten
for competitive-programming code. The label-free signal the proposer reads = PUBLIC sample-test results
(hidden tests are never shown). Run as a worker:  python -m ...lcb_proposer --worker|--picker <task.json>
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import lcb_bridge as bridge
from text_to_sql import claude_wrapper
from .lcb_common import PKG, PKG_DIR, AGENTS_DIR, MH_ROOT

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def desc(res):
    if not res.get("ok"):
        return "ERROR: " + str(res.get("error", ""))[:120]
    if not res.get("rows"):
        return "(no output)"
    return ("; ".join(str(r) for r in res["rows"][:3]))[:160]


def _check_script(new_name, batch_json):
    """A standalone script the generator runs to VERIFY its harness on the batch (public tests only)."""
    return (
        "import json\n"
        f"from {PKG} import lcb_bridge as bridge\n"
        f"from {PKG}.lcb_common import load_harness\n"
        f"batch = json.load(open({json.dumps(batch_json)}, encoding='utf-8'))\n"
        "probs = {p.qid: p for p in bridge.load_problems('test6', stdin_only=True)}\n"
        "ok = 0\n"
        "for i, it in enumerate(batch):\n"
        "    p = probs[it['qid']]\n"
        "    try:\n"
        f"        code = load_harness({json.dumps(new_name)}, p).solve()\n"
        "        res = bridge.run_code(code, p.public_tests)\n"
        "        allp = res['n_total'] > 0 and res['n_pass'] == res['n_total']\n"
        "        ok += allp\n"
        "        print('Q%d [%s/%s] public=%d/%d%s code_len=%d' % (i, it['qid'], it.get('difficulty','?'),\n"
        "              res['n_pass'], res['n_total'], ' ALL' if allp else '', len(code)))\n"
        "    except Exception:\n"
        "        import traceback; print('Q%d HARNESS-CRASH' % i); traceback.print_exc()\n"
        "print('SUMMARY: %d/%d pass ALL public tests' % (ok, len(batch)))\n")


def propose_batch(tk):
    """ONE GENERATOR session (branch gi): deep-read ALL peers' traces for context, but IMPROVE only its
    assigned base harness -> write agents/<new_name>.py. Worker subprocess."""
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
    shutil.copyfile(base_path, cand_path)
    check_path.write_text(_check_script(new_name, tk["batch_json"]))
    cand_rows = []
    for c in candidates:
        relation = "ASSIGNED BASE" if c == base_candidate else "PEER EVIDENCE"
        cand_rows.append(f"  - `{c}` [{relation}]  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)")
    cand_list = "\n".join(cand_rows)
    prompt = (
        f"You are evolving a GENERAL competitive-programming HARNESS: arbitrary Python wrapping a FROZEN weak "
        f"coder (deepseek-flash). There are {len(candidates)} candidate harnesses; each was run on a BATCH of "
        f"problems and its FULL trace saved. You CANNOT see hidden tests — only the PUBLIC sample tests "
        f"(label-free).\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"BRANCH LINEAGE — this is mandatory:\n"
        f"- You are branch G{gi}. Your assigned base is `{base_candidate}`.\n"
        f"- `{cand_path}` has already been copied byte-for-byte from that base. Edit this copy in place.\n"
        f"- Other candidates and their traces are PEER EVIDENCE: learn from and test their mechanisms, but do "
        f"NOT switch your parent or replace your harness wholesale with a peer harness. The output must remain a "
        f"descendant of `{base_candidate}`; if the evidence is insufficient, make a minimal or no-op change "
        f"rather than adopting another branch.\n\n"
        f"DEEP-READ THE FULL TRACES (most important step). Each `{trace_dir}/<harness>__q<j>.md` has, for ONE "
        f"harness on ONE problem: the PROBLEM statement, every coder call (its prompt, response, and whether "
        f"THINKING was on/off), the FINAL code, and the PUBLIC-TEST RESULTS (per test: input, expected, got, "
        f"pass/fail). A harness that passes MORE public tests is better. You may READ THE SOURCE of any peer at "
        f"`{AGENTS_DIR}/<name>.py` to understand a mechanism, but your implementation base remains "
        f"`{base_candidate}` — port only a small general mechanism when its trace evidence supports it. IMPROVE "
        f"your assigned base so it passes MORE public tests across the batch.\n\n"
        f"KEY LEVERS you can write into the harness (target the failures you see in the traces):\n"
        f"  • THINKING CONTROL — `self.llm(prompt, system='', thinking=False|'low'|'medium'|'high')`. This is a "
        f"real trade-off with NO preset answer — DECIDE FROM THE TRACES, do not assume: thinking often improves "
        f"correctness on hard problems, but on a slow endpoint it can run past the solve timeout or emit NO code "
        f"(empty/truncated output). If the traces show empty/truncated coder outputs or timeouts, lower it; if "
        f"they show wrong-but-complete code on hard problems, raising it usually helps. Deciding when and how "
        f"much to think is a core part of the harness.\n"
        f"  • TEST-DRIVEN REPAIR — run `self.run_public(code)`; if some public tests fail, feed the failing "
        f"input/expected/got back to the coder and ask for a fix; retry a few times; keep the best-passing code.\n"
        f"  • ROBUSTNESS BEYOND PUBLIC (underused, high-value) — a program can pass every small public sample "
        f"yet CRASH / TLE on large inputs, which is exactly what the hidden suite tests. `self.stress(code)` "
        f"runs it on SELF-GENERATED max-constraint / boundary inputs (from the problem's stated limits) and "
        f"reports crash/timeout — use it to REJECT a public-passing but non-robust solution and drive repair "
        f"(e.g. switch to a faster algorithm on TLE). All label-free from the problem alone — never touch the "
        f"hidden tests. (Consensus across samples is a WEAK prior, not proof of correctness: it often agrees "
        f"with the truth on easy problems, but on hard ones the model's errors can be correlated and the "
        f"majority is then confidently wrong — so don't decide correctness by vote; prefer the objective "
        f"execution signals, use agreement only as a soft tiebreaker.)\n"
        f"  • ROBUST EXTRACTION — handle empty/garbled model output (re-prompt 'output ONLY the code'); ensure "
        f"the program reads stdin and prints exactly the expected format (no extra prints/debug).\n\n"
        f"Harness API: self.content (problem text); self.public_tests (list of {{input,output}}); "
        f"self.starter_code; self.llm(prompt, system='', thinking=False, n=1); self.run_public(code) -> "
        f"{{n_pass,n_total,results}}; self.stress(code) (self-generated max-constraint inputs); "
        f"bridge.extract_code(text). GENERAL — never hardcode a problem's answer; NEVER touch the hidden "
        f"tests (inputs or outputs) or is_correct; "
        f"no per-problem special-casing. Keep `from ..harness_base import CodeHarness` and `from .. import "
        f"lcb_bridge as bridge`. Do NOT write infinite loops / catastrophic regex (they HANG).\n\n"
        f"VERIFY: edit `{cand_path}`, then run  PYTHONPATH={MH_ROOT} python {check_path}  (prints per-problem "
        f"public pass-counts + a SUMMARY); iterate to RAISE the count without breaking others. Leave the final "
        f"harness at EXACTLY `{cand_path}`."
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return cand_path.exists()


DIVERSITY = [
    (
        "CONSERVATIVE REPAIR ROLE: preserve every parent behavior that has sound trace evidence. "
        "Make the smallest general change that fixes a concrete public-test mismatch or runtime failure "
        "(a wrong stdin/stdout I-O format, a missed edge case, a crash, a TLE on large N). Verify with "
        "self.run_public(code) and self.stress(code). Do not turn one problem-specific interpretation "
        "into a universal algorithm-choice, edge-case, I-O-format, complexity, or thinking-policy rule."
    ),
    (
        "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis from the conservative "
        "proposal. Enumerate plausible algorithm choices, edge cases, exact stdin/stdout I-O formats, and "
        "complexity / TLE risks on large N, then use self.run_public(code) (the label-free public sample "
        "tests) and self.stress(code) (max-constraint / boundary inputs for TLE / overflow) to falsify them. "
        "Implement the simplest surviving general mechanism; do not copy the other proposal's mechanism "
        "merely because it is the current default."
    ),
    (
        "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements among the existing candidates. "
        "Compare the problem statement, bridge.back_translate(code) (a plain-English description of what "
        "each program actually does), and the programs' outputs clause by clause — algorithm choice, edge "
        "cases, exact I-O format, complexity, integer overflow. Use self.run_public(code) and "
        "self.stress(code) as counterexamples to falsify unsupported mechanisms, then synthesize only "
        "independently supported strengths. Preserve runtime robustness, but never equate "
        "'runs / passes-some-public-tests' with full correctness."
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
    """Advance fixed proposer branches once.

    Every branch sees the same active candidate--trace pairs, while branch ``gi`` receives ``branches[gi]``
    as its mechanically pre-copied edit base. The returned list preserves branch order and uses ``None`` for
    a failed child."""
    active = list(dict.fromkeys(branches))
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(json.dumps([{"qid": p.qid, "difficulty": getattr(p, "difficulty", "?")} for p in batch]))
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
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.lcb_proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    _spawn_and_wait(procs, timeout)
    return [name if (AGENTS_DIR / f"{name}.py").exists() else None for name in names]


def run_picker(tk):
    """ONE JUDGE session: deep-read all candidates' traces, pick the harness passing the MOST public tests."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    cand_list = "\n".join(f"  - `{c}`  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)" for c in cands)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate competitive-programming harnesses; each ran on a "
        f"BATCH of problems with its FULL trace saved. You do NOT write/change harnesses — ONLY pick the SINGLE "
        f"best (passing the MOST public tests across the batch). You CANNOT see hidden tests.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each `{trace_dir}/<harness>__q<j>.md` has the PROBLEM, every coder call (+ thinking on/off), the final "
        f"code, and the PUBLIC-TEST RESULTS.\n\n"
        f"HARD RULES:\n"
        f"1. DO NOT EYEBALL. VERIFY by re-running each candidate's code yourself on the PUBLIC tests (the "
        f"GIVEN, label-free samples). Run: PYTHONPATH={MH_ROOT} python -c "
        f"\"from {PKG} import lcb_bridge as b; from {PKG}.lcb_common import load_harness; "
        f"probs={{p.qid:p for p in b.load_problems('test6',stdin_only=True)}}; "
        f"p=probs['<qid>']; c=load_harness('<name>',p).solve(); print('public', b.run_code(c, p.public_tests))\".\n"
        f"2. Rank candidates by public-test passes over the batch. Among the top, prefer the harness whose OWN "
        f"TRACE shows it EARNED correctness rather than guessed it: it reproduced the exact I-O format, checked "
        f"edge cases, and — if it chose to — ran its OWN robustness checks (a harness that stress-tests its "
        f"candidates and repairs the ones that crash/TLE is showing evidence in its favour), and whose intent "
        f"(back-translation) matches the problem. Do NOT run a stress oracle of your OWN to score them — "
        f"robustness testing is a technique a harness MAY own, not a signal you inject at judging time; judge "
        f"only from the given public tests and what each harness's trace demonstrates. Treat consensus across "
        f"candidates as a WEAK prior, not proof: agreement correlates with correctness on easy problems but can "
        f"be confidently wrong on hard ones (correlated errors), so don't pick a harness by vote alone. Never "
        f"use hidden tests or the answer key. Ties -> the simpler/more general harness.\n\n"
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
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.lcb_proposer", "--picker", str(tpath)],
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
