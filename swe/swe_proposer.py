"""The PROPOSER for the SWE-bench Verified domain: agentic Claude-Code-on-flash generators + a judge, each
an isolated hard-killed subprocess. Mechanics mirror livecodebench/lcb_proposer.py; prompts are rewritten for
SWE-bench issue-fixing. The label-free signal the proposer reads = the agent's own trajectory + final patch
(gold hidden tests are never shown). Generators do NOT self-run their harness (a Docker rollout is far too
expensive) — they deep-read the traces and WRITE agents/<new_name>.py; verification happens later in OBSERVE.
Run as a worker:  python -m ...swe_proposer --worker|--picker <task.json>
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import swe_bridge as bridge
from text_to_sql import claude_wrapper  # shared agentic Claude-Code wrapper, sibling package under REPO_ROOT
from .swe_common import PKG, PKG_DIR, AGENTS_DIR, MH_ROOT

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def propose_batch(tk):
    """One branch generator: deep-read all peer traces, but IMPROVE only its assigned base harness ->
    write agents/<new_name>.py. Worker subprocess. (No self-run: a Docker rollout is too expensive.)"""
    candidates, trace_dir = tk["candidates"], tk["trace_dir"]
    new_name, gi, tag, run_dir = tk["new_name"], tk["gi"], tk["tag"], tk["run_dir"]
    base_candidate = tk.get("base_candidate", candidates[0])
    if base_candidate not in candidates:
        raise ValueError(f"branch base {base_candidate!r} is not in active candidates")
    cand_path = AGENTS_DIR / f"{new_name}.py"
    base_path = AGENTS_DIR / f"{base_candidate}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    cand_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, cand_path)                                   # child starts byte-for-byte from its assigned base
    cand_rows = []
    for c in candidates:
        relation = "ASSIGNED BASE" if c == base_candidate else "PEER EVIDENCE"
        cand_rows.append(f"  - `{c}` [{relation}] (traces: {trace_dir}/{c}__i*.md ; code: {AGENTS_DIR}/{c}.py)")
    cand_list = "\n".join(cand_rows)
    prompt = (
        f"You are evolving a GENERAL SWE-bench-fixing HARNESS: arbitrary Python wrapping a FROZEN coding agent "
        f"(mini-swe-agent driving deepseek-flash in a bash/Docker sandbox). There are {len(candidates)} candidate "
        f"harnesses; each ran on a BATCH of GitHub issues and its FULL trace (agent trajectory + final patch) is "
        f"saved. You CANNOT see the hidden gold tests — only the agent's own in-sandbox execution (its "
        f"reproduction scripts, test runs) and whether it produced a focused patch.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"BRANCH LINEAGE — this is mandatory:\n"
        f"- You are branch G{gi}. Your assigned base is `{base_candidate}`.\n"
        f"- `{cand_path}` has already been copied byte-for-byte from that base. Edit this copy in place.\n"
        f"- Other candidates and traces are PEER EVIDENCE: learn from and test their mechanisms, but do not "
        f"switch your parent or replace the target wholesale with a peer harness.\n"
        f"- The output must remain a descendant of `{base_candidate}`. If evidence is insufficient, make a "
        f"minimal or no-op refactor rather than adopting another branch.\n\n"
        f"DEEP-READ THE FULL TRACES (most important step). Each `{trace_dir}/<harness>__i<j>.md` has, for ONE "
        f"harness on ONE issue: the ISSUE text, the agent's TRAJECTORY (its bash commands, reproduction scripts, "
        f"test runs as observations), and the FINAL PATCH. You may READ THE SOURCE of ANY candidate at "
        f"`{AGENTS_DIR}/<name>.py` to understand behavioral differences, but your implementation base remains "
        f"`{base_candidate}`; you may port a small general mechanism from a peer only when its trace evidence "
        f"supports it. IMPROVE your assigned base so it fixes MORE of the batch issues.\n\n"
        f"KEY LEVERS you can write into the harness (target the failures you see in the traces):\n"
        f"  • PROMPT ENGINEERING — rewrite the `system_template`/`instance_template` you pass to "
        f"`self.run_agent(...)` to push a better workflow: ALWAYS reproduce the bug first; locate the real ROOT "
        f"CAUSE not a surface patch; run the repo's existing tests before submitting; keep the patch MINIMAL and "
        f"only touch SOURCE files (never tests).\n"
        f"  • STEP/TIME BUDGET — `step_limit` / `wall_time` passed to `self.run_agent(...)`.\n"
        f"  • POST-ROLLOUT VERIFY->REPAIR — after `patch = self.run_agent(...)`, if the trajectory shows the issue "
        f"wasn't actually reproduced/fixed (or the patch is empty), call `self.run_agent(...)` AGAIN with a sharper "
        f"instance_template that includes what went wrong.\n"
        f"  • ROBUST SUBMISSION — ensure a non-empty `git diff` patch comes back.\n\n"
        f"Harness API: self.instance (dict); self.problem (issue text); self.repo; "
        f"self.run_agent(system_template=None, instance_template=None, step_limit=80, wall_time=1500) -> patch; "
        f"self._trace. GENERAL — never hardcode an instance's answer; no per-instance special-casing. Keep "
        f"`from ..harness_base import SWEHarness` and `from .. import swe_bridge as bridge`.\n\n"
        f"Do NOT run the harness (a Docker rollout is far too expensive) — just WRITE it. Edit ONLY `{cand_path}`; "
        f"verification happens later in the loop's OBSERVE."
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return cand_path.exists()


DIVERSITY = [
    (
        "CONSERVATIVE REPAIR ROLE: preserve every parent behavior that has sound trace evidence. "
        "Make the smallest general change that fixes a concrete failure you observed in the rollout "
        "trajectory (the issue was never reproduced, the reproduction still shows the wrong behavior "
        "after the patch, or the patch came back empty). Do not turn one instance-specific fix into a "
        "universal reproduce-first, localization, minimal-fix, or submission rule."
    ),
    (
        "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis from the conservative "
        "proposal. Enumerate plausible root-cause localizations, minimal single-source-file fixes (never "
        "touch tests/), and post-rollout verify->repair strategies, then use the agent's own reproduce-first "
        "script and post-rollout verification (re-run the reproduction, check whether the wrong behavior is "
        "gone) and the observed rollout trajectory to falsify them. Implement the simplest surviving general "
        "mechanism; do not copy the other proposal's mechanism merely because it is the current default."
    ),
    (
        "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements among the existing candidates. "
        "Compare the issue, the agent's reproduction/verification evidence, the rollout trajectory, and the "
        "final patch clause by clause. Use the reproduce-first script and post-rollout reproduction re-runs "
        "as counterexamples to falsify unsupported mechanisms, then synthesize only independently supported "
        "strengths. Preserve robust submission and empty-patch 'paralysis' avoidance, but never equate "
        "producing a non-empty patch with actually fixing the issue."
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
    """Advance fixed proposer branches once. Every branch sees the same active candidate--trace pairs,
    while branch ``gi`` receives ``branches[gi]`` as its mechanically pre-copied edit base. The returned
    list preserves branch order and uses ``None`` for a failed child."""
    active = list(dict.fromkeys(branches))
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(json.dumps([{"instance_id": inst["instance_id"], "repo": inst["repo"]} for inst in batch]))
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
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.swe_proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    _spawn_and_wait(procs, timeout)
    return [name if (AGENTS_DIR / f"{name}.py").exists() else None for name in names]


def run_picker(tk):
    """ONE JUDGE session: deep-read all candidates' traces, pick the harness whose patches best fix the issues."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    cand_list = "\n".join(f"  - `{c}`  (traces: {trace_dir}/{c}__i*.md ; code: {AGENTS_DIR}/{c}.py)" for c in cands)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate SWE-bench-fixing harnesses; each ran on a BATCH of "
        f"GitHub issues with its FULL trace saved. You do NOT write/change harnesses — ONLY pick the SINGLE best "
        f"(whose patches best fix the issues). You CANNOT see the hidden gold tests.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each `{trace_dir}/<harness>__i<j>.md` has the ISSUE, the agent's TRAJECTORY (reproduction scripts + test "
        f"runs), and the FINAL PATCH. Judge ONLY from this label-free evidence: did the agent REPRODUCE the issue "
        f"and VERIFY the fix? is the patch NON-EMPTY, focused on the right source file, and NOT touching tests?\n\n"
        f"HARD RULES:\n"
        f"1. To save cost you do NOT re-run anything — decide from the traces.\n"
        f"2. Pick the harness whose patches are most likely to genuinely resolve the issues. Ties -> prefer the "
        f"simpler/more general harness.\n\n"
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
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.swe_proposer", "--picker", str(tpath)],
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
