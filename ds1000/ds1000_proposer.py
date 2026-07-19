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
        "        ran += bool(sc['ran'])   # sc['checkable'] is False when no probe could be built\n"
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
        f"on/off), the FINAL code, the SELF-CHECK result (ran? error? output? redefines?), and a BACK-TRANSLATION. "
        f"You may READ THE SOURCE of ANY candidate at `{AGENTS_DIR}/<name>.py` to understand behavioral "
        f"differences, but your implementation base remains `{base_candidate}` — improve it so its solutions are "
        f"more often CORRECT; port a peer mechanism only when its trace evidence supports it.\n"
        f"Correctness is NOT observable here, so you cannot optimise it directly. Therefore prefer a change "
        f"justified by a REPRODUCIBLE CAUSAL MECHANISM you can point to in a trace ('the reply was indented, the "
        f"extractor mangled it, here is the resulting SyntaxError') over a change justified by a COUNT going up. "
        f"Counts on a 10-problem batch move by +-3 from solver nondeterminism alone: three harnesses with "
        f"byte-identical prompts scored 6, 3 and 4 last round. A count that moves less than that measured "
        f"nothing.\n\n"
        f"DO NOT BREAK WHAT ALREADY WORKS. Every problem your base already handles is something you can lose, "
        f"and a lost one costs exactly as much as a gained one. Before you ship, go problem by problem: for each "
        f"one where your base produced a clean solution, state whether your change can alter its behaviour. If a "
        f"mechanism can fire on a problem the base already handled, gate it so it cannot, or drop it. Last round "
        f"nine candidates were produced; not one turned a failing problem into a passing one, and every score "
        f"change came from breaking problems the base had already solved.\n"
        f"A NO-CHANGE result is legitimate and often correct. You are not required to ship an edit. Shipping a "
        f"speculative mechanism is worse than shipping nothing.\n\n"
        f"TASK CONTRACT (what the benchmark requires, not advice): DS-1000 is an INSERTION task. The problem "
        f"already DEFINES its input variables (e.g. df, a, X) inside its `<code>` block, and the graded solution "
        f"is ONLY the lines that compute `result` FROM those given variables. The hidden test re-runs the same "
        f"solution against DIFFERENT inputs.\n"
        f"THE WORKED EXAMPLE IN A PROBLEM STATEMENT IS ILLUSTRATIVE, NOT THE SPEC. It is frequently formatted "
        f"differently from what the grader compares against — DS-1000 statements routinely print an ID column "
        f"where the index appears. Never add a mechanism, system-prompt line or retry hint that pushes the coder "
        f"to REPRODUCE the printed example: index labels, row order, column order, dtype and rounding must come "
        f"from the problem TEXT. Measured: six candidates lost pid 75 by emitting `reset_index(drop=True)` to "
        f"make the output look like a printed sample.\n\n"
        f"WHAT THE SELF-CHECK ACTUALLY MEASURES (know its limits): `ran` only means the snippet EXECUTED without "
        f"raising — it does NOT check that `result` is correct, because no correct answer is available anywhere "
        f"label-free. A large share of the solutions it lets through are still wrong on the hidden test. Treat "
        f"`ran=True` as a weak filter, never as proof — and equally, do not read it as evidence that your base is "
        f"broken and must be changed.\n\n"
        f"There is NO prescribed recipe and no menu of techniques — the design is ENTIRELY yours. The traces are "
        f"your only evidence of how each harness actually behaves. Read them, diagnose the failures YOURSELF, and "
        f"decide what (if anything) to change. Infer what helps from the trace evidence, not from assumptions.\n\n"
        f"YOUR ACTION SPACE IS THE WHOLE PYTHON FILE, not a prompt string. `solve()` may do anything Python "
        f"can: decompose the problem into stages, plan then implement, ask the coder several times in "
        f"different roles, generate competing solutions and adjudicate between them, build and run your own "
        f"tests, loop until a condition you define holds. Every candidate produced so far only tweaked the "
        f"system prompt and the retry branch — that is a fraction of what a harness is, and it happened "
        f"because `selfcheck` used to be the only execution available, so a boolean `ran` was the only thing "
        f"any change could move.\n"
        f"THAT LIMIT IS GONE: `self.run(script, timeout=15) -> (rc, stdout, stderr)` executes ANY Python you "
        f"compose. It is how you MANUFACTURE evidence rather than only consuming it, and none of it needs an "
        f"answer key. For instance — these are illustrations, not a menu, and you should invent better: run a "
        f"solution on several inputs you construct from the problem prose and see whether it survives all of "
        f"them; obtain TWO independent solutions and DIFF their outputs on the same input, since disagreement "
        f"PROVES at least one is wrong; encode an invariant you can read straight off the problem text (a "
        f"filter cannot lengthen its input; a sort preserves multiset; the output dtype the prose names) and "
        f"assert it. Note the asymmetry that makes these worth having: DISAGREEMENT and violated invariants "
        f"are near-proof of a fault, whereas agreement is weak (the frozen coder's errors are correlated — "
        f"several attempts are routinely wrong in the SAME way).\n"
        f"Harness API — these primitives EXIST; WHETHER, WHEN and HOW to use them is entirely up to you to decide "
        f"from the traces: self.problem; self.prompt; self.library; "
        f"self.llm(prompt, system='', thinking=False|'low'|'medium'|'high', n=1) -> the frozen coder; "
        f"self.run(script, timeout=15) -> (rc, stdout, stderr), arbitrary label-free execution; "
        f"self.selfcheck(code) -> {{checkable, ran, error, output, redefines}}. `checkable` is False when the "
        f"probe could NOT be built (the prompt supplies no concrete example input) — that is ABSENCE OF "
        f"EVIDENCE, not a failed run, and treating it as a failure makes the model rewrite correct code. "
        f"`redefines` lists input variables the "
        f"solution assigned to itself; bridge.extract_code(text); bridge.back_translate(code) -> plain-English "
        f"description of what the code computes. NOTE on back_translate: it is a DESCRIPTION produced by the "
        f"same weak model, not a verdict. Any mechanism that REJECTS or REWRITES a snippet which already ran "
        f"clean needs evidence that the rewrite is right more often than wrong, and no such evidence is "
        f"available to you — so default to never touching a snippet that ran. "
        f"GENERAL — never hardcode a problem's answer; no per-problem "
        f"special-casing. Keep `from ..harness_base import DS1000Harness` and "
        f"`from .. import ds1000_bridge as bridge`. Do NOT write infinite loops / catastrophic regex (they HANG).\n\n"
        f"VERIFY: edit `{cand_path}`, then run  PYTHONPATH={MH_ROOT} python {check_path}  (prints per-problem "
        f"self-check ran-status + a SUMMARY).\n"
        f"THE SUMMARY COUNT IS A SMOKE TEST, NOT A SCORE. Do NOT tune your harness to raise it. It counts only "
        f"'did not raise', it is blind to correctness, and it is dominated by solver noise — so a change that "
        f"raises it is about as likely to be harmful as helpful. Use it for ONE purpose: confirming you did not "
        f"introduce a crash. If you want to know whether your change did anything, run the check on the "
        f"UNCHANGED base a few times first and look at the spread; a move smaller than that spread is not a "
        f"result. Leave the final harness at EXACTLY `{cand_path}`."
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return cand_path.exists()


# Three SEARCH STANCES (not technique recipes) — each branch explores the harness space from a different
# angle, so the batch covers more ground. They say HOW to search, never WHICH mechanism to write.
DIVERSITY = [
    (
        "CONSERVATIVE ROLE: preserve every parent behavior that has sound trace evidence. Make the smallest "
        "general change that fixes a concrete failure you can actually see in the traces. Do not over-generalize "
        "one problem's fix into a universal rule."
    ),
    (
        "EXPLORATION ROLE: test a genuinely different hypothesis about WHY the harness fails than the "
        "conservative branch would. Implement the simplest general mechanism your hypothesis implies; do not "
        "copy another branch's mechanism merely because it is the current default."
    ),
    (
        "ADVERSARIAL ROLE: start from the disagreements among the existing candidates and the problems where "
        "they fail. Scrutinize what each harness ACTUALLY does in the traces before trusting it; never assume a "
        "harness is correct just because its solutions run without error. Keep only what the evidence supports."
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
    INVESTIGATE which harness is genuinely correct on the most problems. The worked example printed in a
    problem statement is a HINT for reading the spec, NEVER an oracle: the hidden test re-runs the solution on
    DIFFERENT inputs, and an illustrative table can disagree with the graded behaviour (measured: six
    candidates lost pid 75 by adding `reset_index(drop=True)` to reproduce a printed example)."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    # The harness currently in force. optimize.py builds the pool H-first, but relying on list order for
    # something the prompt asserts as fact is too fragile — it is passed explicitly.
    incumbent = tk.get("incumbent") or cands[0]
    cand_list = "\n".join(f"  - `{c}`  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)" for c in cands)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate data-science-coding harnesses (DS-1000); each ran "
        f"on a BATCH of problems with its FULL trace saved. You do NOT write/change harnesses — ONLY pick the "
        f"SINGLE best. You CANNOT see the hidden gold test.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each `{trace_dir}/<harness>__q<j>.md` has the PROBLEM, every coder call (+ thinking on/off), the final "
        f"code, the SELF-CHECK (ran? error? output?), and a BACK-TRANSLATION.\n\n"
        f"THE TRAP YOU MUST AVOID: a solution RUNNING is not a solution being RIGHT. `ran=True` only means the "
        f"snippet executed; nothing label-free ever checks that `result` holds the right value. Measured on this "
        f"slice, ~61% of the solutions self-check let through are WRONG on the hidden test. So do NOT rank by "
        f"'how many ran clean' — that metric is close to uninformative here. Running is a NECESSARY filter, not "
        f"proof.\n\n"
        f"YOUR JOB IS TO INVESTIGATE, not to tally. On the problems where candidates DIFFER, work out which one "
        f"is actually right, using any label-free evidence you can generate yourself. You have a shell (Bash) and "
        f"the bridge; you are ENCOURAGED to build your own checks. Things you can look at or construct:\n"
        f"  - the recorded SELF-CHECK (ran / error / output / redefines) and BACK-TRANSLATION in each trace;\n"
        f"  - the CANDIDATE SOURCE at `{AGENTS_DIR}/<name>.py` and the solution code in the trace — read what it "
        f"actually computes and compare it clause by clause to the problem;\n"
        f"  - DIFFERENTIAL TESTING — your strongest instrument. The problem's `<code>` block DEFINES the input "
        f"variables, so you can run SEVERAL candidates' recorded solutions for the SAME problem on the SAME "
        f"input and diff their `result`. Where they disagree, at least one is WRONG, and that is where all your "
        f"effort belongs. Where they all agree, the problem tells you NOTHING about which harness is better.\n"
        f"  - INDEPENDENT RE-DERIVATION — for a disagreement, implement the problem's spec YOURSELF from the "
        f"prose, without looking at any candidate's code, then see which side your implementation lands on. "
        f"The statement's worked example is a HINT to check your reading against, NEVER the standard of "
        f"correctness: it may be illustrative, inconsistently formatted, or produced by a different rule than "
        f"the grader uses. A candidate that reproduces the printed example EXACTLY may simply have overfitted "
        f"to it.\n"
        f"  - `checkable=False` means NO probe could be built (the prompt supplies no concrete example input). "
        f"That is ABSENCE OF EVIDENCE — it is NOT a failure and NOT an unsolvable problem. You MUST still "
        f"adjudicate these: construct your own inputs from the problem prose and run the recorded solutions on "
        f"them. In the last batch the judge silently dropped every `checkable=False` problem, called the batch "
        f"'7 solvable', and thereby deleted the very problems that separated the top candidates. Unprobeable "
        f"problems are the LAST thing you may set aside, not the first.\n"
        f"Do NOT re-run `solve()` to score candidates: generation is non-deterministic, so a re-run measures a "
        f"different sample, not the candidate you are judging. Judge the RECORDED solutions (re-executing those "
        f"is fine and encouraged). NEVER call b.is_correct or touch the gold test.\n\n"
        f"CAVEATS (do not be naive): DS-1000 is an INSERTION task — the hidden test re-runs the solution on "
        f"DIFFERENT inputs than the ones shown, so a solution that supplies its own data instead of using the "
        f"given variables can look perfect here and still fail. The coder's errors are also CORRELATED — several "
        f"candidates can be identically wrong, so agreement is a weak prior, never a majority vote; the odd one "
        f"out is a SUSPECT, not automatically the loser. If a candidate crashes on an input YOU or the "
        f"self-check invented, suspect the INPUT before the program — the graded test supplies well-formed data.\n\n"
        f"HOW TO WEIGH WHAT YOU FIND. Weight problems by DISCRIMINATION, not by count. A problem where exactly "
        f"one candidate is right is worth more than five problems where everyone agrees. If you verify that one "
        f"candidate alone solves a hard problem, that is strong evidence FOR it — do NOT let a tally over easy, "
        f"undiscriminating problems reverse a finding you actually confirmed. (This is the exact error made last "
        f"batch: the judge established that one candidate uniquely had the right algorithm on the hardest "
        f"problem, then demoted it on a 5-of-7 vs 6-of-7 count and picked a harness that was joint-worst.)\n"
        f"NEVER break a tie on ARCHITECTURE. 'more robust regex', 'better system prompt', 'more careful error "
        f"handling', 'looks more general' describe SOURCE CODE, not OUTPUTS — that is how a judge talks itself "
        f"into the wrong answer. If candidates are genuinely tied on measured results, go find another problem "
        f"to discriminate on.\n"
        f"THE INCUMBENT. `{incumbent}` is the harness currently in force — every other candidate is a CHALLENGER "
        f"descended from it. The burden of proof is on the challenger: keep `{incumbent}` unless you can point "
        f"to specific problems where a challenger is verifiably right and it is wrong. A tie goes to the "
        f"incumbent. Evolution that cannot demonstrate an improvement should not be adopted.\n\n"
        f"BEFORE you choose, write `{choice_path}.matrix`: one ROW per candidate, one COLUMN per problem in the "
        f"batch — ALL of them, none dropped — each cell your own verdict RIGHT / WRONG / UNRESOLVED plus the "
        f"check you ran to decide it. A cell may be UNRESOLVED; a cell may NOT be missing. Then reconcile: if "
        f"the harness you are about to name is not the argmax of your own RIGHT counts, either your matrix or "
        f"your conclusion is wrong — say explicitly which, and why, before proceeding.\n"
        f"Then write ONLY the chosen harness's exact NAME to `{choice_path}` "
        f"(run:  echo <name> > {choice_path} ). Then STOP.")
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(tk["run_dir"]) / "claude_sessions"), name=f"judge_{tk['tag']}",
                       timeout_seconds=tk["timeout"], progress=False)


def pick_batch(candidates, trace_dir, run_dir, tag, model, timeout, incumbent=None):
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    choice_path = Path(run_dir) / f"judge_{tag}.txt"
    choice_path.unlink(missing_ok=True)
    tpath = Path(run_dir) / f"judgetask_{tag}.json"
    tpath.write_text(json.dumps({"candidates": candidates, "trace_dir": str(trace_dir),
                                 "choice_path": str(choice_path), "run_dir": str(run_dir), "tag": tag,
                                 "model": model, "timeout": timeout,
                                 "incumbent": incumbent or candidates[0]}))
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
