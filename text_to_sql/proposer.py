"""The PROPOSER (policy): generates candidate harness edits. The SAME frozen weak model, given tools.

Two backends:
  - agentic  : Claude-Code-on-flash, ONE candidate per isolated SUBPROCESS + process group, with a HARD
               external deadline that killpg's any stuck worker (only OUR procs — never the user's claude
               sessions). This is the hang-guard: one stuck Claude Code call can't freeze the run.
  - single   : flash sampled G times at temperature (fast, lower quality).

Run as a module for the isolated worker:  python -m ...proposer --worker <task.json>
"""
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import bridge
from . import claude_wrapper
from .evolve import load_harness, PKG, PKG_DIR, AGENTS_DIR, MH_ROOT

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
PROPOSE_SYS = ("You are a Python+SQL engineer writing Text-to-SQL HARNESSES (a class wrapping a FROZEN weak "
               "SQL model). Output complete runnable code only.")


def desc(res):
    if not res.get("ok"):
        return "ERROR"
    if not res["rows"]:
        return "EMPTY"
    return ("; ".join(str(r) for r in res["rows"][:3]))[:160]


# ── agentic backend ─────────────────────────────────────────────────────────
def propose_agentic(question, sql0, res_desc, cur_name, new_name, run_dir, t, gi, model, timeout, db_id,
                    diversity=""):
    """ONE agentic proposer turn — writes an improved harness to cand_path (GRPO selection replaces the
    proposer's keep/stop judgment, so it ALWAYS writes one). Runs synchronously inside a worker subprocess."""
    cand_path = AGENTS_DIR / f"{new_name}.py"
    obs_path = Path(run_dir) / f"obs_t{t}_g{gi}.md"
    q_path = Path(run_dir) / f"q_t{t}_g{gi}.txt"
    check_path = Path(run_dir) / f"check_t{t}_g{gi}.py"
    cand_path.unlink(missing_ok=True)
    q_path.write_text(question, encoding="utf-8")
    obs_path.write_text(f"# Coder's attempt — NO gold; improve the harness from the executed result only.\n\n"
                        f"Question: {question}\n\nThe coder's SQL:\n{sql0.strip()[:400]}\n\n"
                        f"Executing it on the database returned:\n{res_desc}\n")
    # self-check: lets the proposer RUN its own harness on this question and SEE the result (its verifier)
    check_path.write_text(
        "import traceback\n"
        f"from {PKG} import bridge\n"
        f"from {PKG}.evolve import load_harness\n"
        f"q = open({json.dumps(str(q_path))}, encoding='utf-8').read()\n"
        f"db = bridge.get_db({json.dumps(db_id)})\n"
        "try:\n"
        f"    h = load_harness({json.dumps(new_name)}, db)\n"
        "    sql = h.solve(q); res = bridge.execute(db, sql)\n"
        "    print('SQL:', ' '.join(str(sql).split())[:300])\n"
        "    if not res.get('ok'): print('STATUS: EXEC-ERROR ::', str(res.get('error'))[:200])\n"
        "    elif not res['rows']: print('STATUS: EMPTY (filter likely matched nothing - wrong value/join)')\n"
        "    else:\n"
        "        ncol = len(res['rows'][0]) if isinstance(res['rows'][0], (list, tuple)) else 1\n"
        "        print('STATUS: OK | rows=%d | columns_per_row=%d | sample=%s' % (len(res['rows']), ncol, str(res['rows'][:3])[:220]))\n"
        "except Exception:\n"
        "    print('STATUS: HARNESS-CRASH'); traceback.print_exc()\n")
    prompt = (
        "A coder (a FROZEN weak SQL model wrapped by a harness YOU control) just answered a question. You "
        "CANNOT see the gold answer. GOAL: improve the coder's HARNESS so it RELIABLY answers this (and "
        "similar) questions — label-free.\n\n"
        f"READ: `{obs_path}` (question + coder's SQL + executed result); `{AGENTS_DIR}/{cur_name}.py` (CURRENT "
        f"harness — copy + improve); `{PKG_DIR}/harness_base.py` (the SQLHarness interface).\n\n"
        "Harness API (NOT sqlite3): self.tables()->{table:[cols]}; self.distinct(table,col,limit=50)->[values]; "
        "self.column_types(table); self.execute(sql)->{'ok','rows'}; self.llm(prompt,system='',temperature=0.0,"
        "n=1); self.schema (string); bridge.extract_sql(text). GENERAL — never hardcode this question's "
        "value/column/answer.\n\n"
        f"Probe the DB freely with Bash, e.g.:\n  PYTHONPATH={MH_ROOT} python -c \"from {PKG} import bridge; "
        f"db=bridge.get_db('{db_id}'); print(bridge.execute(db, 'SELECT ...'))\"\n\n"
        "The frozen model SILENTLY ERRS (wrong table/column, wrong-CASE value, wrong aggregation) yet returns a "
        "plausible result. YOU decide how to make the harness robust — e.g. value-grounding via self.distinct, "
        "decomposing the question, re-prompting/self-debug on error, multi-sample voting — your choice, whatever "
        "the EVIDENCE says works. Constraints: keep it GENERAL (no hardcoding) and FOCUSED; do NOT write a custom "
        "SQL parser/tokenizer or heavy/backtracking regex over the SQL text (it HANGS — catastrophic backtracking "
        "holds the GIL; inspect SQL by EXECUTING it, not by parsing its text).\n\n"
        "VERIFY BY RE-RUNNING — do not guess, and do not finish on a broken result:\n"
        f"  1. Edit `{cand_path}` (start from the current harness).\n"
        f"  2. Run your harness on THIS question:  PYTHONPATH={MH_ROOT} python {check_path}\n"
        "     It prints STATUS: OK / EMPTY / EXEC-ERROR / HARNESS-CRASH, plus the SQL, the rows, and "
        "columns_per_row.\n"
        "  3. If STATUS is not OK, diagnose from the output + DB, fix, and re-run. Iterate.\n"
        "  4. EVEN ON STATUS: OK, verify the result actually ANSWERS the question — a query that runs and "
        "returns a plausible result can still be the wrong answer. Re-read the question and the Hint literally "
        "and make the output match them EXACTLY: the right TYPE of answer, the exact attribute(s)/quantity the "
        "question asks for and nothing extra, and every condition in the question genuinely applied. The Hint is "
        "AUTHORITATIVE — it states exactly how to interpret the question (which columns, filters, thresholds, "
        "comparisons): follow it EXACTLY, do NOT override, 'correct', or second-guess it even if you think a "
        "different formulation is cleverer or catches a subtlety. PROBE THE DB to ground your reasoning (e.g. "
        "SELECT DISTINCT to see real stored values your filters must match, or check what a column actually "
        "contains) rather than assuming. Return exactly what is asked — no more, no less. If it does not match "
        "the question and Hint, the SQL is wrong even though it ran; fix and re-run.\n"
        "  5. ONLY stop when re-running shows STATUS: OK AND the output shape matches what the question asks. "
        "Do NOT stop on EMPTY / EXEC-ERROR / HARNESS-CRASH or with extra/missing columns.\n\n"
        "MINIMAL, LITERAL changes — do NOT over-engineer or INVENT requirements the question never stated. "
        "Answer EXACTLY what is asked, no more: do NOT add LIMIT, MAX/MIN, GROUP BY, extra filters or "
        "aggregations the question did not explicitly request (e.g. 'list the rulings ... in descending date "
        "order' means ALL of them ordered by date — NOT only the latest). If the result is correct except "
        "for an extra column or wrong case, fix ONLY that (drop the column / fix the case) and leave the rest "
        "of the working query unchanged — do not restructure logic that already works.\n\n"
        f"Keep `from ..harness_base import SQLHarness` and `from .. import bridge`. Leave the final verified "
        f"harness at EXACTLY `{cand_path}`."
        + (f"\n\nDIVERSITY (this is one of several parallel attempts): {diversity}" if diversity else "")
    )
    claude_wrapper.run(prompt=prompt, model=model, allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"t{t}g{gi}",
                       timeout_seconds=timeout, progress=False)
    return cand_path.exists()


DIVERSITY = [
    (
        "CONSERVATIVE REPAIR ROLE: preserve every parent behavior that has sound trace evidence. "
        "Make the smallest general change that fixes a concrete mismatch or runtime failure. Do not "
        "turn one batch-specific interpretation into a universal output, aggregation, tie, address, "
        "name-format, or deduplication rule."
    ),
    (
        "INDEPENDENT EXPLORATION ROLE: test a genuinely different causal hypothesis from the conservative "
        "proposal. Enumerate plausible tables, join paths, aggregation grains, and output shapes, then use "
        "read-only DB probes to falsify them. Implement the simplest surviving general mechanism; do not "
        "copy the other proposal's mechanism merely because it is the current default."
    ),
    (
        "ADVERSARIAL AUDIT AND SYNTHESIS ROLE: start from disagreements among the existing candidates. "
        "Compare Question, authoritative Hint, back-translation, final rows, aggregation grain, and output "
        "shape clause by clause. Use DB counterexamples to falsify unsupported mechanisms, then synthesize "
        "only independently supported strengths. Preserve runtime robustness, but never equate executable "
        "or non-empty SQL with semantic correctness."
    ),
]


def candidate_diversity(gi):
    """Return one fixed proposer role; generation round never changes it."""
    return DIVERSITY[gi % len(DIVERSITY)]


def proposal_card_path(run_dir, candidate):
    return Path(run_dir) / "proposal_cards" / f"{candidate}.json"


def valid_proposal_card(
    path,
    candidate,
    base_candidate=None,
    branch_id=None,
    generation_round=None,
    peer_candidates=None,
):
    try:
        card = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        card.get("candidate") == candidate
        and isinstance(card.get("branch_id"), int)
        and isinstance(card.get("generation_round"), str)
        and isinstance(card.get("base_candidate"), str)
        and (
            base_candidate is None
            or card.get("base_candidate") == base_candidate
        )
        and (
            branch_id is None
            or card.get("branch_id") == branch_id
        )
        and (
            generation_round is None
            or card.get("generation_round") == generation_round
        )
        and isinstance(card.get("peer_candidates"), list)
        and (
            peer_candidates is None
            or sorted(card.get("peer_candidates")) == sorted(peer_candidates)
        )
        and isinstance(card.get("role"), str)
        and isinstance(card.get("behavior_changes"), list)
        and isinstance(card.get("preserved_behaviors"), list)
        and isinstance(card.get("verification"), list)
        and isinstance(card.get("risks"), list)
    )


def sample_agentic(question, sql0, res_desc, cur_name, t, g, run_dir, model, timeout, db_id):
    """Sample G DIVERSE candidates = G isolated worker subprocesses, hard-killed past timeout+120s."""
    names = [f"cand_t{t}_g{gi}" for gi in range(g)]
    procs = []
    for gi in range(g):
        (AGENTS_DIR / f"{names[gi]}.py").unlink(missing_ok=True)
        task = {"question": question, "sql0": sql0, "res_desc": res_desc, "cur_name": cur_name,
                "new_name": names[gi], "run_dir": str(run_dir), "t": t, "gi": gi,
                "model": model, "timeout": timeout, "db_id": db_id,
                "diversity": DIVERSITY[gi % len(DIVERSITY)]}
        tpath = Path(run_dir) / f"task_t{t}_g{gi}.json"
        tpath.write_text(json.dumps(task))
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
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
    return [names[gi] for gi in range(g) if (AGENTS_DIR / f"{names[gi]}.py").exists()]


def _choice_path(run_dir, t, r):
    return Path(run_dir) / f"choice_t{t}r{r}.txt"


def run_selector(tk):
    """ONE agentic session: read the N candidate (SQL, result) pairs, pick the index whose result best answers
    the question (may probe the DB), and write that index to the choice file. This is the GROUP-RELATIVE
    selection — the proposer judging its own children from execution evidence (no fixed reward formula)."""
    cands, db_id = tk["cands"], tk["db_id"]
    choice_path = _choice_path(tk["run_dir"], tk["t"], tk["r"])
    lines = [f"QUESTION: {tk['question']}\n",
             f"Below are {len(cands)} candidate answers (each = a SQL and its executed result). They may take "
             f"DIFFERENT interpretations of the question. Pick the SINGLE index whose result MOST CORRECTLY "
             f"answers the question. Judge from EVIDENCE: re-read the question literally and check the result "
             f"answers EXACTLY what is asked — the right type of answer and the exact attribute(s)/quantity "
             f"asked, nothing extra. The Hint in the question is AUTHORITATIVE — it states exactly how to "
             f"interpret the question (which columns, filters, thresholds, comparisons to use). A candidate that "
             f"FOLLOWS the Hint beats one that overrides, 'corrects', or second-guesses it — even if the "
             f"override looks cleverer or catches a subtlety. PROBE THE DB to decide between interpretations "
             f"(run SELECT ... to see what is really there). Do NOT prefer an answer just because more candidates returned it — several "
             f"candidates can share the SAME mistake, and the lone different one can be the correct fix; decide "
             f"from the actual data and the question, never from how many candidates agree.\n"
             f"  PYTHONPATH={MH_ROOT} python -c \"from {PKG} import bridge; db=bridge.get_db('{db_id}'); "
             f"print(bridge.execute(db,'SELECT ...'))\""]
    for i, c in enumerate(cands):
        btline = f"\n    back-translation mismatch: {c['bt']}" if c.get("bt") is not None else ""
        lines.append(f"\n[{i}] SQL: {str(c['sql'])[:320]}\n    RESULT: {c['res']}{btline}")
    has_bt = any(c.get("bt") is not None for c in cands)
    bt_note = ("\n\nThe 'back-translation mismatch' score (0.0..1.0) re-describes each SQL in English and checks "
               "it against the question: LOWER means the SQL's literal behaviour matches the question better. "
               "Treat it as ONE clue alongside the executed RESULT and your own DB probing — it is NOT always "
               "reliable, so weigh it, do not follow it blindly." if has_bt else "")
    prompt = ("\n".join(lines) + bt_note + f"\n\nDecide, then write ONLY the chosen integer index "
              f"(0..{len(cands)-1}) to `{choice_path}` (run:  echo <i> > {choice_path} ). Then STOP.")
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(tk["run_dir"]) / "claude_sessions"), name=f"sel_t{tk['t']}r{tk['r']}",
                       timeout_seconds=tk["timeout"], progress=False)


def select_agentic(question, cands, run_dir, t, r, model, timeout, db_id):
    """Run the selector in an isolated, hard-killed subprocess; return the chosen candidate name (or None)."""
    if len(cands) <= 1:
        return cands[0]["name"] if cands else None
    tpath = Path(run_dir) / f"seltask_t{t}r{r}.json"
    tpath.write_text(json.dumps({"question": question, "cands": cands, "run_dir": str(run_dir), "t": t, "r": r,
                                 "model": model, "timeout": timeout, "db_id": db_id}))
    choice_path = _choice_path(run_dir, t, r)
    choice_path.unlink(missing_ok=True)
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.proposer", "--selector", str(tpath)],
                         cwd=str(MH_ROOT), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        pass
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if choice_path.exists():
        m = re.search(r"\d+", choice_path.read_text())
        if m and 0 <= int(m.group()) < len(cands):
            return cands[int(m.group())]["name"]
    return None


# ── BATCH harness evolution: ONE general harness, evolved by deep-reading FULL traces over a batch ─────
def propose_batch(tk):
    """One branch generator: inspect all peer traces, but edit only its assigned base."""
    candidates, trace_dir = tk["candidates"], tk["trace_dir"]
    new_name, gi, tag, run_dir = tk["new_name"], tk["gi"], tk["tag"], tk["run_dir"]
    base_candidate = tk.get("base_candidate", candidates[0])
    if base_candidate not in candidates:
        raise ValueError(f"branch base {base_candidate!r} is not in active candidates")
    cand_path = AGENTS_DIR / f"{new_name}.py"
    base_path = AGENTS_DIR / f"{base_candidate}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    card_path = proposal_card_path(run_dir, new_name)
    expected_peer_candidates = [
        candidate for candidate in candidates if candidate != base_candidate
    ]
    card_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.unlink(missing_ok=True)
    card_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, cand_path)
    cand_rows = []
    for candidate in candidates:
        prior_card = proposal_card_path(run_dir, candidate)
        card_note = f" ; prior proposal card: {prior_card}" if prior_card.exists() else ""
        relation = "ASSIGNED BASE" if candidate == base_candidate else "PEER EVIDENCE"
        cand_rows.append(
            f"  - `{candidate}` [{relation}] (full traces: {trace_dir}/{candidate}__q*.md ; "
            f"code: {AGENTS_DIR}/{candidate}.py{card_note})"
        )
    cand_list = "\n".join(cand_rows)
    prompt = (
        f"You are evolving a GENERAL Text-to-SQL HARNESS (arbitrary Python wrapping a FROZEN weak coder). There "
        f"are {len(candidates)} candidate harnesses; each was run on a BATCH of questions and its FULL execution "
        f"trace was saved. You CANNOT see gold answers — label-free.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"BRANCH LINEAGE — this is mandatory:\n"
        f"- You are branch G{gi}. Your assigned base is `{base_candidate}`.\n"
        f"- `{cand_path}` has already been copied byte-for-byte from that base. Edit this copy in place.\n"
        f"- Other candidates and traces are peer evidence: learn from and test their mechanisms, but do not "
        f"switch your parent or replace the target wholesale with a peer harness.\n"
        f"- The output must remain a descendant of `{base_candidate}`. If evidence is insufficient, make a "
        f"minimal or no-op refactor rather than adopting another branch.\n\n"
        f"DEEP-READ THE FULL TRACES (this is the most important step). Each trace file "
        f"`{trace_dir}/<harness>__q<j>.md` contains, for ONE harness on ONE question: the QUESTION, the HINT "
        f"(AUTHORITATIVE), the DATABASE SCHEMA, EVERY step the harness took (each coder-model call's "
        f"prompt+response, each SQL it ran + that SQL's result), the FINAL SQL+result, and a BACK-TRANSLATION "
        f"(the final SQL described literally in English). USE THE BACK-TRANSLATION: compare it to the question "
        f"+ Hint — if the English says something different from what the question asks, that SQL is wrong.\n\n"
        f"You may READ the peer source code to understand behavioral differences, but your implementation base "
        f"remains `{base_candidate}`. You may port a small general mechanism from a peer only when its trace and "
        f"DB evidence support it, and the proposal card must identify that evidence. IMPROVE your branch so it "
        f"answers MORE of the batch "
        f"correctly. Target the FAILING questions you saw in the traces with GENERAL mechanisms (value-grounding "
        f"via self.distinct, decomposition, self-debug/re-prompt on error, making the coder FOLLOW the Hint, ...). "
        f"Harness API (NOT sqlite3): self.tables(); self.distinct(table,col); self.column_types(table); "
        f"self.execute(sql); self.llm(prompt,system='',temperature=0.0,n=1); self.schema; bridge.extract_sql. "
        f"GENERAL — never hardcode a question's value/column/answer; the Hint is authoritative. Do NOT write a "
        f"custom SQL parser/regex (it HANGS the GIL). Probe the DB freely: PYTHONPATH={MH_ROOT} python -c "
        f"\"from {PKG} import bridge; db=bridge.get_db('<db_id>'); print(bridge.execute(db,'SELECT ...'))\".\n\n"
        f"OUTPUT ALIGNMENT — a VERY FREQUENT silent failure where the data is right but the SHAPE is wrong, so "
        f"it scores as WRONG: the result must contain EXACTLY the column(s) the question asks for and NOTHING "
        f"else. If it asks 'which superhero' return ONLY the name (NOT name+score); if it asks for a percentage "
        f"return ONLY that number. Do NOT add 'helpful' extra columns (the metric you sorted by, an id, a label). "
        f"Do NOT round/truncate numbers unless the question or Hint explicitly says to — return FULL precision "
        f"(avoid ROUND()). Return only as many rows as asked (e.g. a single 'the most ...' wants ONE row, not all "
        f"ties). Make the harness force the coder's SELECT list + row count to match EXACTLY what is asked.\n\n"
        f"EXECUTION RESPONSIBILITY — STRICT:\n"
        f"- The fixed controller, not you, owns all harness evaluation. After you exit, it will run the child "
        f"exactly once on each batch question, save the official traces, and expose those traces in the next "
        f"generation round. That controller execution is the R1/R2/R3 loop.\n"
        f"- NEVER execute, import, instantiate, or load any parent, peer, or child harness. NEVER run a "
        f"`check_*.py`, `solve_worker`, evaluator, optimizer, or batch script. NEVER create a checker or repeat "
        f"a batch/question to test the child.\n"
        f"- Bash is available only for READ-ONLY database investigation that helps interpret an EXISTING trace, "
        f"using `bridge.get_db(...)` plus `bridge.execute(...)` with SELECT/CTE/PRAGMA queries. You may inspect "
        f"files, but must not use Bash/Python to call a harness or model.\n"
        f"- Base every change on existing parent/peer traces and optional read-only DB probes. The child is "
        f"unexecuted, so do not claim that it ran successfully, improved a runtime count, or achieved a score.\n"
        f"Edit only `{cand_path}` and keep `from ..harness_base import SQLHarness` and `from .. import bridge`.\n\n"
        f"Write a proposal card to `{card_path}` as one JSON object with EXACTLY these keys:\n"
        f"- `candidate`: exactly `{new_name}`\n"
        f"- `branch_id`: integer `{gi}`\n"
        f"- `generation_round`: exactly `{tag}`\n"
        f"- `base_candidate`: exactly `{base_candidate}`\n"
        f"- `peer_candidates`: exactly this JSON list, with no additions or omissions: "
        f"`{json.dumps(expected_peer_candidates)}`. Concurrent sibling proposals are NOT active candidates in "
        f"this round and must not be listed.\n"
        f"- `role`: the search role/focus you followed\n"
        f"- `behavior_changes`: a list of objects with `trace`, `observed_issue`, `change`, `db_evidence`, "
        f"and `expected_effect`\n"
        f"- `preserved_behaviors`: a list of important behaviors deliberately left unchanged\n"
        f"- `verification`: a list of objects with `trace`, `runtime_status`, and `evidence`; these may describe "
        f"only existing parent/peer traces and read-only DB probes, never an execution of the new child\n"
        f"- `risks`: unresolved ambiguities or plausible regressions\n"
        f"Do not claim an absolute score and do not mention gold. The card is an auditable change report, not "
        f"proof of correctness. Write or edit ONLY `{cand_path}` and `{card_path}`. Leave both files present."
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return (
        cand_path.exists()
        and valid_proposal_card(
            card_path,
            new_name,
            base_candidate,
            gi,
            tag,
            expected_peer_candidates,
        )
    )


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


def sample_batch(candidates, trace_dir, run_dir, tag, run_name, batch, g, model, timeout):
    """G parallel GENERATOR workers — each deep-reads ALL `candidates`' full traces and writes ONE improved
    harness. Returns the new children names."""
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(json.dumps([{"db_id": d, "q": q.question} for d, q in batch]))
    names = [f"cand_{run_name}_{tag}_g{gi}" for gi in range(g)]      # run-name prefix -> never collide across runs
    procs = []
    for gi in range(g):
        (AGENTS_DIR / f"{names[gi]}.py").unlink(missing_ok=True)
        task = {"kind": "generate_batch", "candidates": candidates, "trace_dir": str(trace_dir),
                "batch_json": str(batch_json), "new_name": names[gi], "run_dir": str(run_dir), "tag": tag,
                "gi": gi, "model": model, "timeout": timeout, "diversity": candidate_diversity(gi)}
        tpath = Path(run_dir) / f"task_{tag}_g{gi}.json"
        tpath.write_text(json.dumps(task))
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    _spawn_and_wait(procs, timeout)
    return [
        names[gi]
        for gi in range(g)
        if (
            (AGENTS_DIR / f"{names[gi]}.py").exists()
            and valid_proposal_card(
                proposal_card_path(run_dir, names[gi]),
                names[gi],
                candidates[0],
                gi,
                tag,
                [candidate for candidate in candidates if candidate != candidates[0]],
            )
        )
    ]


def sample_branches(branches, trace_dir, run_dir, tag, run_name, batch, model, timeout,
                    solve_timeout=180):
    """Advance fixed proposer branches once.

    Every branch sees the same active harness--trace pairs, while branch ``gi``
    receives ``branches[gi]`` as its mechanically pre-copied edit base. The
    returned list preserves branch order and uses ``None`` for a failed child.
    """
    active = list(dict.fromkeys(branches))
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(
        json.dumps([{"db_id": d, "q": q.question} for d, q in batch]),
        encoding="utf-8",
    )
    names = [f"cand_{run_name}_{tag}_g{gi}" for gi in range(len(branches))]
    procs = []
    for gi, (base_candidate, name) in enumerate(zip(branches, names)):
        (AGENTS_DIR / f"{name}.py").unlink(missing_ok=True)
        proposal_card_path(run_dir, name).unlink(missing_ok=True)
        task = {
            "kind": "generate_batch",
            "candidates": active,
            "base_candidate": base_candidate,
            "trace_dir": str(trace_dir),
            "batch_json": str(batch_json),
            "new_name": name,
            "run_dir": str(run_dir),
            "tag": tag,
            "gi": gi,
            "model": model,
            "timeout": timeout,
            "solve_timeout": solve_timeout,
            "diversity": candidate_diversity(gi),
        }
        task_path = Path(run_dir) / f"task_{tag}_g{gi}.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "-m", f"{PKG}.proposer", "--worker", str(task_path)],
            cwd=str(MH_ROOT),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ))
    _spawn_and_wait(procs, timeout)

    children = []
    for gi, (base_candidate, name) in enumerate(zip(branches, names)):
        valid = (
            (AGENTS_DIR / f"{name}.py").exists()
            and valid_proposal_card(
                proposal_card_path(run_dir, name),
                name,
                base_candidate,
                gi,
                tag,
                [candidate for candidate in active if candidate != base_candidate],
            )
        )
        children.append(name if valid else None)
    return children


def run_picker(tk):
    """ONE JUDGE session: deep-read the FULL traces of ALL candidates and write the single best harness name."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    cand_rows = []
    for candidate in cands:
        card = proposal_card_path(tk["run_dir"], candidate)
        card_note = str(card) if valid_proposal_card(card, candidate) else "(none: carried parent/baseline)"
        cand_rows.append(
            f"  - `{candidate}`  (full traces: {trace_dir}/{candidate}__q*.md ; "
            f"code: {AGENTS_DIR}/{candidate}.py ; proposal card: {card_note})"
        )
    cand_list = "\n".join(cand_rows)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate GENERAL Text-to-SQL harnesses; each was run on a "
        f"BATCH of questions and its FULL execution trace was saved. You do NOT write or change any harness — "
        f"ONLY pick the SINGLE best one (correct on the MOST batch questions). You CANNOT see gold answers.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each trace `{trace_dir}/<harness>__q<j>.md` has: the QUESTION, the HINT, the SCHEMA, every step the "
        f"harness took, the final SQL+result, and a BACK-TRANSLATION (the SQL in plain English).\n\n"
        f"Each new candidate also has a PROPOSAL CARD describing intended changes, evidence, preservation, and "
        f"risks. Read it to locate behavioral differences, but treat it as an UNTRUSTED SELF-REPORT: verify every "
        f"claim against the corresponding trace and your own DB probe. A persuasive or longer card is not evidence, "
        f"and a carried parent without a card is not disadvantaged.\n\n"
        f"HARD RULES — ignore these and you WILL pick wrong:\n"
        f"1. DO NOT EYEBALL. Never judge a candidate right because its SQL or number 'looks right'. For each "
        f"question you MUST VERIFY by actually RUNNING checks — re-run the SQL and PROBE the DB to confirm — and "
        f"base every verdict ONLY on a check you actually ran. Probe like: PYTHONPATH={MH_ROOT} python -c \"from "
        f"{PKG} import bridge; db=bridge.get_db('<db_id>'); print(bridge.execute(db,'SELECT ...'))\".\n"
        f"2. CONSENSUS IS A WEAK PRIOR, NOT PROOF. Agreement across candidates correlates with the truth on easy "
        f"questions, but on hard ones the model's errors can be correlated and the majority is then confidently "
        f"wrong — so do NOT decide correctness by vote. Judge each candidate independently against the DB and the "
        f"Hint, NOT against what the other candidates produced; use agreement only as a soft tiebreaker.\n"
        f"3. HINT IS GROUND TRUTH. The Hint gives the exact columns / formula / filters. PROBE the DB to check a "
        f"candidate's SQL implements the Hint EXACTLY; a candidate that follows the Hint beats one that deviates, "
        f"even if the deviating number looks plausible.\n"
        f"4. EVIDENCE REQUIRED. On the questions where candidates DIFFER, write the actual DB probe (query + its "
        f"output) that justifies your verdict for each. A verdict with no probe behind it is invalid.\n\n"
        f"Do not announce an absolute N/N score: gold is unavailable. Record uncertainty when Question/Hint and "
        f"observable DB evidence cannot distinguish two output contracts.\n\n"
        f"Go question by question over the questions where candidates differ; for each, determine FROM YOUR OWN "
        f"PROBES which candidate(s) truly satisfy the Hint and return the right answer. Then pick the harness "
        f"correct on the most questions. Write ONLY the chosen harness's exact NAME to `{choice_path}` (run:  "
        f"echo <name> > {choice_path} ). Then STOP.")
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(tk["run_dir"]) / "claude_sessions"), name=f"judge_{tk['tag']}",
                       timeout_seconds=tk["timeout"], progress=False)


def pick_batch(candidates, trace_dir, run_dir, tag, model, timeout):
    """Run the JUDGE in a hard-killed subprocess; return the chosen harness name (or None)."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    choice_path = Path(run_dir) / f"judge_{tag}.txt"
    choice_path.unlink(missing_ok=True)
    tpath = Path(run_dir) / f"judgetask_{tag}.json"
    tpath.write_text(json.dumps({"candidates": candidates, "trace_dir": str(trace_dir),
                                 "choice_path": str(choice_path), "run_dir": str(run_dir), "tag": tag,
                                 "model": model, "timeout": timeout}))
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.proposer", "--picker", str(tpath)],
                         cwd=str(MH_ROOT), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _spawn_and_wait([p], timeout)
    if choice_path.exists():
        nm = choice_path.read_text().strip().split()
        if nm and nm[0] in candidates:
            return nm[0]
    return None


# ── single-shot backend ─────────────────────────────────────────────────────
def sample_single(interface, cur_code, feedback, t, g, model):
    prompt = (
        "Improve the Text-to-SQL HARNESS so the FROZEN weak model answers correctly (query the DB for exact "
        "stored values via self.distinct, pick the right table/column, retry on empty/error; GENERAL, robust, "
        "no per-question hardcoding, prefer real parsing/execution over brittle regex).\n\n"
        f"## Interface\n```python\n{interface}\n```\n\n## Current harness\n```python\n{cur_code}\n```\n\n"
        f"## Execution feedback (label-free)\n{feedback}\n\nKeep `from ..harness_base import SQLHarness` and "
        "`from .. import bridge`. Output the COMPLETE harness as ONE ```python block."
    )
    outs = bridge.solver_llm(prompt, system=PROPOSE_SYS, temperature=0.8, n=g)
    outs = outs if isinstance(outs, list) else [outs]
    names = []
    for gi, o in enumerate(outs):
        m = re.search(r"```(?:python)?[ \t]*\n(.*)\n```", o, re.DOTALL)
        if m:
            name = f"cand_t{t}_g{gi}"
            (AGENTS_DIR / f"{name}.py").write_text(m.group(1).strip())
            names.append(name)
    return names


if __name__ == "__main__":
    # isolated worker: do ONE agentic proposal, then exit
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--picker")
    a = ap.parse_args()
    if a.picker:                                    # batch-evolution JUDGE: deep-read all traces, pick the best
        run_picker(json.load(open(a.picker)))
    else:
        tk = json.load(open(a.worker))
        if tk.get("kind") == "generate_batch":      # batch-evolution generator: deep-read traces, improve the best
            propose_batch(tk)
        else:                                       # legacy single-question proposer (unused by batch loop)
            propose_agentic(tk["question"], tk["sql0"], tk["res_desc"], tk["cur_name"], tk["new_name"],
                            tk["run_dir"], tk["t"], tk["gi"], tk["model"], tk["timeout"], tk["db_id"],
                            diversity=tk.get("diversity", ""))
