"""Verify public-test results for each candidate by re-running final code from traces."""
import re, sys, os
sys.path.insert(0, "/data/ygzhang/nj/TTHE_mono")
from livecodebench import lcb_bridge as b

TRACE_DIR = "/data/ygzhang/nj/TTHE_mono/livecodebench/logs/lcb_hardbatch2_g3r3/traces/b0"
CANDIDATES = ["cand_lcb_hardbatch2_g3r3_b0r2_g0",
              "cand_lcb_hardbatch2_g3r3_b0r2_g1",
              "cand_lcb_hardbatch2_g3r3_b0r2_g2"]

# Problem mapping from trace headers
QID_MAP = {
    "q0": "abc398_g", "q1": "abc397_d", "q2": "abc389_f",
    "q3": "arc191_c", "q4": "arc194_b", "q5": "abc393_e",
    "q6": "arc195_c", "q7": "arc191_d", "q8": "arc195_d", "q9": "arc196_d"
}

# Load problems
probs = {p.qid: p for p in b.load_problems("test6", stdin_only=True)}

def extract_final_code(trace_path):
    """Extract the final code block from a trace markdown file."""
    with open(trace_path) as f:
        content = f.read()

    # Find the FINAL CODE section
    m = re.search(r'^## FINAL CODE\s*\n+```python\s*\n(.*?)\n```', content, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None

def run_and_report(harness_name, trace_qid, problem_qid):
    """Extract code from trace, run against public tests, report."""
    trace_path = os.path.join(TRACE_DIR, f"{harness_name}__{trace_qid}.md")
    if not os.path.exists(trace_path):
        return "NO-TRACE"

    code = extract_final_code(trace_path)
    if not code:
        return "NO-CODE"

    p = probs.get(problem_qid)
    if not p:
        return f"NO-PROB({problem_qid})"

    tests = p.public_tests
    if not tests:
        return "NO-TESTS"

    result = b.run_code(code, tests)
    n_pass = result["n_pass"]
    n_total = result["n_total"]

    # Also show individual test results
    details = []
    for i, r in enumerate(result["results"]):
        status = "PASS" if r["ok"] else "FAIL"
        details.append(f"{status}")

    return f"{n_pass}/{n_total} [{' '.join(details)}]"

# Run verification
print("=" * 80)
print(f"{'Problem':<10} {'g0':<25} {'g1':<25} {'g2':<25}")
print("=" * 80)

for tq in [f"q{i}" for i in range(10)]:
    pq = QID_MAP[tq]
    g0_r = run_and_report(CANDIDATES[0], tq, pq)
    g1_r = run_and_report(CANDIDATES[1], tq, pq)
    g2_r = run_and_report(CANDIDATES[2], tq, pq)
    print(f"{pq:<10} {g0_r:<25} {g1_r:<25} {g2_r:<25}")

print("=" * 80)
print("\nSummary:")
for cand in CANDIDATES:
    full = 0
    partial = 0
    for tq in [f"q{i}" for i in range(10)]:
        pq = QID_MAP[tq]
        r = run_and_report(cand, tq, pq)
        m = re.match(r'^(\d+)/(\d+)', r)
        if m:
            passed = int(m.group(1))
            total = int(m.group(2))
            if passed > 0 and passed < total:
                partial += 1
            elif passed == total:
                full += 1
    print(f"  {cand}: {full} fully passing, {partial} partial")
