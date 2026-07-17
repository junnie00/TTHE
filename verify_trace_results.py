"""Read PUBLIC-TEST RESULTS from traces (these are ground truth)."""
import re, os

TRACE_DIR = "/data/ygzhang/nj/TTHE_mono/livecodebench/logs/lcb_hardbatch2_g3r3/traces/b0"
CANDIDATES = ["cand_lcb_hardbatch2_g3r3_b0r2_g0",
              "cand_lcb_hardbatch2_g3r3_b0r2_g1",
              "cand_lcb_hardbatch2_g3r3_b0r2_g2"]

QID_MAP = {
    "q0": "abc398_g", "q1": "abc397_d", "q2": "abc389_f",
    "q3": "arc191_c", "q4": "arc194_b", "q5": "abc393_e",
    "q6": "arc195_c", "q7": "arc191_d", "q8": "arc195_d", "q9": "arc196_d"
}

for cand in CANDIDATES:
    print(f"\n{'='*70}")
    print(f"  {cand}")
    print(f"{'='*70}")
    full = 0
    partial = 0
    zero = 0
    total_tests_passed = 0
    total_tests_total = 0

    for qi in range(10):
        tq = f"q{qi}"
        pq = QID_MAP[tq]
        path = os.path.join(TRACE_DIR, f"{cand}__{tq}.md")
        if not os.path.exists(path):
            print(f"  {pq:10s} NO TRACE")
            continue

        with open(path) as f:
            content = f.read()

        # Find the PUBLIC-TEST RESULTS line
        m = re.search(r'## PUBLIC-TEST RESULTS \((\d+)/(\d+) passed\)', content)
        if m:
            n_pass = int(m.group(1))
            n_total = int(m.group(2))
            total_tests_passed += n_pass
            total_tests_total += n_total

            # Get all individual test results
            test_results = []
            for line in content[m.end():].split('\n'):
                line = line.strip()
                tm = re.match(r'(test\d+):\s+(PASS|FAIL)', line)
                if tm:
                    test_results.append(tm.group(2))

            result_str = f"{n_pass}/{n_total}"
            if n_pass == n_total and n_total > 0:
                result_str += " ★ ALL PASS"
                full += 1
            elif n_pass == 0:
                result_str += " ✗ ALL FAIL"
                zero += 1
            else:
                result_str += f" partial"
                partial += 1

            tests_str = ' '.join(test_results) if test_results else '(no detail)'
            print(f"  {pq:10s} {result_str:30s} [{tests_str}]")
        else:
            # Check for other patterns
            no_code = 'NO-CODE' in content or 'no code' in content.lower()
            print(f"  {pq:10s} NO PUBLIC TEST RESULTS SECTION" + (" (no code)" if no_code else ""))

    print(f"  {'─'*60}")
    print(f"  Summary: {full} fully passing, {partial} partial, {zero} zero")
    print(f"  Total individual tests: {total_tests_passed}/{total_tests_total}")
