"""Check all traces for truncated FINAL CODE."""
import re, os, sys

TRACE_DIR = "/data/ygzhang/nj/TTHE_mono/livecodebench/logs/lcb_hardbatch2_g3r3/traces/b0"
CANDIDATES = ["cand_lcb_hardbatch2_g3r3_b0r2_g0",
              "cand_lcb_hardbatch2_g3r3_b0r2_g1",
              "cand_lcb_hardbatch2_g3r3_b0r2_g2"]

def check_code_completeness(code):
    """Check if code looks complete."""
    if not code or len(code) < 50:
        return "TOO_SHORT"
    # Check if the last non-empty, non-comment line ends mid-statement
    lines = code.split('\n')
    # Find last non-empty line
    last_content = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            last_content = stripped
            break
    # If the last line is a partial line or ends with truncated operator
    truncated_markers = ['# We ', '# self', '# The ', '# This ', 'self.', 'We ', '#\n']
    for marker in truncated_markers:
        if code.rstrip().endswith(marker):
            return f"TRUNCATED (ends with '{marker}')"
    # Check if the code ends with an unfinished statement
    # Common patterns: last content line doesn't end properly
    if last_content:
        incomplete_endings = ['#', 'self', 'def ', 'class ']
        for e in incomplete_endings:
            if code.rstrip().endswith(e):
                return f"TRUNCATED (ends with '{e}')"
    return "OK"

for cand in CANDIDATES:
    for qi in range(10):
        path = os.path.join(TRACE_DIR, f"{cand}__q{qi}.md")
        if not os.path.exists(path):
            print(f"{cand} q{qi}: NO FILE")
            continue
        with open(path) as f:
            content = f.read()

        idx = content.find("## FINAL CODE")
        if idx < 0:
            print(f"{cand} q{qi}: NO FINAL CODE SECTION")
            continue

        after = content[idx:]
        m = re.search(r'```python\s*\n', after)
        if m:
            end_m = re.search(r'\n```', after[m.end():])
            if end_m:
                code = after[m.end():m.end()+end_m.start()]
                status = check_code_completeness(code)
                last_60 = code.rstrip()[-60:] if len(code) > 60 else code.rstrip()
                print(f"{cand} q{qi}: {status} (len={len(code)})")
                if status != "OK":
                    print(f"  Last 60 chars: {repr(last_60)}")
            else:
                print(f"{cand} q{qi}: NO CLOSING BACKTICK")
        else:
            print(f"{cand} q{qi}: NO CODE BLOCK")
