"""Debug code extraction."""
import re, sys

trace_path = "/data/ygzhang/nj/TTHE_mono/livecodebench/logs/lcb_hardbatch2_g3r3/traces/b0/cand_lcb_hardbatch2_g3r3_b0r2_g2__q2.md"
with open(trace_path) as f:
    content = f.read()

idx = content.find("## FINAL CODE")
if idx >= 0:
    after = content[idx:]
    m = re.search(r"```python\s*\n", after)
    if m:
        code_start = m.end()
        end_m = re.search(r"\n```", after[code_start:])
        if end_m:
            code = after[code_start:code_start+end_m.start()]
            print(f"Code extracted, length={len(code)}")
            print("First 100 chars:", repr(code[:100]))
            print("Last 100 chars:", repr(code[-100:]))
        else:
            print("No closing backtick found")
            print("After FINAL CODE, first 2000 chars:")
            print(repr(after[:2000]))
    else:
        print("No python code block found")
        print(repr(after[:500]))
else:
    print("No FINAL CODE section found")
