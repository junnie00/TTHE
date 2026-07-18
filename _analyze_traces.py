"""Analyze traces across all 3 harnesses."""
import glob, re, os

TRACES = "/data/ygzhang/nj/TTHE_mono/livecodebench/logs/tthe_hard50/traces/b2"

def extract_results(text):
    results = []
    for m in re.finditer(r'public-test run:\s*(\d+)/(\d+)', text):
        results.append((int(m.group(1)), int(m.group(2))))
    return results

def extract_final_code_info(text):
    m = re.search(r'## FINAL CODE\n+```python\n(.*?)\n```', text, re.DOTALL)
    if not m:
        # Maybe final code is empty
        if '## FINAL CODE' in text:
            snippet = text[text.index('## FINAL CODE'):][:500]
            return snippet[:80], False, False, False, True
        return '(n/a)', False, False, False, False
    code = m.group(1)
    has_solution = bool(re.search(r'class\s+Solution', code))
    has_main = bool(re.search(r'if\s+__name__\s*==', code))
    has_stdin_read = bool(re.search(r'(?:input\s*\(|sys\.stdin|raw_input)', code))
    return code[:100], has_solution, has_main, has_stdin_read, len(code) > 0

for prefix in ['cand_tthe_hard50_b2r0_g0', 'cand_tthe_hard50_b2r0_g1', 'cand_tthe_hard50_b2r0_g2']:
    print(f'=== {prefix} ===')
    total_pass = 0
    total_all = 0
    for q in range(10):
        f = os.path.join(TRACES, f'{prefix}__q{q}.md')
        with open(f) as fh:
            text = fh.read()

        qid_m = re.search(r'Q(\d+)\s+\[(\w+)', text)
        qid_num = qid_m.group(1) if qid_m else '?'
        qid_name = qid_m.group(2) if qid_m else '?'

        # Get ALL test results including final section
        results = extract_results(text)

        # Also check the PUBLIC-TEST RESULTS section at the bottom
        last_section = text.split('## PUBLIC-TEST RESULTS')
        final_section_result = None
        if len(last_section) > 1:
            section = last_section[-1]
            m = re.search(r'(\d+)\s*/\s*(\d+)\s+passed', section)
            if m:
                final_section_result = (int(m.group(1)), int(m.group(2)))

        # Check for got='' (empty output)
        empty_count = len(re.findall(r"got=''" , text))
        crash_count = len(re.findall(r"err='[^']+'", text))

        code_preview, has_sol, has_main, has_stdin, has_code = extract_final_code_info(text)

        steps = re.findall(r'### step \d+ — (.+)', text)
        n_llm = sum(1 for s in steps if 'coder call' in s)
        n_eval = sum(1 for s in steps if 'public-test' in s)

        # Print all per-step results
        result_str = ' '.join(f'{p}/{t}' for p, t in results)

        status = '?'
        if final_section_result:
            p, t = final_section_result
            status = f'{t==p and t>0 and "ALL" or ""} {p}/{t}'

        empty_str = f' EMPTYx{empty_count}' if empty_count > 0 else ''
        crash_str = f' CRASHx{crash_count}' if crash_count > 0 else ''
        sol_str = f' sol={has_sol}' if has_sol else ''
        main_str = f' main={has_main}' if has_main else ''

        print(f'  Q{q}[{qid_name}]: {status}{empty_str}{crash_str}{sol_str}{main_str}  [{result_str}]')

        # Track pass info
        if final_section_result:
            p, t = final_section_result
            total_pass += p
            total_all += t

    if total_all > 0:
        print(f'  TOTAL: {total_pass}/{total_all} = {total_pass/total_all*100:.1f}%')
    print()
