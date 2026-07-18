import os, re

traces = '/data/ygzhang/nj/TTHE_mono/livecodebench/logs/tthe_hard50/traces/b2'

for prefix in ['cand_tthe_hard50_b2r0_g1']:
    for q in range(10):
        f = os.path.join(traces, f'{prefix}__q{q}.md')
        with open(f) as fh:
            text = fh.read()

        qid_m = re.search(r'Q(\d+)\s+\[(\w+)', text)
        qid = qid_m.group(1) if qid_m else '?'

        # Check final code
        final_code = ''
        m = re.search(r'## FINAL CODE\n+\n*```python\n(.*?)\n```', text, re.DOTALL)
        if not m:
            # Maybe empty code
            idx = text.find('## FINAL CODE')
            if idx >= 0:
                rest = text[idx:][:200]
            else:
                rest = 'N/A'
            print(f'Q{q}[{qid}]: NO_FINAL_CODE  rest={rest[:80]}')
            continue
        final_code = m.group(1).strip()

        has_class_sol = 'class Solution' in final_code
        has_main = '__main__' in final_code
        has_input = 'input(' in final_code or 'sys.stdin' in final_code or 'raw_input' in final_code

        # Check responses for class Solution
        responses = re.findall(r'RESPONSE:\n(.+?)(?=\n### step|\n## FINAL)', text, re.DOTALL)
        class_sol_in_responses = sum(1 for r in responses if 'class Solution' in r)

        print(f'Q{q}[{qid}]: final_classSol={has_class_sol} final_main={has_main} final_stdin={has_input} classSol_in_resp={class_sol_in_responses}')
