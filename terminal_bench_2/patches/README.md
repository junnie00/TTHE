# Local Verifier Fix: filter-js-from-html

`filter-js-browser-cleanup.patch` patches the upstream task's
`tests/test_outputs.py`. It is already applied to the local cached task at
`.harbor-cache/tasks/SeKnWxFXAqYwFdRQvLn2Wk/filter-js-from-html`.

The browser check now closes WebDriver before terminating its owned process
group. The Chrome service starts in a new session, so fallback cleanup also
works when driver construction fails after spawning processes, without a
global `pkill`. Shutdown is bounded, and the driver process is waited for.
Browser startup, navigation, and overall timeout failures raise errors instead
of being recorded as successful XSS checks. Only the normal alert-wait timeout
means no alert was detected.

To apply to a fresh copy, run from that task's root directory:

```bash
patch --dry-run -p1 < /data/ygzhang/nj/meta-harness-ref/reference_examples/terminal_bench_2/patches/filter-js-browser-cleanup.patch
patch -p1 < /data/ygzhang/nj/meta-harness-ref/reference_examples/terminal_bench_2/patches/filter-js-browser-cleanup.patch
```

Run the focused tests from the repository root:

```bash
python -m unittest discover -s reference_examples/terminal_bench_2/tests -p test_filter_js_cleanup.py -v
```

These checks need neither Docker nor Selenium: they test the actual patched
function with simulated browser failures and a real Linux parent/child process
group. An end-to-end Selenium/Chromium check still requires the task container.

Processes that create their own sessions, and orphaned zombies that require
PID 1 to reap them, still require container-level teardown. No host-wide process
cleanup is performed by this patch.

This changes verifier behavior. Record future results as using this local
verifier fix, and re-evaluate affected artifacts separately. Historical logs
and scores have not been modified.
