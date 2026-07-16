"""ASE-SQL: label-free co-adversarial self-evolution for Text-to-SQL.

See doc/experiment.md §0.5 for the design. Modules:
  llm.py          API client (controller / solver) + parsing helpers
  db.py           SQLite execution, schema introspection, result comparison
  demo_db.py      tiny built-in database so the loop runs offline
  dataset.py      Dataset interface: DemoDataset + BirdDataset (BIRD dev)
  agent_config.py the evolvable config ("小抄") + escalation ladder definition
  probe.py        Probe dataclass
  solver.py       Solver (考生): question + config -> SQL
  proposer.py     Proposer (考官): generates well-posed probes; evolves its strategy
  evaluate.py     run solver on probes, score, build weakness profile
  ladder.py       solver-side escalation ladder L0..L5 (cheapest fix first)
  ceiling.py      per-skill state machine: reachable / solved / ceiling
  loop.py         the unified evolution loop; single vs coevolve is one flag
  archive.py      append-only jsonl evolution log
"""
