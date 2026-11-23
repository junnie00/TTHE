"""Static invariant audit for evolved Terminal-Bench AgentHarness files."""

from __future__ import annotations

import ast
from pathlib import Path


FROZEN_ATTRS = {"_model_name", "_llm", "_max_episodes"}
FORBIDDEN_KWARGS = {"api_key", "api_base", "base_url", "max_episodes"}
GOLD_LITERALS = {
    "verifier_result",
    "/logs/verifier",
    "/tests/test.sh",
    "/solution/solve.sh",
    "reward.txt",
    "rewards.json",
}


def _is_self_attr(node: ast.AST, *attrs: str) -> bool:
    current = node
    for attr in reversed(attrs):
        if not isinstance(current, ast.Attribute) or current.attr != attr:
            return False
        current = current.value
    return isinstance(current, ast.Name) and current.id == "self"


def _is_fixed_model_value(node: ast.AST) -> bool:
    # baseline_kira forwards its ``model`` method parameter after every call site
    # has supplied ``self._model_name``; direct call-site model kwargs are audited too.
    return _is_self_attr(node, "_model_name") or (
        isinstance(node, ast.Name) and node.id == "model"
    )


def audit_source(
    source: str,
    filename: str = "<agent>",
    forbidden_task_names: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [{"rule": "PARSE", "line": exc.lineno or 0, "detail": exc.msg}]

    violations: list[dict[str, object]] = []
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
    ]
    if not classes:
        violations.append(
            {"rule": "INTERFACE", "line": 0, "detail": "missing AgentHarness class"}
        )
    elif not any(
        (isinstance(base, ast.Name) and base.id == "Terminus2")
        or (isinstance(base, ast.Attribute) and base.attr == "Terminus2")
        for base in classes[0].bases
    ):
        violations.append(
            {
                "rule": "INTERFACE",
                "line": classes[0].lineno,
                "detail": "AgentHarness must directly subclass Terminus2",
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    name = (
                        part.attr
                        if isinstance(part, ast.Attribute)
                        else part.id
                        if isinstance(part, ast.Name)
                        else ""
                    )
                    if name in FROZEN_ATTRS:
                        violations.append(
                            {
                                "rule": "FROZEN_SOLVER",
                                "line": node.lineno,
                                "detail": f"assigns fixed field {name}",
                            }
                        )
        if isinstance(node, ast.Call):
            for kwarg in node.keywords:
                if kwarg.arg in FORBIDDEN_KWARGS:
                    violations.append(
                        {
                            "rule": "FROZEN_SOLVER",
                            "line": node.lineno,
                            "detail": f"passes forbidden kwarg {kwarg.arg}",
                        }
                    )
                if kwarg.arg == "model" and not _is_fixed_model_value(kwarg.value):
                    violations.append(
                        {
                            "rule": "FROZEN_SOLVER",
                            "line": node.lineno,
                            "detail": "model must remain self._model_name",
                        }
                    )
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                key_value = key.value if isinstance(key, ast.Constant) else None
                if key_value == "model" and not _is_fixed_model_value(value):
                    violations.append(
                        {
                            "rule": "FROZEN_SOLVER",
                            "line": node.lineno,
                            "detail": "model dictionary entry must remain self._model_name",
                        }
                    )
                if key_value in {"api_key", "base_url"}:
                    violations.append(
                        {
                            "rule": "FROZEN_SOLVER",
                            "line": node.lineno,
                            "detail": f"constructs protected connection field {key_value}",
                        }
                    )
                if key_value == "api_base" and not _is_self_attr(
                    value, "_llm", "_api_base"
                ):
                    violations.append(
                        {
                            "rule": "FROZEN_SOLVER",
                            "line": node.lineno,
                            "detail": "api_base must remain self._llm._api_base",
                        }
                    )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for literal in GOLD_LITERALS:
                if literal.lower() in lowered:
                    violations.append(
                        {
                            "rule": "LABEL_FREE",
                            "line": node.lineno,
                            "detail": f"references protected verifier artifact {literal!r}",
                        }
                    )
            for task_name in forbidden_task_names:
                if task_name.lower() in lowered:
                    violations.append(
                        {
                            "rule": "NO_TASK_HARDCODING",
                            "line": node.lineno,
                            "detail": f"hardcodes task name {task_name!r}",
                        }
                    )

    unique: list[dict[str, object]] = []
    seen = set()
    for item in violations:
        key = (item["rule"], item["line"], item["detail"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return sorted(unique, key=lambda item: int(item["line"]))


def audit_file(
    path: Path, forbidden_task_names: tuple[str, ...] = ()
) -> list[dict[str, object]]:
    return audit_source(
        path.read_text(encoding="utf-8"),
        str(path),
        forbidden_task_names=forbidden_task_names,
    )
