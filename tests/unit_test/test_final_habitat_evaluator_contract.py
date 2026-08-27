import ast
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "internnav"
    / "evaluator"
    / "final_habitat_vln_evaluator.py"
)


def _attribute_name(node):
    values = []
    while isinstance(node, ast.Attribute):
        values.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        values.append(node.id)
    return ".".join(reversed(values))


def _run_episode_calls():
    module = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "run_episode":
            return sorted(
                (
                    call.lineno,
                    _attribute_name(call.func),
                )
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
    raise AssertionError("Evaluator.run_episode was not found")


def test_demo_reset_boundary_does_not_step_target_environment():
    calls = _run_episode_calls()
    reset_line = next(line for line, name in calls if name == "self.agent.reset")
    first_query_line = next(line for line, name in calls if name == "self.agent.act")

    assert reset_line < first_query_line
    assert not [
        line
        for line, name in calls
        if name == "self.env.step" and reset_line < line < first_query_line
    ]


def test_terminal_release_precedes_completed_result_append():
    calls = _run_episode_calls()
    end_line = next(line for line, name in calls if name == "self.agent.on_episode_end")
    result_open_line = next(
        line
        for line, name in calls
        if name == "open" and line > end_line
    )

    assert end_line < result_open_line
