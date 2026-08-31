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


def _method(name):
    module = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Evaluator.{name} was not found")


def _method_calls(name):
    return sorted(
        _attribute_name(call.func)
        for call in ast.walk(_method(name))
        if isinstance(call, ast.Call)
    )


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


def test_render_repair_checks_depth_finiteness_without_stepping_environment():
    depth_calls = _method_calls("_is_corrupt_depth")
    repair_calls = _method_calls("_repair_observation_render")

    assert "np.isfinite" in depth_calls
    assert "Evaluator._is_corrupt_depth" in repair_calls
    assert "self._fresh_sensor_observations" in repair_calls
    assert "self.env.step" not in repair_calls


def test_fresh_sensor_rerender_reapplies_habitat_sensor_transforms():
    calls = _method_calls("_fresh_sensor_observations")

    assert "self.env.sim.get_sensor_observations" in calls
    assert "self.env.sim.sensor_suite.get_observations" in calls


def test_render_repair_replaces_rgb_and_depth_independently():
    method = _method("_repair_observation_render")
    guarded_replacements = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name):
            continue
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if not isinstance(target.value, ast.Name) or target.value.id != "repaired":
                    continue
                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    guarded_replacements.add((node.test.id, target.slice.value))

    assert ("repair_rgb", "rgb") in guarded_replacements
    assert ("repair_depth", "depth") in guarded_replacements
