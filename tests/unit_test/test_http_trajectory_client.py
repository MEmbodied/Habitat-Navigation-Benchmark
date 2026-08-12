import importlib.util
from pathlib import Path
import sys
import types

import json_numpy
import numpy as np
import pytest
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_http_client_module(monkeypatch):
    internnav_package = types.ModuleType("internnav")
    internnav_package.__path__ = [str(PROJECT_ROOT / "internnav")]
    evaluator_package = types.ModuleType("internnav.evaluator")
    evaluator_package.__path__ = [str(PROJECT_ROOT / "internnav" / "evaluator")]
    evaluator_stub = types.ModuleType(
        "internnav.evaluator.final_habitat_vln_evaluator"
    )
    evaluator_stub.BaseTrajectoryClient = object
    monkeypatch.setitem(sys.modules, "internnav", internnav_package)
    monkeypatch.setitem(sys.modules, "internnav.evaluator", evaluator_package)
    monkeypatch.setitem(
        sys.modules,
        "internnav.evaluator.final_habitat_vln_evaluator",
        evaluator_stub,
    )

    canonical_name = "internnav.evaluator.canonical_action"
    canonical_spec = importlib.util.spec_from_file_location(
        canonical_name,
        PROJECT_ROOT / "internnav" / "evaluator" / "canonical_action.py",
    )
    canonical_module = importlib.util.module_from_spec(canonical_spec)
    monkeypatch.setitem(sys.modules, canonical_name, canonical_module)
    canonical_spec.loader.exec_module(canonical_module)

    client_spec = importlib.util.spec_from_file_location(
        "habitat_http_trajectory_client",
        PROJECT_ROOT / "internnav" / "evaluator" / "HTTPTrajectoryClient.py",
    )
    client_module = importlib.util.module_from_spec(client_spec)
    client_spec.loader.exec_module(client_module)
    return client_module


class _Response:
    def __init__(self, payload):
        self.text = json_numpy.dumps(payload)

    def raise_for_status(self):
        return None


def _observation(**values):
    observation = {
        "instruction": "go forward",
        "gps": [0.0, 0.0],
        "yaw": 0.0,
        "camera_height": 0.0,
    }
    observation.update(values)
    return observation


def test_prepare_observation_preserves_habitat_native_coordinates(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    client = client_module.Gr00tTrajectoryClient("http://policy/act")

    prepared = client._prepare_observation_payload(
        _observation(
            gps=np.array([1.25, -0.5], dtype=np.float32),
            yaw=np.float32(np.deg2rad(90.0)),
            camera_height=np.float32(0.75),
            reference_path_gps=[[0.0, 0.0], [2.0, 0.5]],
            metrics={
                "reference_path_gps": [[0.0, 0.0], [2.0, 0.5]],
                "goal_gps": [3.0, -1.0],
            },
        )
    )

    assert "state_xyzyaw" not in prepared
    assert np.allclose(prepared["gps"], [1.25, -0.5])
    assert np.allclose(prepared["reference_path_gps"], [[0.0, 0.0], [2.0, 0.5]])
    assert np.allclose(
        prepared["metrics"]["reference_path_gps"],
        [[0.0, 0.0], [2.0, 0.5]],
    )
    assert np.allclose(prepared["metrics"]["goal_gps"], [3.0, -1.0])


def test_action_response_keeps_server_owned_habitat_oracle_goal(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    response = client_module._action_response(
        {"oracle_goal_gps": [2.0, 1.5]},
        [],
        0,
    )

    assert np.allclose(response["oracle_goal_gps"], [2.0, 1.5])


def test_query_negotiates_chunk_and_reuses_feedback_during_replan(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    chunk = np.zeros((16, 4), dtype=np.float32)
    chunk[0, 0] = 0.25
    responses = iter(
        [
            {
                "schema_version": 2,
                "control_mode": "continuous",
                "stop": False,
                "replan_required": True,
                "control_event": {
                    "type": "high_policy_replan",
                    "token": "segment-7",
                },
            },
            {
                "schema_version": 2,
                "control_mode": "continuous",
                "stop": False,
                "action_horizon": 16,
                "valid_action_dim": 4,
                "execution_semantics": "canonical_relative_v1",
                "action_format": "relative_delta_xyzyaw",
                "action_unit": {"xyz": "m", "yaw": "deg"},
                "continuous_action": chunk,
                "chunk_execute_horizon": 1,
            },
        ]
    )
    requests = []

    def fake_post(url, data, headers, timeout):
        requests.append(json_numpy.loads(data))
        assert url == "http://policy/act"
        assert headers == {"Content-Type": "application/json"}
        assert timeout == 300.0
        return _Response(next(responses))

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = client_module.Gr00tTrajectoryClient("http://policy/act")

    result = client.query(_observation(executed_actions=[2, 1]))

    assert result["actions"] == [1]
    assert result["replan_rounds"] == 1
    assert result["action_transport"] == "chunk"
    assert result["schema_version"] == 2
    assert result["chunk_execute_horizon"] == 1
    assert len(requests) == 2
    for request in requests:
        observation = request["observation"]
        assert observation["executed_actions"] == [2, 1]
        assert "state_xyzyaw" not in observation
        assert observation["client_capabilities"] == {
            "action_transports": ["chunk", "discrete"],
            "execution_semantics": ["canonical_relative_v1"],
            "control_protocols": ["high_policy_replan_ack_v1"],
        }
    assert requests[1]["observation"]["control_event"] == {
        "type": "high_policy_replan_ack",
        "token": "segment-7",
    }


def test_query_rejects_legacy_http_200_error_response(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    def fake_post(*args, **kwargs):
        del args, kwargs
        return _Response({"error": "invalid observation", "stop": True})

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = client_module.Gr00tTrajectoryClient("http://policy/act")

    with pytest.raises(RuntimeError, match="invalid observation"):
        client.query(_observation())


def test_replan_control_requires_non_empty_token(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    with pytest.raises(ValueError, match="non-empty token"):
        client_module._replan_ack(
            {
                "replan_required": True,
                "control_event": {
                    "type": "high_policy_replan",
                    "token": "",
                },
            }
        )


def test_query_retries_same_ack_after_transport_failure(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    chunk = np.zeros((16, 4), dtype=np.float32)
    responses = iter(
        [
            _Response(
                {
                    "schema_version": 2,
                    "control_mode": "continuous",
                    "stop": False,
                    "replan_required": True,
                    "control_event": {
                        "type": "high_policy_replan",
                        "token": "segment-11",
                    },
                }
            ),
            requests.ConnectionError("response lost"),
            _Response(
                {
                    "schema_version": 2,
                    "control_mode": "continuous",
                    "stop": False,
                    "action_horizon": 16,
                    "valid_action_dim": 4,
                    "execution_semantics": "canonical_relative_v1",
                    "action_format": "relative_delta_xyzyaw",
                    "action_unit": {"xyz": "m", "yaw": "deg"},
                    "continuous_action": chunk,
                    "chunk_execute_horizon": 1,
                }
            ),
        ]
    )
    requests_seen = []

    def fake_post(url, data, headers, timeout):
        del url, headers, timeout
        requests_seen.append(json_numpy.loads(data))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    result = client_module.Gr00tTrajectoryClient("http://policy/act").query(
        _observation()
    )

    assert result["actions"] == [1]
    assert len(requests_seen) == 3
    assert requests_seen[1] == requests_seen[2]
    assert requests_seen[1]["observation"]["control_event"]["token"] == "segment-11"
