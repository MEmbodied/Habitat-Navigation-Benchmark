from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ENACTIVE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ENACTIVE_ROOT))

from internnav.evaluator.dagger_lifecycle import DaggerEpisodeAbort  # noqa: E402
from internnav.evaluator.final_habitat_vln_evaluator import (  # noqa: E402
    Action,
    Evaluator,
    LLMAgent,
)


def _observation() -> dict[str, np.ndarray]:
    return {
        "gps": np.asarray([0.0, 0.0], dtype=np.float32),
        "compass": np.asarray([0.0], dtype=np.float32),
    }


class _Agent:
    def __init__(self) -> None:
        self.active_episode = ""
        self.reset_ids: list[str] = []
        self.terminal_events: list[dict] = []

    def reset(self, instruction: str, **kwargs) -> None:
        del instruction
        self.active_episode = str(kwargs["episode_id"])
        self.reset_ids.append(self.active_episode)

    def act(self, observation) -> Action:
        del observation
        if self.active_episode == "bad":
            raise DaggerEpisodeAbort("navmesh_path_unreachable")
        return Action.STOP

    def on_episode_end(self, event: dict) -> None:
        self.terminal_events.append(dict(event))
        self.active_episode = ""


class _Env:
    def __init__(self) -> None:
        self.episode_over = False
        self.step_count = 0
        self.sim = SimpleNamespace(
            get_agent_state=lambda: SimpleNamespace(
                position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            )
        )

    def step(self, action: int) -> dict[str, np.ndarray]:
        assert action == Action.STOP.value
        self.step_count += 1
        self.episode_over = True
        return _observation()

    def get_metrics(self) -> dict[str, float]:
        return {
            "distance_to_goal": 0.1 if self.episode_over else 5.0,
            "success": 1.0 if self.episode_over else 0.0,
            "spl": 1.0 if self.episode_over else 0.0,
            "ndtw": 1.0 if self.episode_over else 0.0,
        }


def _episode(episode_id: str):
    return SimpleNamespace(
        episode_id=episode_id,
        scene_id="/datasets/scene/scene.glb",
        instruction=SimpleNamespace(instruction_text=f"instruction-{episode_id}"),
    )


def test_teacher_unavailable_aborts_one_episode_then_next_episode_runs(tmp_path) -> None:
    evaluator = object.__new__(Evaluator)
    evaluator.agent = _Agent()
    evaluator.env = _Env()
    evaluator.max_steps = 4
    evaluator.output_path = str(tmp_path)
    evaluator.save_snapshots = False
    evaluator.vis_frames = []
    evaluator._prev_video_map_coord = None
    evaluator.sucs = []
    evaluator.spls = []
    evaluator.oss = []
    evaluator.nes = []
    evaluator.steps = []
    evaluator._last_result_record = None
    evaluator.args = SimpleNamespace(manual_instruction="")
    evaluator.config = SimpleNamespace(
        habitat=SimpleNamespace(
            task=SimpleNamespace(
                measurements=SimpleNamespace(
                    success=SimpleNamespace(success_distance=3.0)
                )
            )
        )
    )

    def init_episode(episode):
        del episode
        evaluator.env.episode_over = False
        evaluator.initial_yaw = 0.0
        evaluator.initial_height = 0.0
        return _observation(), 0.0, 5.0

    evaluator._init_episode = init_episode
    evaluator._build_observation = lambda *args, **kwargs: SimpleNamespace()
    evaluator._repair_observation_render = lambda observations, reason: observations

    evaluator.run_episode(_episode("bad"))
    evaluator.run_episode(_episode("good"))

    rows = [
        json.loads(line)
        for line in (tmp_path / "result.json").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["termination_kind"] for row in rows] == [
        "accepted_teacher_unavailable",
        "model_stop",
    ]
    assert rows[0]["termination_reason"] == "navmesh_path_unreachable"
    assert rows[0]["success"] == 0.0
    assert evaluator.agent.reset_ids == ["bad", "good"]
    assert [event["termination_kind"] for event in evaluator.agent.terminal_events] == [
        "accepted_teacher_unavailable",
        "model_stop",
    ]
    assert evaluator.env.step_count == 1


class _StopFollower:
    def get_next_action(self, goal) -> int:
        del goal
        return Action.STOP.value


def _oracle_agent(distance: float) -> LLMAgent:
    agent = object.__new__(LLMAgent)
    agent._dagger_oracle_goal_world = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    agent._dagger_oracle_goal_index = 1
    agent._dagger_oracle_remaining = 8
    agent._dagger_oracle_phase = "rejoin"
    agent._dagger_oracle_source = "dagger"
    agent._dagger_oracle_follower = _StopFollower()
    agent._dagger_oracle_goal_radius_m = 0.35
    agent.env = SimpleNamespace(
        sim=SimpleNamespace(
            get_agent_state=lambda: SimpleNamespace(
                position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            ),
            geodesic_distance=lambda start, goal: distance,
        )
    )
    return agent


def test_native_oracle_stop_uses_habitat_sim_geodesic_distance() -> None:
    agent = _oracle_agent(0.2)

    assert agent._next_dagger_oracle_action() is None
    assert agent._dagger_oracle_follower is None


def test_native_oracle_false_stop_aborts_dagger_episode() -> None:
    agent = _oracle_agent(1.0)

    try:
        agent._next_dagger_oracle_action()
    except DaggerEpisodeAbort as exc:
        assert str(exc).startswith("native_oracle_false_reached:")
    else:
        raise AssertionError("far native-oracle STOP must abort the DAgger episode")
