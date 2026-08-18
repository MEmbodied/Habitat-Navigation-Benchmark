from __future__ import annotations

import json

from internnav.evaluator.episode_plan import (
    load_or_create_episode_plan,
    remaining_episode_keys,
)


def test_worker_restart_keeps_original_shard_plan(tmp_path) -> None:
    plan_path = tmp_path / "episode_plan.json"
    candidates = [("scene", str(index)) for index in range(6)]

    original = load_or_create_episode_plan(
        plan_path,
        candidates,
        shard_rank=1,
        num_shards=2,
        max_episodes=3,
    )
    original_payload = plan_path.read_text(encoding="utf-8")

    restarted = load_or_create_episode_plan(
        plan_path,
        [("scene", "0"), ("scene", "2"), ("scene", "3")],
        shard_rank=1,
        num_shards=2,
        max_episodes=3,
    )

    assert original == (("scene", "1"), ("scene", "3"), ("scene", "5"))
    assert restarted == original
    assert remaining_episode_keys(restarted, {("scene", "1")}) == (
        ("scene", "3"),
        ("scene", "5"),
    )
    assert plan_path.read_text(encoding="utf-8") == original_payload
    assert json.loads(original_payload)["episode_count"] == 3


def test_workers_receive_disjoint_plans(tmp_path) -> None:
    candidates = [("scene", str(index)) for index in range(7)]
    plans = [
        load_or_create_episode_plan(
            tmp_path / f"worker_{rank}" / "episode_plan.json",
            candidates,
            shard_rank=rank,
            num_shards=3,
            max_episodes=0,
        )
        for rank in range(3)
    ]

    assert set(plans[0]).isdisjoint(plans[1])
    assert set(plans[0]).isdisjoint(plans[2])
    assert set(plans[1]).isdisjoint(plans[2])
    assert set().union(*(set(plan) for plan in plans)) == set(candidates)
