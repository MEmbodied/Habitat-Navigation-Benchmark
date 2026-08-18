"""Persistent per-worker episode plans for resumable evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence


EpisodeKey = tuple[str, str]


def load_or_create_episode_plan(
    path: str | Path,
    candidate_keys: Sequence[EpisodeKey],
    *,
    shard_rank: int,
    num_shards: int,
    max_episodes: int,
) -> tuple[EpisodeKey, ...]:
    """Return the immutable plan owned by one worker output directory."""
    plan_path = Path(path)
    if plan_path.is_file():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        return tuple(
            (str(item["scene_id"]), str(item["episode_id"]))
            for item in payload["episodes"]
        )

    plan = list(candidate_keys[shard_rank::num_shards])
    if max_episodes > 0:
        plan = plan[:max_episodes]
    payload = {
        "shard_rank": int(shard_rank),
        "num_shards": int(num_shards),
        "episode_count": len(plan),
        "episodes": [
            {"scene_id": scene_id, "episode_id": episode_id}
            for scene_id, episode_id in plan
        ],
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, plan_path)
    return tuple(plan)


def remaining_episode_keys(
    plan: Sequence[EpisodeKey], completed: Iterable[EpisodeKey]
) -> tuple[EpisodeKey, ...]:
    completed_keys = set(completed)
    return tuple(key for key in plan if key not in completed_keys)


__all__ = ["EpisodeKey", "load_or_create_episode_plan", "remaining_episode_keys"]
