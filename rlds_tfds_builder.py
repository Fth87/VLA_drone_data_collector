from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = "roblox_vla_rlds"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "dataset"
MAX_EPISODES_ENV = "ROBLOX_VLA_MAX_EPISODES"
SOURCE_DIR_ENV = "ROBLOX_VLA_SOURCE_DIR"


def _resolve_source_dir() -> Path:
    source_dir = os.environ.get(SOURCE_DIR_ENV)
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    return DEFAULT_SOURCE_DIR.resolve()


def _resolve_episode_limit() -> int | None:
    raw_value = os.environ.get(MAX_EPISODES_ENV)
    if not raw_value:
        return None
    episode_limit = int(raw_value)
    if episode_limit <= 0:
        raise ValueError(f"{MAX_EPISODES_ENV} must be greater than zero")
    return episode_limit


def _load_episode(episode_path: Path) -> dict:
    return json.loads(episode_path.read_text(encoding="utf-8"))


def _episode_paths(source_dir: Path, limit: int | None = None) -> list[Path]:
    data_dir = source_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset data directory not found: {data_dir}")

    episode_paths = sorted(data_dir.glob("episode_*.json"))
    if limit is not None:
        episode_paths = episode_paths[:limit]
    return episode_paths


def _step_image_path(source_dir: Path, image_value: str) -> Path:
    image_path = source_dir / image_value
    if not image_path.exists():
        raise FileNotFoundError(f"Frame image not found: {image_path}")
    return image_path


class RobloxVlaRldsBuilder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for the Roblox VLA episodic dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial TFDS/RLDS export from episode JSON and PNG frames.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Roblox VLA episodic dataset in RLDS-compatible TFDS format. "
                "Each example is one episode containing a natural-language instruction, "
                "episode metadata, and ordered step sequences of image observations and actions."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "episode_id": tf.int64,
                    "instruction": tfds.features.Text(),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "success": tf.bool,
                            "num_steps": tf.int64,
                            "fps": tf.int64,
                        }
                    ),
                    "steps": tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                "timestep": tf.int32,
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(shape=(None, None, 3)),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                                "is_first": tf.bool,
                                "is_last": tf.bool,
                                "is_terminal": tf.bool,
                            }
                        )
                    ),
                }
            ),
            disable_shuffling=True,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        source_dir = _resolve_source_dir()
        episode_limit = _resolve_episode_limit()
        episode_paths = _episode_paths(source_dir, episode_limit)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "source_dir": source_dir,
                    "episode_paths": episode_paths,
                },
            )
        ]

    def _generate_examples(
        self,
        source_dir: Path,
        episode_paths: list[Path],
    ) -> Iterator[tuple[str, dict]]:
        for episode_path in episode_paths:
            episode = _load_episode(episode_path)
            steps = episode.get("steps") or []
            metadata = episode.get("metadata") or {}

            episode_id = int(episode["episode_id"])
            expected_steps = int(metadata.get("num_steps", len(steps)))
            if expected_steps != len(steps):
                raise ValueError(
                    f"Episode {episode_path.name} metadata.num_steps={expected_steps} "
                    f"does not match steps length={len(steps)}"
                )

            serialized_steps: list[dict] = []
            for index, step in enumerate(steps):
                timestep = int(step["timestep"])
                if timestep != index:
                    raise ValueError(
                        f"Episode {episode_path.name} has non-sequential timestep {timestep} at index {index}"
                    )

                image_value = step["observation"]["image"]
                image_path = _step_image_path(source_dir, image_value)
                action = step["action"]
                if len(action) != 4:
                    raise ValueError(
                        f"Episode {episode_path.name} step {index} action length must be 4, got {len(action)}"
                    )

                is_last = index == len(steps) - 1
                serialized_steps.append(
                    {
                        "timestep": timestep,
                        "observation": {"image": image_path},
                        "action": action,
                        "is_first": index == 0,
                        "is_last": is_last,
                        "is_terminal": bool(step.get("is_terminal", is_last)),
                    }
                )

            yield f"episode_{episode_id:04d}", {
                "episode_id": episode_id,
                "instruction": episode["instruction"],
                "metadata": {
                    "success": bool(metadata.get("success", True)),
                    "num_steps": len(serialized_steps),
                    "fps": int(metadata.get("fps", 0)),
                },
                "steps": serialized_steps,
            }


def build_builder(output_dir: Path | None = None) -> RobloxVlaRldsBuilder:
    if output_dir is None:
        return RobloxVlaRldsBuilder()
    return RobloxVlaRldsBuilder(data_dir=str(output_dir))
