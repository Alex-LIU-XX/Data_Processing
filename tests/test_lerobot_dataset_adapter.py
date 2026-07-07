"""Tests for the LeRobot dataset adapter (v3.0) using the pick_banana dataset."""

import json
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.lerobot_dataset_adapter import (
    LeRobotDatasetAdapter,
    LeRobotDatasetMetadataAdapter,
    detect_dataset_version,
)

DATASET_ROOT = Path("/home/lxx/repo/datasets/lerobot/miku112/pick_banana_100_newTable_1_offset_state")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def meta() -> LeRobotDatasetMetadataAdapter:
    return LeRobotDatasetMetadataAdapter(
        "miku112/pick_banana_100_newTable_1_offset_state", root=str(DATASET_ROOT)
    )


@pytest.fixture(scope="module")
def info() -> dict:
    with open(DATASET_ROOT / "meta" / "info.json") as f:
        return json.load(f)


# ============================================================================
# detect_dataset_version
# ============================================================================

class TestDetectVersion:
    def test_detects_v3_0(self):
        assert detect_dataset_version(DATASET_ROOT) == "v3.0"

    def test_raises_on_missing_root(self):
        with pytest.raises(Exception):
            detect_dataset_version("/nonexistent/path")


# ============================================================================
# Properties from info.json
# ============================================================================

class TestInfoProperties:
    def test_repo_id(self, meta):
        assert meta.repo_id == "miku112/pick_banana_100_newTable_1_offset_state"

    def test_root(self, meta):
        assert str(meta.root) == str(DATASET_ROOT)

    def test_revision(self, meta):
        assert meta.revision == "v3.0"

    def test_version(self, meta):
        assert meta._version == packaging.version.parse("v3.0")

    def test_codebase_version_string(self, meta, info):
        assert meta.info["codebase_version"] == info["codebase_version"]

    def test_fps(self, meta):
        assert meta.fps == 10

    def test_robot_type(self, meta):
        assert meta.robot_type == "piper"

    def test_data_path_pattern(self, meta, info):
        assert meta.data_path == info["data_path"]

    def test_video_path_pattern(self, meta, info):
        assert meta.video_path == info["video_path"]

    def test_total_episodes(self, meta):
        assert meta.total_episodes == 100

    def test_total_frames(self, meta):
        assert meta.total_frames == 12209

    def test_total_tasks(self, meta):
        assert meta.total_tasks == 42

    def test_chunks_size(self, meta):
        assert meta.chunks_size == 1000


# ============================================================================
# Features
# ============================================================================

class TestFeatures:
    def test_feature_keys(self, meta):
        expected = {
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        assert set(meta.features) == expected

    def test_feature_shapes(self, meta):
        assert meta.features["observation.state"]["shape"] == [7]
        assert meta.features["action"]["shape"] == [7]
        assert meta.features["observation.images.image"]["shape"] == [3, 480, 640]
        assert meta.features["observation.images.wrist_image"]["shape"] == [3, 480, 640]

    def test_feature_dtypes(self, meta):
        assert meta.features["observation.state"]["dtype"] == "float64"
        assert meta.features["action"]["dtype"] == "float64"
        assert meta.features["observation.images.image"]["dtype"] == "video"
        assert meta.features["observation.images.wrist_image"]["dtype"] == "video"


# ============================================================================
# Video / Image / Camera keys
# ============================================================================

class TestMediaKeys:
    def test_video_keys(self, meta):
        assert set(meta.video_keys) == {
            "observation.images.image",
            "observation.images.wrist_image",
        }

    def test_image_keys(self, meta):
        assert meta.image_keys == []

    def test_camera_keys(self, meta):
        assert set(meta.camera_keys) == {
            "observation.images.image",
            "observation.images.wrist_image",
        }


# ============================================================================
# Names & Shapes
# ============================================================================

class TestNamesAndShapes:
    def test_names(self, meta):
        names = meta.names
        assert names["observation.state"] == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        assert names["action"] == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        assert names["observation.images.image"] == ["channels", "height", "width"]

    def test_shapes(self, meta):
        shapes = meta.shapes
        assert shapes["observation.state"] == (7,)
        assert shapes["action"] == (7,)
        assert shapes["observation.images.image"] == (3, 480, 640)
        assert shapes["timestamp"] == (1,)


# ============================================================================
# Tasks
# ============================================================================

class TestTasks:
    def test_tasks_count(self, meta):
        assert len(meta.tasks) == 42

    def test_task_indices(self, meta):
        assert meta.tasks[0] == "Acquire the banana and put it within the bowl."
        assert meta.tasks[1] == "Fetch the banana and position it in the bowl."

    def test_task_to_task_index(self, meta):
        assert meta.task_to_task_index["Acquire the banana and put it within the bowl."] == 0
        assert len(meta.task_to_task_index) == 42

    def test_get_task_index(self, meta):
        assert meta.get_task_index("Fetch the banana and position it in the bowl.") == 1
        assert meta.get_task_index("Nonexistent task") is None


# ============================================================================
# Episodes
# ============================================================================

class TestEpisodes:
    def test_episode_count(self, meta):
        assert len(meta.episodes) == 100

    def test_episode_0(self, meta):
        ep = meta.episodes[0]
        assert ep["episode_index"] == 0
        assert ep["length"] == 116
        assert "banana" in str(ep["tasks"][0]).lower()
        assert ep["data/chunk_index"] == 0
        assert ep["data/file_index"] == 0
        assert ep["dataset_from_index"] == 0
        assert ep["dataset_to_index"] == 116

    def test_episode_1(self, meta):
        ep = meta.episodes[1]
        assert ep["episode_index"] == 1
        assert ep["length"] == 120
        assert ep["dataset_from_index"] == 116
        assert ep["dataset_to_index"] == 236

    def test_episode_99(self, meta):
        ep = meta.episodes[99]
        assert ep["episode_index"] == 99
        assert ep["length"] == 123
        assert ep["dataset_from_index"] == 12086
        assert ep["dataset_to_index"] == 12209

    def test_episode_video_offsets(self, meta):
        ep0 = meta.episodes[0]
        assert ep0["videos/observation.images.image/from_timestamp"] == 0.0
        assert ep0["videos/observation.images.wrist_image/from_timestamp"] == 0.0

        ep1 = meta.episodes[1]
        assert ep1["videos/observation.images.image/from_timestamp"] == 11.6
        assert ep1["videos/observation.images.wrist_image/from_timestamp"] == 11.6


# ============================================================================
# Path resolution
# ============================================================================

class TestPathResolution:
    def test_get_data_file_path_ep0(self, meta):
        path = meta.get_data_file_path(0)
        assert str(path) == "data/chunk-000/file-000.parquet"

    def test_get_data_file_path_ep99(self, meta):
        path = meta.get_data_file_path(99)
        assert str(path) == "data/chunk-000/file-000.parquet"

    def test_get_video_file_path_ep0(self, meta):
        path = meta.get_video_file_path(0, "observation.images.image")
        assert str(path) == "videos/observation.images.image/chunk-000/file-000.mp4"

    def test_get_video_file_path_ep1(self, meta):
        path = meta.get_video_file_path(1, "observation.images.wrist_image")
        assert str(path) == "videos/observation.images.wrist_image/chunk-000/file-000.mp4"

    def test_get_episode_chunk(self, meta):
        assert meta.get_episode_chunk(0) == 0
        assert meta.get_episode_chunk(99) == 0


# ============================================================================
# Unsupported operations
# ============================================================================

class TestUnsupported:
    def test_add_task_raises(self, meta):
        with pytest.raises(NotImplementedError, match="add_task"):
            meta.add_task("some task")

    def test_save_episode_raises(self, meta):
        with pytest.raises(NotImplementedError, match="save_episode"):
            meta.save_episode(0, 10, ["task"], {})

    def test_pull_from_repo_raises(self, meta):
        with pytest.raises(NotImplementedError, match="pull_from_repo"):
            meta.pull_from_repo()


# ============================================================================
# Stats
# ============================================================================

class TestStats:
    def test_stats_loaded(self, meta):
        assert meta.stats is not None, "stats should not be None"
        assert "observation.state" in meta.stats, f"Keys: {list(meta.stats.keys())}"
        assert "action" in meta.stats

    def test_state_stats_shape(self, meta):
        state_stats = meta.stats["observation.state"]
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            assert key in state_stats, f"Missing {key} in state stats"
            assert len(state_stats[key]) == 7, f"{key} should have 7 values"

    def test_episodes_stats(self, meta):
        assert len(meta.episodes_stats) > 0


# ============================================================================
# __repr__
# ============================================================================

class TestRepr:
    def test_repr_contains_repo_id(self, meta):
        r = repr(meta)
        assert "miku112/pick_banana_100_newTable_1_offset_state" in r

    def test_repr_contains_episode_count(self, meta):
        r = repr(meta)
        assert "100" in r or "episodes" in r


# ============================================================================
# LeRobotDatasetAdapter integration tests
# ============================================================================

def _make_adapter() -> LeRobotDatasetAdapter:
    return LeRobotDatasetAdapter(
        repo_id="miku112/pick_banana_100_newTable_1_offset_state",
        root=str(DATASET_ROOT),
        delta_timestamps=None,
    )


class TestAdapterQuery:
    def test_basic_query(self):
        dataset = _make_adapter()
        assert len(dataset) > 0

        item = dataset[0]
        expected_keys = {
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "task",
        }
        for key in expected_keys:
            assert key in item, f"Missing key: {key}"

        assert item["observation.state"].shape == (7,)
        assert item["action"].shape == (7,)
        assert isinstance(item["observation.images.image"], torch.Tensor)
        assert isinstance(item["observation.images.wrist_image"], torch.Tensor)
        assert isinstance(item["task"], str)
        assert len(item["task"]) > 0
        assert item["episode_index"].item() == 0

    def test_multi_episode_boundary(self):
        dataset = _make_adapter()
        item0 = dataset[0]
        assert item0["episode_index"].item() == 0

        # Episode 0 has length 116, so frame 116 is the first frame of episode 1
        item1 = dataset[116]
        assert item1["episode_index"].item() == 1

    def test_out_of_range(self):
        dataset = _make_adapter()
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    def test_task_mapping(self):
        dataset = _make_adapter()
        item = dataset[0]
        task_str = item["task"]
        expected = dataset.meta.tasks[0]
        assert task_str == expected

    def test_with_delta_timestamps(self):
        dataset = LeRobotDatasetAdapter(
            repo_id="miku112/pick_banana_100_newTable_1_offset_state",
            root=str(DATASET_ROOT),
            delta_timestamps={"action": [t / 10.0 for t in range(10)]},
        )
        item = dataset[0]
        assert item["action"].shape == (10, 7)
        assert "action_is_pad" in item
        assert item["action_is_pad"].shape == (10,)

