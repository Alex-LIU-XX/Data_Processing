"""LeRobot dataset adapter supporting v3.0 and custom dataset formats.

The standard ``LeRobotDataset`` from the ``lerobot`` package only supports
v2.0/v2.1 format (one parquet file per episode, JSONL metadata). This adapter
extends it to handle:

- **v3.0**: multiple episodes per parquet file sliced by
  ``dataset_from_index``/``dataset_to_index``, parquet-based metadata,
  multi-episode video files with ``from_timestamp`` offsets.
"""

import json
import logging
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import packaging.version
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.utils import (
    get_delta_indices,
    get_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import decode_video_frames

logger = logging.getLogger(__name__)


class LeRobotDatasetVersionError(ValueError):
    """Raised when the dataset version is not supported."""


def detect_dataset_version(root: Path | str) -> str:
    """Detect the LeRobot dataset version from ``meta/info.json``."""
    info_path = Path(root) / "meta" / "info.json"
    try:
        with open(info_path) as f:
            info = json.load(f)
        return str(info.get("codebase_version", "v2.0"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise LeRobotDatasetVersionError(f"Cannot detect dataset version: {e}") from e


def _parse_version(v: str) -> packaging.version.Version:
    return packaging.version.parse(v)


# ============================================================================
# Metadata adapter — inherits from LeRobotDatasetMetadata
# ============================================================================

class LeRobotDatasetMetadataAdapter(lerobot_dataset.LeRobotDatasetMetadata):
    """Extends ``LeRobotDatasetMetadata`` to support v3.0 format from local disk.

    v2.x/v3.0 are detected from ``meta/info.json``.  The v3.0 path loads
    episodes and tasks from parquet files instead of JSONL, and never contacts
    the Hugging Face Hub.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        root_path = Path(root) if root is not None else lerobot_dataset.HF_LEROBOT_HOME / repo_id

        # Detect version from local info.json (avoids HF Hub call for v3.0)
        try:
            self.detected_version = detect_dataset_version(root_path)
        except LeRobotDatasetVersionError:
            self.detected_version = "v2.1"

        if _parse_version(self.detected_version) < _parse_version("v3.0"):
            # Delegate to parent — works unchanged
            super().__init__(repo_id=repo_id, root=root, revision=revision, force_cache_sync=force_cache_sync)
            return

        # ---- v3.0 path: skip parent init (avoids JSONL loading + HF Hub) ----
        # Use super() of the grandparent (object.__init__) to bypass
        # LeRobotDatasetMetadata.__init__
        super(lerobot_dataset.LeRobotDatasetMetadata, self).__init__()

        with open(root_path / "meta" / "info.json") as f:
            info = json.load(f)

        self.repo_id = repo_id
        self.revision = revision if revision else "v3.0"
        self.root = root_path
        self.info = info

        self.root.mkdir(exist_ok=True, parents=True)
        self.load_metadata()

    # ------------------------------------------------------------------
    # Properties — all inherited from parent (fps, features, video_keys,
    # camera_keys, total_episodes, total_frames, etc.)  They read from
    # self.info which is already loaded.
    # ------------------------------------------------------------------

    @property
    def _version(self) -> packaging.version.Version:
        return _parse_version(self.detected_version)

    # ------------------------------------------------------------------
    # Metadata loading override
    # ------------------------------------------------------------------

    def load_metadata(self):
        """Load metadata from local files.  Dispatches by version."""
        if self._version < _parse_version("v3.0"):
            super().load_metadata()
            return

        # v3.0 does not have tasks.jsonl / episodes.jsonl
        self.tasks, self.task_to_task_index = self._load_tasks_v3()
        self.episodes = self._load_episodes_v3()

        # stats.json still exists in v3.0
        from lerobot.common.datasets.utils import load_stats
        try:
            self.stats = load_stats(self.root)
        except FileNotFoundError:
            self.stats = {}

        # episodes_stats: for v3.0, store per-episode stats from the parquet metadata
        self.episodes_stats = {}
        for ep_idx, ep_meta in self.episodes.items():
            ep_stats = {}
            for key in self.features:
                for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                    col = f"stats/{key}/{stat_name}"
                    if col in ep_meta:
                        ep_stats.setdefault(key, {})[stat_name] = np.atleast_1d(np.asarray(ep_meta[col]))
            if ep_stats:
                self.episodes_stats[ep_idx] = ep_stats

    # ------------------------------------------------------------------
    # v3.0-specific loaders
    # ------------------------------------------------------------------

    def _load_episodes_v3(self) -> dict[int, dict]:
        """Load episode metadata from ``meta/episodes/**/*.parquet``."""
        ep_dir = self.root / "meta" / "episodes"
        if not ep_dir.is_dir():
            logger.warning("No v3.0 episode metadata directory: %s", ep_dir)
            return {}

        import pandas as pd
        files = sorted(ep_dir.rglob("*.parquet"))
        if not files:
            return {}

        dfs = [pd.read_parquet(f) for f in files]
        ep_df = pd.concat(dfs, axis=0, ignore_index=True)
        if "episode_index" in ep_df.columns:
            ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)

        return {int(row["episode_index"]): dict(row) for _, row in ep_df.iterrows()}

    def _load_tasks_v3(self) -> tuple[dict[int, str], dict[str, int]]:
        """Load tasks from ``meta/tasks.parquet``."""
        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return {}, {}

        import pandas as pd
        tasks_df = pd.read_parquet(tasks_path)
        tasks = {int(row["task_index"]): str(task_name) for task_name, row in tasks_df.iterrows()}
        task_to_index = {v: k for k, v in tasks.items()}
        return tasks, task_to_index

    # ------------------------------------------------------------------
    # Path resolution overrides for v3.0
    # ------------------------------------------------------------------

    def get_data_file_path(self, ep_index: int) -> Path:
        """Return the parquet file path for an episode.

        v2.x: uses ``{episode_chunk}`` / ``{episode_index}`` pattern.
        v3.0: uses ``{chunk_index}`` / ``{file_index}`` from episode metadata.
        """
        if self._version < _parse_version("v3.0"):
            return super().get_data_file_path(ep_index)

        meta = self.episodes.get(ep_index, {})
        ck = int(meta.get("data/chunk_index", ep_index // self.chunks_size))
        fi = int(meta.get("data/file_index", 0))
        return Path(f"data/chunk-{ck:03d}/file-{fi:03d}.parquet")

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Return the video file path for an episode and camera key.

        v2.x: uses ``{episode_chunk}`` / ``{video_key}`` / ``{episode_index}``.
        v3.0: uses ``{video_key}`` / ``{chunk_index}`` / ``{file_index}``.
        """
        if self._version < _parse_version("v3.0"):
            return super().get_video_file_path(ep_index, vid_key)

        meta = self.episodes.get(ep_index, {})
        ck = int(meta.get(f"videos/{vid_key}/chunk_index", ep_index // self.chunks_size))
        fi = int(meta.get(f"videos/{vid_key}/file_index", 0))
        return Path(f"videos/{vid_key}/chunk-{ck:03d}/file-{fi:03d}.mp4")

    def get_episode_chunk(self, ep_index: int) -> int:
        """Override: for v3.0, use episode metadata chunk_index."""
        if self._version >= _parse_version("v3.0"):
            meta = self.episodes.get(ep_index, {})
            if meta:
                return int(meta.get("data/chunk_index", ep_index // self.chunks_size))
        return super().get_episode_chunk(ep_index)

    # ------------------------------------------------------------------
    # Unsupported operations for v3.0
    # ------------------------------------------------------------------

    def add_task(self, task: str):
        if self._version >= _parse_version("v3.0"):
            raise NotImplementedError("add_task is not supported for v3.0 read-only datasets")
        super().add_task(task)

    def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats):
        if self._version >= _parse_version("v3.0"):
            raise NotImplementedError("save_episode is not supported for v3.0 read-only datasets")
        super().save_episode(episode_index, episode_length, episode_tasks, episode_stats)

    def pull_from_repo(self, *args, **kwargs):
        if self._version >= _parse_version("v3.0"):
            raise NotImplementedError("pull_from_repo is not supported for v3.0 (data is local)")
        super().pull_from_repo(*args, **kwargs)


# ============================================================================
# Main adapter class — inherits from LeRobotDataset
# ============================================================================

class LeRobotDatasetAdapter(lerobot_dataset.LeRobotDataset):
    """Adapter that extends ``LeRobotDataset`` to support v3.0 format.

    v3.0 differences:
    - Metadata in parquet (``meta/episodes/**/*.parquet``, ``meta/tasks.parquet``)
      instead of JSONL.
    - Multiple episodes per parquet file, sliced by
      ``dataset_from_index``/``dataset_to_index``.
    - Path patterns using ``{chunk_index}``, ``{file_index}``.
    - Multi-episode video files with ``from_timestamp`` frame offsets.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Any | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        # Detect version early from local root
        root_path = Path(root) if root else lerobot_dataset.HF_LEROBOT_HOME / repo_id
        try:
            version = detect_dataset_version(root_path)
        except LeRobotDatasetVersionError:
            version = "v2.1"

        if version in ("v2.0", "v2.1"):
            # Delegate to parent — works unchanged
            super().__init__(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                revision=revision,
                force_cache_sync=force_cache_sync,
                download_videos=download_videos,
                video_backend=video_backend,
            )
            return

        # ---- v3.0 path: bypass parent's metadata (which expects JSONL) ----
        super(lerobot_dataset.LeRobotDataset, self).__init__()

        self.repo_id = repo_id
        self.root = root_path
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else "v3.0"
        self.video_backend = video_backend or "torchcodec"
        self.delta_indices = None
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load v3.0 metadata using the metadata adapter
        self.meta = LeRobotDatasetMetadataAdapter(
            self.repo_id, root=str(self.root), revision=self.revision
        )

        # Load multi-episode parquet data
        self.hf_dataset = self._load_v3_hf_dataset()

        # Build episode data index
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.tensor(self.hf_dataset["timestamp"]).numpy()
        episode_indices_np = torch.tensor(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        from lerobot.common.datasets.utils import check_timestamps_sync
        check_timestamps_sync(timestamps, episode_indices_np, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            from lerobot.common.datasets.utils import check_delta_timestamps
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    # ------------------------------------------------------------------
    # Data loading overrides
    # ------------------------------------------------------------------

    def _load_v3_hf_dataset(self) -> datasets.Dataset:
        """Load v3.0 multi-episode parquet data into a HuggingFace Dataset."""
        ep_indices = self.episodes if self.episodes is not None else sorted(self.meta.episodes.keys())

        # Group episodes by parquet file
        chunk_files: dict[tuple[int, int], list[int]] = {}
        for ep_idx in ep_indices:
            meta = self.meta.episodes.get(ep_idx)
            if meta is None:
                continue
            key = (int(meta["data/chunk_index"]), int(meta["data/file_index"]))
            chunk_files.setdefault(key, []).append(ep_idx)

        import pandas as pd
        tables = []
        for (ck, fi), eps in sorted(chunk_files.items()):
            data_path = self.root / f"data/chunk-{ck:03d}/file-{fi:03d}.parquet"
            df = pd.read_parquet(data_path)
            for ep_idx in sorted(eps):
                meta = self.meta.episodes[ep_idx]
                from_idx = int(meta["dataset_from_index"])
                to_idx = int(meta["dataset_to_index"])
                ep_df = df.iloc[from_idx:to_idx].copy()
                ep_df["episode_index"] = int(ep_idx)
                tables.append(ep_df)

        if not tables:
            raise RuntimeError(f"No data loaded for episodes {ep_indices}")

        combined = pd.concat(tables, axis=0, ignore_index=True)
        hf_dataset = datasets.Dataset.from_pandas(combined)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def load_hf_dataset(self) -> datasets.Dataset:
        """Override: dispatch to v3.0 loader when needed."""
        if hasattr(self, "meta") and hasattr(self.meta, "_version"):
            if self.meta._version >= _parse_version("v3.0"):
                return self._load_v3_hf_dataset()
        return super().load_hf_dataset()

    def get_episodes_file_paths(self) -> list[Path]:
        """Override: use v3.0 path patterns for episode data files."""
        if hasattr(self.meta, "_version") and self.meta._version >= _parse_version("v3.0"):
            ep_indices = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
            fpaths = [self.root / self.meta.get_data_file_path(ep_idx) for ep_idx in ep_indices]
            if len(self.meta.video_keys) > 0:
                video_files = [
                    self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                    for vid_key in self.meta.video_keys
                    for ep_idx in ep_indices
                ]
                fpaths += video_files
            return fpaths
        return super().get_episodes_file_paths()

    # ------------------------------------------------------------------
    # Video query override for v3.0 multi-episode files
    # ------------------------------------------------------------------

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Override: adjust timestamps by ``from_timestamp`` for v3.0 multi-episode video files."""
        if not (hasattr(self.meta, "_version") and self.meta._version >= _parse_version("v3.0")):
            return super()._query_videos(query_timestamps, ep_idx)

        item = {}
        ep_meta = self.meta.episodes.get(ep_idx, {})

        for vid_key, query_ts in query_timestamps.items():
            ck = int(ep_meta.get(f"videos/{vid_key}/chunk_index", ep_idx // self.meta.chunks_size)) if ep_meta else ep_idx // self.meta.chunks_size
            fi = int(ep_meta.get(f"videos/{vid_key}/file_index", 0)) if ep_meta else 0

            video_path = self.root / f"videos/{vid_key}/chunk-{ck:03d}/file-{fi:03d}.mp4"

            from_ts = float(ep_meta.get(f"videos/{vid_key}/from_timestamp", 0.0)) if ep_meta else 0.0
            adjusted_ts = [ts + from_ts for ts in query_ts] if from_ts > 0 else query_ts

            frames = decode_video_frames(str(video_path), adjusted_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    # ------------------------------------------------------------------
    # __getitem__ override for v3.0 task resolution
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Override: resolve task from parquet-based tasks for v3.0."""
        item = super().__getitem__(idx)

        if hasattr(self.meta, "_version") and self.meta._version >= _parse_version("v3.0") and hasattr(self.meta, "tasks"):
            task_idx = int(item["task_index"].item())
            if task_idx in self.meta.tasks:
                item["task"] = self.meta.tasks[task_idx]

        return item
