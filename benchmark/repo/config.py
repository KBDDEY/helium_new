"""Editable runtime knobs for the experiment-group scheduler (no shared .env)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# `python -m repo` selects the run group here; both groups share the settings below.
# helium2: FIFO single-card control. helium_new2: adaptive experiment (+ optional tail).
RUN_FLAVOR: Literal["helium2", "helium_new2"] = "helium_new2"

# Change GPU list / count here (like newResearch DEVICES).
DEVICES: list[str] = [
    # "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
    # "cuda:4",
    # "cuda:5",
    # "cuda:6",  # occupied
    # "cuda:7",
]

# Initial task order: (task type, zero-based sample index in its dataset).
# Repeating an entry creates another independent task from the same sample.
TASK_QUEUE: list[tuple[str, int]] = [
    ("iterative", 0),
    ("debate", 0),
    ("iterative", 1),
    ("debate", 1),
    ("iterative", 2),
    ("parallel", 1),
    ("debate", 3),
]

# Optional tail expansion, independent of the experiment group's adaptive base policy.
ENABLE_TAIL_ACCELERATION = True

# Scheduler result file (plan + per-task k/time + SUM).
RESULT_PATH = Path(__file__).resolve().parent / "results" / "experiment_latest.txt"

HOST = "127.0.0.1"
BASE_PORT = 19100

# Concurrent Helium runtimes (one per in-flight task). Scheduler assigns like vLLM BASE_PORT.
# LMCache: yaml controller_url is the monitor port; controller process binds monitor-1.
HELIUM_SERVER_BASE_PORT = 20000
LMCACHE_MONITOR_BASE_PORT = 22001

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"

# Per-engine vLLM settings (aligned with newResearch ref scripts).
GPU_MEMORY_UTILIZATION = 0.5
MAX_MODEL_LEN = 8192
ENABLE_PREFIX_CACHING = True

# Provisional allocation caps; tune after isolated card-count measurements.
# The adaptive base policy reads these caps; the control group stays at 1.
K_MAX_TABLE: dict[str, int] = {
    "iterative": 1,
    "debate": 3,
    "parallel": 3,
}

# Initial P allocation for adaptive branches 2/3 and upper bound for tail expansion.
# This target never makes a task accumulate or wait for cards.
K_WISH_TABLE: dict[str, int] = {
    "parallel": 2,
}

# Generation defaults for compiled graphs / execute.
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Health-check wait when starting the pool.
POOL_READY_TIMEOUT_S = 600.0
POOL_POLL_INTERVAL_S = 2.0
