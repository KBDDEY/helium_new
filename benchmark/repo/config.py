"""Editable runtime knobs for the experiment-group scheduler (no shared .env)."""

from __future__ import annotations

from pathlib import Path

# Change GPU list / count here (like newResearch DEVICES).
DEVICES: list[str] = [
    # "cuda:0",
    # "cuda:1",
    "cuda:2",
    "cuda:3",
    "cuda:4",
    # "cuda:5",
    # "cuda:6",  # occupied
    # "cuda:7",
]

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
# Both multi-card policies read this table; AllSingleCard stays at 1.
K_MAX_TABLE: dict[str, int] = {
    "iterative": 1,
    "debate": 3,
    "parallel": 3,
}

# Active waiting target for ParallelReserveStrategy only (not an allocation cap).
# Effective target is min(k_wait, k_max, actual total cards).
K_WAIT_TABLE: dict[str, int] = {
    "parallel": 2,
}

# Optional, experiment-group-only tail expansion. Existing launch policies are unchanged.
ENABLE_TAIL_ACCELERATION = False

# Upper bound for tail expansion; this never makes a task wait for cards.
# Keep the legacy waiting-policy table above independent and unchanged.
K_WISH_TABLE: dict[str, int] = dict(K_WAIT_TABLE)

# Generation defaults for compiled graphs / execute.
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Health-check wait when starting the pool.
POOL_READY_TIMEOUT_S = 600.0
POOL_POLL_INTERVAL_S = 2.0
