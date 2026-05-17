from __future__ import annotations

import os


def cpu_count(default: int = 4) -> int:
    return max(int(os.cpu_count() or default), 1)


def default_chat_worker_count() -> int:
    cpu_total = cpu_count()
    # Chat workers are mostly blocked on network I/O to the LLM provider,
    # so a moderate oversubscription improves throughput without exploding
    # local CPU contention on a single small VM.
    return min(max(cpu_total * 4, 8), 20)


def default_chat_queue_max_size(worker_count: int) -> int:
    return max(int(worker_count or 0) * 24, 128)


def default_waitress_threads() -> int:
    cpu_total = cpu_count()
    return min(max(cpu_total * 10, 32), 80)


def default_waitress_connection_limit() -> int:
    return min(max(default_waitress_threads() * 32, 1280), 2560)


def default_report_worker_count() -> int:
    cpu_total = cpu_count()
    # Learning-report jobs are read/transform heavy but do not call the LLM,
    # so a moderate worker pool keeps snapshots fresh without overwhelming PG.
    return min(max(cpu_total * 2, 4), 12)


def default_pg_pool_max_size() -> int:
    cpu_total = cpu_count()
    return min(max(cpu_total * 10, 24), 80)


def default_pg_pool_min_size(max_size: int) -> int:
    return min(8, max(int(max_size or 1), 1))


def default_pg_pool_num_workers() -> int:
    cpu_total = cpu_count()
    return min(max(cpu_total, 3), 8)
