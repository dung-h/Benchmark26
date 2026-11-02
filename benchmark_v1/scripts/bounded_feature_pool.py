"""Bounded CPU feature extraction; callers alone persist completed results.

Workers and their initializer must be importable top-level functions because
the pool uses spawn. An ``exclusive`` task drains the pool and runs alone.
There is no task time limit: slow parsing is not scientific unavailability.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
import multiprocessing
import math
import os
from pathlib import Path
import time


class ResourceLimitError(RuntimeError):
    """Running preprocessing crossed the declared memory budget."""


def _resource_usage():
    """Return (MemAvailable, this process-tree RSS), in GiB, on Linux."""
    available_kib = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    if available_kib is None:
        raise RuntimeError("/proc/meminfo does not report MemAvailable")
    pending = [os.getpid()]
    seen = set()
    rss_kib = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
            pending.extend(
                int(child)
                for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            )
        except (FileNotFoundError, ProcessLookupError):
            pass  # A worker may exit between proc reads.
    return available_kib / 1024**2, rss_kib / 1024**2


def _abort_pool(executor):
    """Cancel queued work and reap parsers without waiting for their completion.

    Python 3.10 exposes no public terminate_workers API. Capture its process
    handles before shutdown clears them, then terminate and, if needed, kill.
    """
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    manager = getattr(executor, "_executor_manager_thread", None)
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2.0
    if manager is not None:
        # The executor manager also joins child processes. Let it finish first
        # to avoid competing waitpid calls from two parent threads.
        manager.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + 2.0
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))


def iter_feature_results(
    tasks,
    worker,
    *,
    workers=2,
    initializer=None,
    initargs=(),
    available_ram_reserve_gib=8.0,
    aggregate_rss_cap_gib=28.0,
    poll_seconds=0.25,
    resource_callback=None,
    task_memory_reservation_gib=0.0,
    idle_worker_reservation_gib=0.0,
    backfill=False,
):
    """Yield results from a memory-admitted pool of up to twelve CPU jobs.

    Low available RAM pauses admission when idle. A limit violation while a
    job is running raises ResourceLimitError and aborts outstanding work;
    callers retain results already yielded in their durable checkpoints.
    Closing this iterator also aborts outstanding jobs. Tasks are dictionaries
    with an optional truthy ``exclusive`` key. Initializers own thread caps.
    The parent invokes resource_callback(available_gib, rss_gib) on every poll,
    before enforcing limits, so its manifest can retain observed resource use.
    An optional per-task memory reservation limits admission before workers
    allocate memory. Existing observed RSS is not counted a second time.
    Tasks may override the default with memory_reservation_gib. Existing idle
    workers retain a separately reserved footprint; unused slots do not.
    With backfill enabled, a finite input sequence may supply smaller tasks
    when its next task does not fit. Exclusive tasks remain order barriers.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 12:
        raise ValueError("feature workers must be an integer from 1 to 12")
    settings = (available_ram_reserve_gib, aggregate_rss_cap_gib, poll_seconds, task_memory_reservation_gib, idle_worker_reservation_gib)
    if not all(math.isfinite(value) for value in settings) or available_ram_reserve_gib < 0 or aggregate_rss_cap_gib <= 0 or poll_seconds <= 0 or task_memory_reservation_gib < 0 or idle_worker_reservation_gib < 0:
        raise ValueError("memory limits and polling interval must be valid")
    if backfill and not isinstance(tasks, (list, tuple)):
        raise TypeError("backfill requires a finite task sequence")
    initial_usage = _resource_usage()
    baseline_rss = initial_usage[1]
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
    waiting = deque(tasks) if backfill else None
    iterator = iter(()) if backfill else iter(tasks)
    pending = {}
    held = None
    exhausted = False
    completed_normally = False
    try:
        while pending or held is not None or not exhausted:
            available, rss = initial_usage if initial_usage is not None else _resource_usage()
            initial_usage = None
            if resource_callback is not None:
                resource_callback(available, rss)
            if rss > aggregate_rss_cap_gib or (pending and available < available_ram_reserve_gib):
                raise ResourceLimitError(
                    f"feature pool memory budget exceeded: available={available:.3f} GiB "
                    f"(reserve={available_ram_reserve_gib}), RSS={rss:.3f} GiB "
                    f"(cap={aggregate_rss_cap_gib})"
                )
            may_admit = available >= available_ram_reserve_gib
            exclusive_running = any(task.get("exclusive", False) for task in pending.values())
            while may_admit and not exclusive_running and len(pending) < workers:
                if held is None:
                    if exhausted:
                        break
                    try:
                        held = waiting.popleft() if waiting else next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    if not isinstance(held, dict):
                        raise TypeError("feature task must be a dictionary")
                if held.get("exclusive", False) and pending:
                    break
                def fits(candidate):
                    leases = [float(task.get("memory_reservation_gib", task_memory_reservation_gib)) for task in [*pending.values(), candidate]]
                    if not all(math.isfinite(value) and value >= 0 for value in leases):
                        raise ValueError("task memory reservations must be finite and nonnegative")
                    existing_workers = len(getattr(executor, "_processes", None) or {})
                    idle_slots = max(0, existing_workers - len(leases))
                    projected_rss = max(rss, baseline_rss + sum(leases) + idle_slots * idle_worker_reservation_gib)
                    remaining_available = available - max(0.0, projected_rss - rss)
                    return projected_rss <= aggregate_rss_cap_gib and remaining_available >= available_ram_reserve_gib

                if not fits(held):
                    found = False
                    if backfill and not held.get("exclusive", False):
                        for index, candidate in enumerate(waiting):
                            if candidate.get("exclusive", False):
                                break
                            if fits(candidate):
                                del waiting[index]
                                waiting.appendleft(held)
                                held = candidate
                                found = True
                                break
                    if not found:
                        break
                task, held = held, None
                pending[executor.submit(worker, task)] = task
                if task.get("exclusive", False):
                    break
            if not pending:
                if exhausted and held is None:
                    break
                time.sleep(poll_seconds)
                continue
            done, _ = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
            # Surface any worker exception before admitting further work.
            results = [(future, future.result()) for future in done]
            for future, result in results:
                yield pending.pop(future), result
        completed_normally = True
    finally:
        if completed_normally:
            executor.shutdown(wait=True)
        else:
            _abort_pool(executor)
