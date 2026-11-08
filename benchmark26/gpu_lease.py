#!/usr/bin/env python3
"""Small process lease preventing overlapping local GPU timing runs."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import socket
import time
from pathlib import Path


LEASE_PATH = Path("/tmp/qre-benchmark-gpu-lease.lock")


class GpuLease:
    def __init__(self, handle, runner: str):
        self.handle = handle
        self.runner = runner
        self.released = False

    def release(self) -> None:
        """Release the lock on the lease file; safe to call more than once."""
        if self.released:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.released = True


def acquire_gpu_lease(runner: str) -> GpuLease:
    """Take the exclusive, non-blocking lock on the GPU lease file and record the owner.

    If another process holds the lock, exits with a message naming that owner. The lease is
    released at exit.
    """
    handle = LEASE_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        raise SystemExit(f"GPU lease is already held: {owner or 'owner metadata unavailable'}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {"pid": os.getpid(), "runner": runner, "hostname": socket.gethostname(), "acquired_unix": time.time()},
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    lease = GpuLease(handle, runner)
    atexit.register(lease.release)
    return lease
