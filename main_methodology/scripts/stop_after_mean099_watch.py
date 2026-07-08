#!/usr/bin/env python3
"""Stop the running dual margin sweep after mean_margin tau=0.99.

This watcher leaves the current HotpotQA/2Wiki sweep running through
mean_margin tau=0.99, then terminates each dataset branch before max_margin
or the next dataset begins.
"""

from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/S113065528/Research/main_methodology")
MASTER = ROOT / "scripts" / "dual_margin_sweep_master.log"
WATCH_LOG = ROOT / "scripts" / "stop_after_mean099_watch.log"

BRANCHES = {
    "hotpotqa": 72120,
    "2wiki": 72121,
}


def ts() -> str:
    return dt.datetime.now().strftime("%F %T")


def log(message: str) -> None:
    line = f"[{ts()}] STOP-MEAN099 {message}\n"
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    with MASTER.open("a", encoding="utf-8") as f:
        f.write(line)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def descendants(pid: int) -> list[int]:
    try:
        output = subprocess.check_output(
            ["pgrep", "-P", str(pid)], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return []
    kids = [int(x) for x in output.split() if x.strip()]
    all_pids: list[int] = []
    for kid in kids:
        all_pids.append(kid)
        all_pids.extend(descendants(kid))
    return all_pids


def terminate_branch(pid: int) -> None:
    targets = descendants(pid) + [pid]
    for target in targets:
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for target in targets:
        if process_exists(target):
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass


def done_marker(dataset: str) -> str:
    return f"DONE  {dataset} signal=mean_margin tau=0.99"


def main() -> int:
    stopped: set[str] = set()
    log("armed: will stop HotpotQA/2Wiki branches after mean_margin tau=0.99 completes")
    while len(stopped) < len(BRANCHES):
        text = MASTER.read_text(encoding="utf-8", errors="replace")
        for dataset, branch_pid in BRANCHES.items():
            if dataset in stopped:
                continue
            if done_marker(dataset) in text:
                log(f"{dataset} tau=0.99 complete; terminating branch pid={branch_pid}")
                terminate_branch(branch_pid)
                stopped.add(dataset)
        if not process_exists(72081):
            log("main sweep parent ended; watcher exiting")
            return 0
        time.sleep(30)
    log("both requested datasets stopped after mean_margin tau=0.99")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
