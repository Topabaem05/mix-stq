from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mixstq"

OFFLINE = [
    "test_invariants.py",
    "test_imatrix_hook.py",
    "test_task_accuracy.py",
    "test_eval_tasks.py",
]
NETWORK = ["test_gguf_roundtrip.py", "test_c_parity.py", "test_vecdot.py"]


def run(name: str) -> tuple[str, int, str]:
    env_path = str(SRC) + ":" + str(ROOT / "tests")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tests" / name)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    tail = (completed.stdout.strip().splitlines() or [""])[-1]
    return name, completed.returncode, tail


def main() -> int:
    include_network = "--offline" not in sys.argv
    selected = OFFLINE + (NETWORK if include_network else [])
    failures = 0
    for name in selected:
        label, code, tail = run(name)
        status = "PASS" if code == 0 else "FAIL(%d)" % code
        if code != 0:
            failures += 1
        print("%-26s %-9s %s" % (label, status, tail[:70]))
    print()
    print("%d/%d passed" % (len(selected) - failures, len(selected)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
