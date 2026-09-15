"""Run the mandatory Stage-0 test suite and write the reproducibility
report section with full output."""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.reproducibility import environment_info         # noqa: E402


def main() -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v",
                        "--tb=short"], cwd=ROOT, capture_output=True,
                       text=True)
    env = environment_info()
    path = os.path.join(ROOT, "artifacts", "reproducibility_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Reproducibility Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Environment\n\n```\n")
        for k, v in env.items():
            f.write(f"{k}: {v}\n")
        f.write("```\n\n## Stage-0 mandatory test suite\n\n")
        f.write(f"Exit code: {r.returncode} "
                f"({'ALL PASSED' if r.returncode == 0 else 'FAILURES'})\n\n")
        f.write("```\n" + r.stdout[-20000:] + "\n```\n")
        if r.stderr.strip():
            f.write("\n### stderr\n```\n" + r.stderr[-5000:] + "\n```\n")
    print(f"tests exit={r.returncode}; report -> {path}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
