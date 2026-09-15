"""Optional checks against a separately maintained current submission manuscript.

Numerical and renderer tests always run. Set RVPINN_CHECK_MANUSCRIPT=1 only
when the matching current manuscript sources and generated tables are present.
The experiments-only publication does not update the archived paper files.
"""
import os


def enabled():
    return os.environ.get("RVPINN_CHECK_MANUSCRIPT") == "1"
