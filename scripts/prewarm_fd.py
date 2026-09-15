"""Precompute and cache the full FD reference solutions (CPU-only)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.exp_common import load_stack, regime_bundle
from scripts.run_fd_verification import fd_reference, REGIME

cfg = load_stack(os.path.join(ROOT, "configs", "paper_pilot.yaml"))
params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
fd_reference(cfg, params[REGIME], profs[REGIME])
print("FD prewarm complete")
