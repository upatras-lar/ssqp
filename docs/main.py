"""Macros for the documentation pages.

Defines `{{ ssqp_version }}` as the version in project() in the top-level
CMakeLists.txt, so release file names and tags in the guides follow the
version without being edited by hand.
"""

from __future__ import annotations

import re
from pathlib import Path


def define_env(env):
    cmakelists = Path(__file__).resolve().parent.parent / "CMakeLists.txt"
    match = re.search(r"project\(ssqp VERSION ([0-9.]+)", cmakelists.read_text())
    if match is None:
        raise ValueError(f"No project(ssqp VERSION ...) in {cmakelists}")
    env.variables["ssqp_version"] = match.group(1)
