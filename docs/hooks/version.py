"""MkDocs hook that fills in the SSQP version.

Replaces `{{ ssqp_version }}` in every page with the version in project() in
the top-level CMakeLists.txt, so release file names and tags in the guides
follow the version without being edited by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER = "{{ ssqp_version }}"

_version = None


def on_config(config):
    global _version
    cmakelists = Path(config.config_file_path).parent.parent / "CMakeLists.txt"
    match = re.search(r"project\(ssqp VERSION ([0-9.]+)", cmakelists.read_text())
    if match is None:
        raise ValueError(f"No project(ssqp VERSION ...) in {cmakelists}")
    _version = match.group(1)
    return config


def on_page_markdown(markdown, page, config, files):
    return markdown.replace(PLACEHOLDER, _version)
