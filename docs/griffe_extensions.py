"""Griffe extension used only by mkdocstrings to render docstrings.

The `ssqpy` docstrings (baked into the nanobind bindings and mirrored into the
generated `.pyi` stubs) use single newlines as intentional line breaks and
plain `- ` prefixed lines as lists, but never blank lines between paragraphs.
Standard Markdown collapses single newlines within a paragraph into a space,
so without this transform the docstrings render as one run-on paragraph.

This only touches text obtained through Griffe (i.e. rendered API reference
pages), not hand-authored Markdown pages, since those never go through this
extension.
"""

from __future__ import annotations

import re

import griffe

_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")


def _add_breaks(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        if _BULLET_RE.match(line):
            if out and out[-1].strip() != "":
                out.append("")
            while i < n and lines[i].strip() != "":
                out.append(lines[i])
                i += 1
            if i < n:
                out.append("")
            continue

        nxt = lines[i + 1] if i + 1 < n else None
        if (
            stripped != ""
            and nxt is not None
            and nxt.strip() != ""
            and not _BULLET_RE.match(nxt)
            and not line.rstrip().endswith("  ")
        ):
            out.append(line.rstrip() + "  ")
        else:
            out.append(line)
        i += 1

    return "\n".join(out)


class DocstringLineBreaks(griffe.Extension):
    def on_object(self, *, obj, loader, **kwargs) -> None:  # noqa: ANN001, ARG002
        if obj.docstring is None:
            return
        obj.docstring.value = _add_breaks(obj.docstring.value)
