#!/usr/bin/env python3
"""Doxygen INPUT_FILTER: hides protected/private members from the docs.

Doxygen has no "EXTRACT_PROTECTED" setting -- a protected/private member is
emitted as soon as it has a doc comment, and (confirmed directly in Doxygen's
own generated XML) some protected members even get mislabeled `prot="public"`
outright (e.g. every field of an anonymous struct/union member, regardless of
the access specifier the declaration itself sits under). There is no config
option that reliably keeps these out of the generated reference.

Doxygen supports piping every input file through an external command before
parsing it (INPUT_FILTER, configured in mkdocs.yml) and parses whatever that
command prints to stdout -- the actual files on disk are never touched. This
filter finds every `protected:`/`private:` access-specifier section in a
class/struct body and wraps it in `@cond`/`@endcond`, Doxygen's own "skip
this entirely" directive, so those members never reach Doxygen's output in
the first place (rather than being generated and then filtered back out of
the rendered markdown). Detection is purely structural (brace depth), not
keyed to any specific class or member name, so this applies uniformly to
every class in the codebase, including ones added later.
"""

from __future__ import annotations

import re
import sys

_ACCESS_RE = re.compile(r"\b(public|protected|private)\s*:")


def _brace_depths(text: str) -> list[int]:
    """Depth *before* processing each character; one extra trailing entry."""
    depths = [0] * (len(text) + 1)
    depth = 0
    for i, ch in enumerate(text):
        depths[i] = depth
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    depths[len(text)] = depth
    return depths


def hide_non_public_sections(text: str) -> str:
    depths = _brace_depths(text)
    matches = list(_ACCESS_RE.finditer(text))

    wraps: list[tuple[int, int]] = []
    for idx, m in enumerate(matches):
        if m.group(1) == "public":
            continue

        own_depth = depths[m.start()]
        start = m.end()

        end = len(text)
        for k in range(start, len(text)):
            if depths[k] < own_depth:
                # depths[k] is the depth *before* processing text[k], so the
                # brace that closed the enclosing scope is text[k - 1] --
                # stop before it, or Doxygen's parser never "sees" that
                # closing brace (its own @cond handling skips parsing, not
                # just documentation, inside the guarded region) and loses
                # track of the class/struct boundary for the rest of the file.
                end = k - 1
                break

        for m2 in matches[idx + 1 :]:
            if m2.start() >= end:
                break
            if depths[m2.start()] == own_depth:
                end = m2.start()
                break

        wraps.append((start, end))

    out: list[str] = []
    pos = 0
    for start, end in wraps:
        if start < pos:
            continue  # nested inside a section we already wrapped
        out.append(text[pos:start])
        out.append(" /** @cond */ ")
        out.append(text[start:end])
        out.append(" /** @endcond */ ")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def main() -> None:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()
    sys.stdout.write(hide_non_public_sections(text))


if __name__ == "__main__":
    main()
