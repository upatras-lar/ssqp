"""MkDocs hook that patches up known mkdoxy markdown-generation bugs.

mkdoxy's `Text` renderer (mkdoxy/markdown.py) unconditionally backslash-escapes
underscores for every plain-text node. Doxygen doesn't always recognize a
`$...$` span as a `<formula>` XML element (e.g. when it contains commas or
several LaTeX macros); when it doesn't, the span falls through as plain text
and gets escape()'d, corrupting `\\ell_k` into `\\ell\\_k` (which breaks
MathJax). Separately, mkdoxy's `MdList` renderer never emits a blank line
before itself, so a `* item` list glued directly onto the preceding paragraph
is never recognized as a real Markdown list -- it renders as literal `*`
characters in one run-on paragraph (the same root cause as the docstring
line-break issue fixed for the Python reference, via griffe_extensions.py).

A third, separate bug lives in Doxygen itself, not mkdoxy: for any identifier
it can't resolve to a real definition (e.g. `uint`, a typedef that comes from
a system header outside our INPUT scope, or bare keywords like `const`),
Doxygen's XML emits a `<ref>` pointing at an essentially-arbitrary unrelated
symbol instead of leaving it unlinked -- e.g. `uint` and `const` both end up
hyperlinked to an unrelated `ManifoldOperators` struct. Since this can't be
fixed at the source (we don't control system headers), the fix is the same
shape as the other two: strip the link (keeping the bold text) whenever the
link text is a bare C++ keyword/primitive type that should never legitimately
point anywhere.

A fourth issue isn't a bug so much as a missing Doxygen feature: there is no
"EXTRACT_PROTECTED" setting, so a protected/private member is emitted as soon
as it has a doc comment, regardless of EXTRACT_PRIVATE/EXTRACT_PROTECTED-style
options (which don't exist for the former either at the class-member level in
the way one might expect) -- and some protected members even get mislabeled
`prot="public"` outright in Doxygen's own XML (e.g. every field of an
anonymous struct/union member, regardless of the access specifier the
declaration actually sits under). The primary fix for this now lives
upstream, in `../doxygen_filter.py` (a Doxygen INPUT_FILTER that wraps every
`protected:`/`private:` section in `@cond`/`@endcond` before Doxygen ever
parses it, so those members never reach its output at all -- which also
covers the mislabeling case, since a member Doxygen never sees can't be
mislabeled). What's left here is a secondary safety net: strip any
"Protected"/"Private" section heading that still slips through (both the
summary tables and the full "... Documentation" sections), in case some
future edge case the filter doesn't structurally catch reintroduces one.

A fifth issue breaks the right-hand "on this page" sidebar entirely: mkdoxy's
class documentation always starts with its own `# {Kind} {name}` H1, e.g.
`# Class ssqp::solvers::MPC`. Embedded on a page that already has its own H1
title, that produces two *sibling* H1 headings instead of one nested under
the other. mkdocs-material's TOC template (partials/toc.html) assumes there's
exactly one H1 (the page title) and renders `page.toc[0].children` -- i.e. it
always drops the very first heading, assuming it's just the title, and shows
whatever is nested under it. When the title has no children (because the
"real" content is a *sibling* H1, not nested beneath it), that children list
is empty and the entire sidebar silently disappears. Confirmed via mkdocs'
own Page.render(): `page.toc` is correct immediately after conversion, so the
bug is purely in how mkdocs-material's template consumes it, not in the data.

The fix is to demote every heading inside a doxy-generated block by enough
levels that it nests properly under whatever heading precedes it in the
curated page. Since that varies per placement (a directive sitting right
under the page's H1 needs a lone +1; one sitting under one of this project's
own `## `/`### ` subheadings needs +2/+3), the exact amount can't be inferred
from content alone -- so each `::: doxy...` directive in `docs/cpp-reference/*.md`
is wrapped with an explicit `<!-- doxy:start level=N --> ... <!-- doxy:end -->`
marker pair (N = the level of the heading immediately preceding it). mkdoxy
only replaces the `::: doxy...` line itself, so these markers survive
expansion untouched and give this hook an unambiguous block boundary to
demote, regardless of the levels doxy happens to use inside.

mkdoxy expands `::: doxy...` snippet directives in place, on whatever page
references them, rather than confining generated content to its own pages --
so all fixes except heading demotion (which is explicitly marker-scoped) are
applied unconditionally, to every page's markdown. This is safe: none of
these transforms do anything unless they find the exact broken pattern (a
literal `\_`/`\*` inside a `$...$`/`\(...\)`/`\[...\]` span, a `* item` line
glued onto a non-blank previous line, a bare-keyword link, or a
`## Protected`/`## Private` heading), and no hand-authored page in this
project contains any of those patterns.
"""

from __future__ import annotations

import re

_MATH_SPAN_RE = re.compile(r"(\${1,2}.+?\${1,2}|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
_BULLET_RE = re.compile(r"^(\s*)\*\s+")

_CPP_KEYWORDS = [
    "const", "constexpr", "static", "virtual", "inline", "mutable", "volatile",
    "noexcept", "override", "final", "nullptr", "auto", "void", "bool", "char",
    "int", "long", "short", "unsigned", "signed", "float", "double", "uint",
    "size_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "true", "false",
]
_BARE_KEYWORD_LINK_RE = re.compile(
    r"\[\*\*(" + "|".join(_CPP_KEYWORDS) + r")\*\*\]\([^)]*\)"
)

# Doxygen has no "hide protected members" setting (EXTRACT_PROTECTED does not
# exist) -- protected/private members are always emitted once documented, so
# any protected/private helper with a doc comment reappears as public-looking
# noise. Strip those sections outright; the curated pages only ever wanted
# the public API surface.
_ACCESS_SECTION_RE = re.compile(
    r"^##\s+(?:Protected|Private)\s+.*?(?=^#{1,6}\s|\Z)", re.M | re.S
)

_DOXY_MARKER_RE = re.compile(
    r"<!--\s*doxy:start\s+level=(\d+)\s*-->(.*?)<!--\s*doxy:end\s*-->", re.S
)
_HEADING_LINE_RE = re.compile(r"^(#{1,6})(\s)", re.M)


def _unescape_math(markdown: str) -> str:
    def repl(m: re.Match) -> str:
        return m.group(0).replace(r"\_", "_").replace(r"\*", "*")

    return _MATH_SPAN_RE.sub(repl, markdown)


def _fix_bullet_lists(markdown: str) -> str:
    lines = markdown.split("\n")
    out: list[str] = []
    n = len(lines)
    for i, line in enumerate(lines):
        is_bullet = bool(_BULLET_RE.match(line))
        if is_bullet and out and out[-1].strip() != "":
            out.append("")
        out.append(line)
        nxt = lines[i + 1] if i + 1 < n else None
        if (
            is_bullet
            and nxt is not None
            and nxt.strip() != ""
            and not _BULLET_RE.match(nxt)
            and not nxt[:1].isspace()
        ):
            out.append("")
    return "\n".join(out)


def _unlink_bare_keywords(markdown: str) -> str:
    return _BARE_KEYWORD_LINK_RE.sub(r"**\1**", markdown)


def _strip_non_public_sections(markdown: str) -> str:
    return _ACCESS_SECTION_RE.sub("", markdown)


def _demote_doxy_headings(markdown: str) -> str:
    def repl(m: re.Match) -> str:
        level = int(m.group(1))
        block = m.group(2)
        return _HEADING_LINE_RE.sub(lambda h: ("#" * level) + h.group(1) + h.group(2), block)

    return _DOXY_MARKER_RE.sub(repl, markdown)


def on_page_markdown(markdown, page, config, files):
    markdown = _unescape_math(markdown)
    markdown = _fix_bullet_lists(markdown)
    markdown = _unlink_bare_keywords(markdown)
    markdown = _strip_non_public_sections(markdown)
    markdown = _demote_doxy_headings(markdown)
    return markdown
