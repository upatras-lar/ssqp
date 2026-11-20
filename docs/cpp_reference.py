#!/usr/bin/env python3
"""Writes the C++ API reference pages from Doxygen's XML output.

Run from this directory, after Doxygen:

    python3 cpp_reference.py api/xml docs/cpp-reference/ssqp

Every namespace under ssqp gets a page, written in the HTML structure that
mkdocstrings uses for the Python API reference, so that both references look
the same and the C++ one is part of the site's navigation and search. The
output directory is replaced on every run.
"""

from __future__ import annotations

import html
import posixpath
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Implementation details and test helpers, left out of the reference
EXCLUDED_NAMESPACES = ("ssqp::pinocchio_wrapper", "ssqp::tests")

# Anchor names for operators, whose symbols can't go in an anchor
OPERATOR_NAMES = {
    "=": "assign", "==": "eq", "!=": "ne", "<": "lt", ">": "gt", "<=": "le", ">=": "ge",
    "+": "add", "-": "sub", "*": "mul", "/": "div", "+=": "iadd", "-=": "isub",
    "*=": "imul", "/=": "idiv", "<<": "shl", ">>": "shr", "[]": "index", "()": "call",
    "->": "arrow", "!": "not", "bool": "bool",
}

# Titles of Doxygen's special sections, shown as admonitions
SECTION_TITLES = {
    "note": "Note", "warning": "Warning", "attention": "Attention", "remark": "Remark",
    "see": "See also", "return": "Returns", "pre": "Precondition", "post": "Postcondition",
    "since": "Since", "deprecated": "Deprecated", "todo": "Todo",
}

SIGNATURE_WIDTH = 90


@dataclass
class Entry:
    """A namespace, class or member, with its place on the site"""

    kind: str  # namespace, class, function or attribute, as mkdocstrings names them
    name: str
    full_name: str
    page: str
    anchor: str
    element: ET.Element
    labels: list[str] = field(default_factory=list)
    children: list[Entry] = field(default_factory=list)
    namespaces: list[Entry] = field(default_factory=list)


def text_of(element: ET.Element | None) -> str:
    return "".join(element.itertext()) if element is not None else ""


def tidy(code: str) -> str:
    """C++ from Doxygen's spaced-out form, std::vector< VecX > to std::vector<VecX>"""
    code = re.sub(r"\s+", " ", code).strip()
    code = re.sub(r"<\s+", "<", code)
    code = re.sub(r"\s+>", ">", code)
    return re.sub(r"\s+,", ",", code)


def escape(text: str) -> str:
    """Plain text, safe from Markdown and HTML"""
    text = text.replace("\\", "\\\\")
    for char in "`*_[]":
        text = text.replace(char, "\\" + char)
    return text.replace("<", "&lt;")


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def code_span(code: str) -> str:
    code = re.sub(r"\s+", " ", code).strip()
    return f"`` {code} ``" if "`" in code else f"`{code}`"


def anchor_name(name: str) -> str:
    if name.startswith("~"):
        return name[1:] + "-destructor"
    if name.startswith("operator") and not re.match(r"operator\w", name):
        symbol = name[len("operator"):].strip()
        return "operator-" + OPERATOR_NAMES.get(symbol, "".join(f"{ord(c):x}" for c in symbol))
    return re.sub(r"[^\w-]", "_", name)


def location_key(element: ET.Element) -> tuple[str, int]:
    location = element.find("location")
    if location is None:
        return ("", 0)
    return (location.get("file", ""), int(location.get("line", "0")))


def page_path(namespace: str, namespaces: list[str]) -> str:
    """ssqp::model::terrain to model/terrain/index, for a namespace with nested ones, so that its
    navigation section opens its page, and ssqp::solvers to solvers otherwise"""
    parts = namespace.split("::")[1:]
    if any(other.startswith(namespace + "::") for other in namespaces):
        parts.append("index")
    return "/".join(parts)


class Reference:
    def __init__(self, xml_dir: Path):
        self.xml_dir = xml_dir
        self.targets: dict[str, tuple[str, str]] = {}
        index = ET.parse(xml_dir / "index.xml").getroot()
        names = {
            compound.get("refid"): compound.findtext("name")
            for compound in index.findall("compound")
            if compound.get("kind") == "namespace"
        }
        documented = sorted((name, refid) for refid, name in names.items() if self.documented(name))
        self.pages = []
        self.current_page = ""
        for name, refid in documented:
            entry = self.namespace(refid, name, page_path(name, [other for other, _ in documented]))
            if entry.children:
                self.pages.append(entry)
        by_name = {entry.name: entry for entry in self.pages}
        for entry in self.pages:
            parent = entry.name.rpartition("::")[0]
            if parent in by_name:
                by_name[parent].namespaces.append(entry)

    @staticmethod
    def documented(namespace: str) -> bool:
        if namespace != "ssqp" and not namespace.startswith("ssqp::"):
            return False
        return not any(
            namespace == excluded or namespace.startswith(excluded + "::")
            for excluded in EXCLUDED_NAMESPACES
        )

    def compound(self, refid: str) -> ET.Element:
        return ET.parse(self.xml_dir / f"{refid}.xml").getroot().find("compounddef")

    # The tree of entries, with their anchors

    def namespace(self, refid: str, name: str, page: str) -> Entry:
        compound = self.compound(refid)
        entry = Entry("namespace", name, name, page, name.replace("::", "."), compound)
        entry.children = self.contents(compound, entry)
        return entry

    def contents(self, compound: ET.Element, owner: Entry) -> list[Entry]:
        """Members and nested classes, grouped and ordered as mkdocstrings does"""
        members = []
        for section in compound.findall("sectiondef"):
            if section.get("kind") in ("friend", "define"):
                continue
            for member in section.findall("memberdef"):
                if member.get("prot") == "public" and not member.findtext("name", "").startswith("@"):
                    if member.get("kind") in ("function", "variable", "typedef", "enum"):
                        members.append(member)
        members.sort(key=location_key)
        classes = [
            self.compound(inner.get("refid"))
            for inner in compound.findall("innerclass")
            if inner.get("prot", "public") == "public"
        ]
        classes.sort(key=location_key)

        seen: dict[str, int] = {}

        def anchor(name: str) -> str:
            base = f"{owner.anchor}.{anchor_name(name)}"
            seen[base] = seen.get(base, 0) + 1
            return base if seen[base] == 1 else f"{base}-{seen[base]}"

        attributes = [self.member(m, owner, anchor) for m in members if m.get("kind") != "function"]
        nested = [self.cls(c, owner) for c in classes]
        functions = [self.member(m, owner, anchor) for m in members if m.get("kind") == "function"]
        return attributes + nested + functions

    def cls(self, compound: ET.Element, owner: Entry) -> Entry:
        full = compound.findtext("compoundname")
        name = full[len(owner.full_name) + 2:] if full.startswith(owner.full_name + "::") else full
        anchor = re.sub(r"[^\w.-]", "_", full.replace("::", "."))
        entry = Entry("class", name, full, owner.page, anchor, compound)
        self.targets[compound.get("id")] = (entry.page, entry.anchor)
        entry.children = self.contents(compound, entry)
        return entry

    def member(self, member: ET.Element, owner: Entry, anchor) -> Entry:
        kind = member.get("kind")
        name = member.findtext("name")
        entry = Entry(
            "function" if kind == "function" else "attribute", name, f"{owner.full_name}::{name}",
            owner.page, anchor(name), member,
        )
        if member.get("static") == "yes":
            entry.labels.append("static")
        if member.get("virt") == "virtual":
            entry.labels.append("virtual")
        if member.get("virt") == "pure-virtual":
            entry.labels.append("pure-virtual")
        if kind == "variable" and member.get("constexpr") == "yes":
            entry.labels.append("constexpr")
        if kind in ("typedef", "enum"):
            entry.labels.append(kind)
        self.targets[member.get("id")] = (entry.page, entry.anchor)
        return entry

    # Pages

    def write(self, out_dir: Path) -> None:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for entry in self.pages:
            self.current_page = entry.page
            text = f"# {escape(entry.name)}\n\n{self.render(entry, 2)}"
            path = out_dir / f"{entry.page}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n")

    def href(self, page: str, anchor: str = "") -> str:
        """Link to a page, relative to the page being written"""
        path = posixpath.relpath(f"{page}.md", posixpath.dirname(self.current_page) or ".")
        return f"{path}#{anchor}" if anchor else path

    def link(self, refid: str | None, label: str) -> str:
        if refid not in self.targets:
            return label
        return f"[{label}]({self.href(*self.targets[refid])})"

    def render(self, entry: Entry, level: int) -> str:
        css = "module" if entry.kind == "namespace" else entry.kind
        name = entry.name if entry.kind != "namespace" else entry.name.rpartition("::")[2]
        labels = "".join(
            f'<small class="doc doc-label doc-label-{label}"><code>{label}</code></small>'
            for label in entry.labels
        )
        labels = f' <span class="doc doc-labels">{labels}</span>' if labels else ""
        out = [
            f'<div class="doc doc-object doc-{css}" markdown>',
            f'{"#" * level} <span class="doc doc-object-name doc-{css}-name">{html.escape(name, quote=False)}</span>'
            f"{labels} {{ #{entry.anchor} .doc .doc-heading }}",
        ]
        signature = self.signature(entry)
        if signature:
            out.append(f"```cpp {{ .doc-signature }}\n{signature}\n```")
        first = " first" if entry.kind == "namespace" else ""
        out.append(f'<div class="doc doc-contents{first}" markdown>')
        out += self.description(entry)
        if entry.children:
            out.append('<div class="doc doc-children" markdown>')
            out += [self.render(child, min(level + 1, 6)) for child in entry.children]
            out.append("</div>")
        out += ["</div>", "</div>"]
        return "\n\n".join(out) + "\n"

    def description(self, entry: Entry) -> list[str]:
        element = entry.element
        out = []
        if entry.kind == "class":
            bases = [
                self.link(base.get("refid"), code_span(tidy(text_of(base))))
                for base in element.findall("basecompoundref")
                if base.get("prot") == "public"
            ]
            if bases:
                out.append(f'<p class="doc doc-class-bases" markdown="span">Bases: {", ".join(bases)}</p>')
            include = element.find("includes")
            if include is not None:
                header = text_of(include).strip()
                out.append(code_span(f'#include "{header}"' if include.get("local") == "yes" else f"#include <{header}>"))
        if entry.kind == "namespace" and entry.namespaces:
            links = [f"[{code_span(child.name.rpartition('::')[2])}]({self.href(child.page)})" for child in entry.namespaces]
            out.append("Namespaces: " + ", ".join(links))
        out += Description(self).blocks(element.find("briefdescription"))
        out += Description(self).blocks(element.find("detaileddescription"))
        if element.get("kind") == "enum":
            rows = ["| Value | Description |", "| --- | --- |"]
            for value in element.findall("enumvalue"):
                label = value.findtext("name") + (" " + tidy(text_of(value.find("initializer"))) if value.find("initializer") is not None else "")
                details = Description(self).blocks(value.find("briefdescription"))
                details += Description(self).blocks(value.find("detaileddescription"))
                rows.append(f"| {code_span(label)} | {' '.join(details).replace('|', chr(92) + '|')} |")
            out.append("\n".join(rows))
        return out

    # Signatures

    def signature(self, entry: Entry) -> str:
        element = entry.element
        kind = element.get("kind")
        template = self.template(element)
        if kind in ("class", "struct", "union"):
            return f"{template}\n{kind} {entry.name}" if template else ""
        if kind == "function":
            line = self.function(element, entry.name)
        elif kind == "variable":
            line = self.variable(element, entry.name)
        elif kind == "typedef":
            target = tidy(text_of(element.find("type")))
            if (element.findtext("definition") or "").startswith("using"):
                line = f"using {entry.name} = {re.sub(r'^typedef ', '', target)}"
            else:
                line = f"typedef {target} {entry.name}{tidy(element.findtext('argsstring') or '')}"
        elif kind == "enum":
            underlying = tidy(text_of(element.find("type")))
            line = f"enum {'class ' if element.get('strong') == 'yes' else ''}{entry.name}"
            line += f" : {underlying}" if underlying else ""
        else:
            return ""
        return f"{template}\n{line}" if template else line

    @staticmethod
    def template(element: ET.Element) -> str:
        params = element.find("templateparamlist")
        if params is None:
            return ""
        parts = []
        for param in params.findall("param"):
            text = tidy(text_of(param.find("type")))
            name = param.findtext("declname") or param.findtext("defname")
            if name and not text.endswith(name):
                text += f" {name}"
            if param.find("defval") is not None:
                text += f" = {tidy(text_of(param.find('defval')))}"
            parts.append(text)
        return f"template<{', '.join(parts)}>"

    @staticmethod
    def parameter(param: ET.Element) -> str:
        text = tidy(text_of(param.find("type")))
        name = param.findtext("declname")
        if name:
            text += ("" if text.endswith(("&", "*")) or not text else " ") + name
        text += tidy(param.findtext("array") or "")
        if param.find("defval") is not None:
            text += f" = {tidy(text_of(param.find('defval')))}"
        return text

    def function(self, element: ET.Element, name: str) -> str:
        qualifiers = []
        if element.get("explicit") == "yes":
            qualifiers.append("explicit")
        if element.get("static") == "yes":
            qualifiers.append("static")
        if element.get("virt") in ("virtual", "pure-virtual"):
            qualifiers.append("virtual")
        if element.get("constexpr") == "yes":
            qualifiers.append("constexpr")
        returns = tidy(text_of(element.find("type")))
        while True:
            stripped = re.sub(r"^(inline|static|virtual|explicit|constexpr)\s+", "", returns)
            if stripped == returns:
                break
            returns = stripped
        head = " ".join(qualifiers + ([returns] if returns else []) + [name])
        params = [self.parameter(param) for param in element.findall("param")]
        args = element.findtext("argsstring") or ""
        suffix = tidy(args[args.rfind(")") + 1:]) if ")" in args else ""
        suffix = re.sub(r"\s*=\s*", " = ", suffix).strip()
        suffix = f" {suffix}" if suffix else ""
        line = f"{head}({', '.join(params)}){suffix}"
        if len(line) <= SIGNATURE_WIDTH or not params:
            return line
        return f"{head}(\n" + ",\n".join(f"    {param}" for param in params) + f"\n){suffix}"

    @staticmethod
    def variable(element: ET.Element, name: str) -> str:
        qualifiers = []
        if element.get("static") == "yes":
            qualifiers.append("static")
        if element.get("constexpr") == "yes":
            qualifiers.append("constexpr")
        vtype = re.sub(r"^(static|constexpr)\s+", "", tidy(text_of(element.find("type"))))
        line = " ".join(qualifiers + [vtype, name]) + tidy(element.findtext("argsstring") or "")
        initializer = tidy(text_of(element.find("initializer")))
        if initializer and len(line) + len(initializer) + 1 <= SIGNATURE_WIDTH:
            line += initializer if initializer.startswith("{") else f" {initializer}"
        return line


class Description:
    """Markdown blocks from a Doxygen description"""

    def __init__(self, reference: Reference):
        self.reference = reference

    def blocks(self, element: ET.Element | None) -> list[str]:
        out = []
        if element is None:
            return out
        for child in element:
            if child.tag == "para":
                out += self.para(child)
            elif child.tag in ("sect1", "sect2", "sect3", "sect4"):
                title = re.sub(r"\s+", " ", text_of(child.find("title"))).strip()
                if title:
                    out.append(f"**{escape(title)}**")
                out += self.blocks(child)
        return out

    def para(self, para: ET.Element) -> list[str]:
        blocks: list[str] = []
        inline: list[str] = []

        def flush():
            text = re.sub(r"\s+", " ", "".join(inline)).strip()
            if re.match(r"([#>+-]|\d+\.)\s", text):
                text = "\\" + text
            if text:
                blocks.append(text)
            inline.clear()

        if para.text:
            inline.append(escape(para.text))
        for child in para:
            block = self.block(child)
            if block is None:
                inline.append(self.inline(child))
            else:
                flush()
                blocks.append(block)
            if child.tail:
                inline.append(escape(child.tail))
        flush()
        return blocks

    def inline_children(self, element: ET.Element) -> str:
        parts = [escape(element.text or "")]
        for child in element:
            parts.append(self.inline(child))
            parts.append(escape(child.tail or ""))
        return "".join(parts)

    def inline(self, element: ET.Element) -> str:
        tag = element.tag
        if tag == "formula":
            return "$" + text_of(element).strip().strip("$").strip() + "$"
        if tag == "computeroutput":
            return code_span(text_of(element))
        if tag == "bold":
            return f"**{self.inline_children(element)}**"
        if tag in ("emphasis", "italic"):
            return f"*{self.inline_children(element)}*"
        if tag == "ulink":
            return f"[{self.inline_children(element)}]({element.get('url')})"
        if tag == "ref":
            return self.reference.link(element.get("refid"), self.inline_children(element))
        if tag in ("linebreak", "sp"):
            return " "
        if tag == "ndash":
            return "–"
        if tag == "mdash":
            return "—"
        if tag == "anchor":
            return ""
        return self.inline_children(element)

    def block(self, element: ET.Element) -> str | None:
        """Markdown for a block-level element inside a paragraph, None for inline ones"""
        tag = element.tag
        if tag == "formula":
            formula = text_of(element).strip()
            if not formula.startswith(("\\[", "\\begin")):
                return None
            if formula.startswith("\\[") and formula.endswith("\\]"):
                formula = formula[2:-2].strip()
            lines = "\n".join(line.strip() for line in formula.split("\n"))
            return lines if lines.startswith("\\begin") else f"$$\n{lines}\n$$"
        if tag in ("itemizedlist", "orderedlist"):
            items = []
            for number, item in enumerate(element.findall("listitem"), 1):
                marker = "- " if tag == "itemizedlist" else f"{number}. "
                body = "\n\n".join(self.blocks(item))
                items.append(marker + indent(body).lstrip())
            loose = any("\n\n" in item for item in items)
            return ("\n\n" if loose else "\n").join(items)
        if tag == "simplesect":
            kind = element.get("kind")
            title = SECTION_TITLES.get(kind, kind.capitalize())
            kind_css = "warning" if kind in ("warning", "attention", "deprecated") else "note"
            heading = f"!!! {kind_css}" if title.lower() == kind_css else f'!!! {kind_css} "{title}"'
            return heading + "\n\n" + indent("\n\n".join(self.blocks(element)))
        if tag == "parameterlist":
            title = {"param": "Parameters", "retval": "Return values", "exception": "Exceptions"}
            rows = []
            for item in element.findall("parameteritem"):
                names = ", ".join(code_span(text_of(n)) for n in item.iter("parametername"))
                details = " ".join(self.blocks(item.find("parameterdescription")))
                rows.append(f"- {names}: {details}")
            return f"**{title.get(element.get('kind'), 'Parameters')}:**\n\n" + "\n".join(rows)
        if tag == "table":
            rows = []
            for row in element.findall("row"):
                cells = [" ".join(self.blocks(entry)).replace("|", "\\|") for entry in row.findall("entry")]
                rows.append("| " + " | ".join(cells) + " |")
            if not rows:
                return ""
            columns = rows[0].count(" | ") + 1
            return "\n".join([rows[0], "|" + " --- |" * columns] + rows[1:])
        if tag in ("programlisting", "verbatim", "preformatted"):
            if tag == "programlisting":
                lines = ["".join(self.code(line)) for line in element.findall("codeline")]
                code = "\n".join(lines)
            else:
                code = text_of(element).strip("\n")
            return f"```cpp\n{code}\n```"
        return None

    def code(self, element: ET.Element) -> list[str]:
        parts = [element.text or ""]
        for child in element:
            parts += [" "] if child.tag == "sp" else self.code(child)
            parts.append(child.tail or "")
        return parts


def main() -> None:
    xml_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    reference = Reference(xml_dir)
    reference.write(out_dir)
    print(f"Wrote {len(reference.pages)} pages to {out_dir}")


if __name__ == "__main__":
    main()
