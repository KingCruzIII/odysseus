#!/usr/bin/env python3
"""Bulk-import the Odysseus agent-tool reference (agent-tool-skills.md) as
real published skills, one per tool, via SkillsManager — no repeated
manage_skills calls, no copy/paste.

Usage (run from the repo root, with the venv active so `services.memory`
is importable):

    python3 scripts/import_tool_skills.py /path/to/agent-tool-skills.md

Writes directly under <repo>/data/skills/<category>/<tool-name>/SKILL.md.
Because docker-compose.yml bind-mounts ./data -> /app/data, a running
container picks these up immediately with no restart needed.
"""
from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from services.memory.skills import SkillsManager  # noqa: E402


_CATEGORY_SLUGS = {
    "Code & Filesystem": "code",
    "Documents (editor-panel, not disk files)": "documents",
    "Web & Research": "web",
    "Sessions / Chats / Multi-Agent": "sessions",
    "Knowledge": "knowledge",
    "Calendar, Notes & Contacts": "calendar",
    "Email": "email",
    "Tasks & Automation": "automation",
    "System / Admin": "system",
    "Media": "media",
    "Model Serving (Cookbook)": "cookbook",
}

_ENTRY_RE = re.compile(
    r"^### (?P<name>\S+)\n"
    r"\*\*When to use\*\* — (?P<when>.+)\n"
    r"\*\*How\*\* — (?P<how>.+)\n"
    r"\*\*Tags\*\* — (?P<tags>.+)$",
    re.MULTILINE,
)


def parse_doc(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    # Stop before the trailing cheat-sheet section — it isn't a tool entry.
    text = text.split("\n## Quick index by situation", 1)[0]

    entries = []
    current_category = "general"
    for block in re.split(r"\n(?=## )", text):
        header_match = re.match(r"^## (.+)$", block.splitlines()[0]) if block.startswith("## ") else None
        if header_match:
            current_category = _CATEGORY_SLUGS.get(header_match.group(1).strip(), "general")
        for m in _ENTRY_RE.finditer(block):
            entries.append({
                "name": m.group("name").strip(),
                "when_to_use": m.group("when").strip(),
                "how": m.group("how").strip(),
                "tags": [t.strip() for t in m.group("tags").split(",") if t.strip()],
                "category": current_category,
            })
    return entries


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/agent-tool-skills.md", file=sys.stderr)
        return 2
    doc_path = sys.argv[1]
    entries = parse_doc(doc_path)
    if not entries:
        print("error: parsed 0 entries — check the doc format/path", file=sys.stderr)
        return 1

    mgr = SkillsManager(os.path.join(REPO_ROOT, "data"))
    created, skipped = 0, 0
    for e in entries:
        result = mgr.add_skill(
            name=e["name"],
            description=e["when_to_use"],
            when_to_use=e["when_to_use"],
            procedure=[e["how"]],
            tags=e["tags"],
            category=e["category"],
            status="published",
            confidence=0.95,
            source="user",
        )
        if result.get("_deduped"):
            skipped += 1
            print(f"skip  (exists): {e['name']}")
        else:
            created += 1
            print(f"added         : {e['name']}  [{e['category']}]")

    print(f"\n{created} skills created, {skipped} skipped (already present) — {len(entries)} total parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
