#!/usr/bin/env python3
"""Publish (or re-publish) a Markdown report at a permanent link: /placeframe/reports/<id>/.

The id is random and unguessable (33 characters, like a Google Drive link): anyone with the link can read the
report and comment on any part of it without signing in; nobody can find it without the link. Re-publishing the
same source keeps its id (and its comments) and updates its "updated" time.

  python ops/publish_report.py cost/COST_STUDY.md [--title "..."]      # prints the link
  python ops/publish_report.py --list
  python ops/publish_report.py --comments <id>                       # the comments, as Markdown, for review

Images the report references by relative path are copied next to it. Reports live in ~/pf_data/reports (SSD).
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import time
from pathlib import Path

REPORTS = Path.home() / "pf_data" / "reports"
REGISTRY = REPORTS / "registry.json"
BASE_URL = "https://28palomas-rtx3090.tail3fa327.ts.net/placeframe/reports"
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--comments", metavar="ID")
    a = ap.parse_args()
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    if a.list:
        for rid, r in sorted(reg.items(), key=lambda kv: kv[1]["updated"]):
            print(f"{r['updated']}  {BASE_URL}/{rid}/  {r['title']}  ({r['source']})")
        return
    if a.comments:
        print(comments_markdown(a.comments, reg))
        return
    if a.source is None:
        ap.error("a Markdown source is required")
    src = a.source.resolve()
    text = src.read_text()
    rid = next((k for k, r in reg.items() if r["source"] == str(src)), None) or secrets.token_urlsafe(25)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    title = a.title or (reg.get(rid) or {}).get("title") or _first_heading(text) or src.stem
    out = REPORTS / rid
    out.mkdir(parents=True, exist_ok=True)
    for ref in IMAGE.findall(text):
        if "://" in ref or ref.startswith("/"):
            continue
        img = (src.parent / ref).resolve()
        if img.is_file() and img.is_relative_to(src.parent):
            (out / ref).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / ref)
    (out / "report.md").write_text(text)
    reg[rid] = {"title": title, "source": str(src), "created": (reg.get(rid) or {}).get("created", now), "updated": now}
    REPORTS.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(REGISTRY)
    print(f"{BASE_URL}/{rid}/")


def comments_markdown(rid: str, reg: dict) -> str:
    """Every comment thread with the text it refers to (and its section), for a person or an agent to review."""
    meta = reg.get(rid)
    if not meta:
        raise SystemExit(f"no report {rid}")
    lines = [f"# Comments on: {meta['title']}", f"Report: {BASE_URL}/{rid}/ (source: {meta['source']})", ""]
    threads: dict[str, dict] = {}
    path = REPORTS / rid / "comments.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        r = json.loads(line)
        if "target" in r:
            targets = r["target"].get("target") or []
            targets = targets if isinstance(targets, list) else [targets]  # W3C: one target or a list
            sel = [s for t in targets if isinstance(t, dict) for s in ([t.get("selector")] if isinstance(t.get("selector"), dict) else t.get("selector") or [])]
            quote = next((s for s in sel if isinstance(s, dict) and s.get("type") == "TextQuoteSelector"), {})
            threads[r["thread"]] = {"section": r.get("section", ""), "quote": quote, "comments": []}
        if r["thread"] in threads:
            threads[r["thread"]]["comments"].append(r)
    for n, t in enumerate(threads.values(), 1):
        q = t["quote"]
        lines += [f"## Thread {n}" + (f" — section: {t['section']}" if t["section"] else ""),
                  f"> …{q.get('prefix', '')}**{q.get('exact', '')}**{q.get('suffix', '')}…", ""]
        lines += [f"- **{c['author']}** ({c['created']}): {c['body']}" for c in t["comments"]] + [""]
    if not threads:
        lines.append("No comments yet.")
    return "\n".join(lines)


def _first_heading(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else None


if __name__ == "__main__":
    main()
