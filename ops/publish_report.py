#!/usr/bin/env python3
"""Publish (or re-publish) a Markdown report at a permanent link: /placeframe/reports/<id>/.

The id is random and unguessable (33 characters, like a Google Drive link): anyone with the link can read the
report and pin comments anywhere on it without signing in; nobody can find it without the link. Re-publishing the
same source keeps its id (and its comments) and updates its "updated" time.

  python ops/publish_report.py cost/COST_STUDY.md [--title "..."]      # prints the link
  python ops/publish_report.py --list
  python ops/publish_report.py cost/COST_STUDY.md --drive              # videos served from Google Drive
  python ops/publish_report.py --comments <id>                       # the comments, as Markdown, for review

Images and videos the report references by relative path are copied next to it. Reports live in ~/pf_data/reports (SSD).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path

REPORTS = Path.home() / "pf_data" / "reports"
REGISTRY = REPORTS / "registry.json"
BASE_URL = "https://28palomas-rtx3090.tail3fa327.ts.net/placeframe/reports"
# Files a report embeds: Markdown images and the src/poster of HTML tags (a <video> and its poster image).
ASSET = re.compile(r"!\[[^\]]*\]\(([^)\s]+)|(?:src|poster)=\"([^\"]+)\"")
# With --drive, each <video> is uploaded to Google Drive (rclone remote; scope drive.file: only files it uploads) and
# shared by link, and the published report shows Drive's player instead: large videos stream from Google, not over
# the home connection. The source Markdown is unchanged; a video is uploaded again only when it changes.
VIDEO = re.compile(r'<video\b[^>]*\bsrc="([^"]+)"[^>]*>(?:\s*</video>)?')
DRIVE_FOLDER = os.environ.get("REPORT_DRIVE_FOLDER", "gdrive:placeframe-shared/reports")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--comments", metavar="ID")
    ap.add_argument("--drive", action="store_true", help="serve the report's videos from Google Drive (rclone)")
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
    drive = dict((reg.get(rid) or {}).get("drive", {})) if a.drive else {}
    published = VIDEO.sub(lambda m: _drive_player(m, src, rid, drive), text) if a.drive else text
    for ref in (x or y for x, y in ASSET.findall(text)):
        if "://" in ref or ref.startswith("/") or ref in drive:
            continue
        # Contained by the path as written, not the resolved one: a large file kept out of git (a video) can be a
        # symlink next to the report; its target is copied.
        asset = Path(os.path.normpath(src.parent / ref))
        if asset.is_file() and asset.is_relative_to(src.parent):
            (out / ref).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, out / ref)
    (out / "report.md").write_text(published)
    reg[rid] = {"title": title, "source": str(src), "created": (reg.get(rid) or {}).get("created", now), "updated": now}
    if drive:
        reg[rid]["drive"] = drive
    REPORTS.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(REGISTRY)
    print(f"{BASE_URL}/{rid}/")


def _drive_player(m: re.Match, src: Path, rid: str, drive: dict) -> str:
    ref = m.group(1)
    asset = Path(os.path.normpath(src.parent / ref))
    if "://" in ref or not asset.is_file():
        return m.group(0)
    st = asset.stat()
    version = f"{st.st_size}:{int(st.st_mtime)}"
    if (drive.get(ref) or {}).get("version") != version:
        dest = f"{DRIVE_FOLDER}/{rid}/{Path(ref).name}"
        subprocess.run(["rclone", "copyto", str(asset.resolve()), dest], check=True)  # the file, not a symlink to it
        url = subprocess.run(["rclone", "link", dest], check=True, capture_output=True, text=True).stdout.strip()
        found = re.search(r"(?:/d/|[?&]id=)([\w-]+)", url)
        if not found:
            raise SystemExit(f"no Drive file id in {url!r}")
        drive[ref] = {"version": version, "id": found.group(1), "url": url}
        print(f"uploaded {ref} to Drive: {url}")
    return (f'<iframe src="https://drive.google.com/file/d/{drive[ref]["id"]}/preview" allow="autoplay; fullscreen" '
            'allowfullscreen style="width:100%;aspect-ratio:16/9;border:0;border-radius:6px"></iframe>')


def comments_markdown(rid: str, reg: dict) -> str:
    """Every comment thread with where it is in the report (section, the text under the pin), for review."""
    meta = reg.get(rid)
    if not meta:
        raise SystemExit(f"no report {rid}")
    lines = [f"# Comments on: {meta['title']}", f"Report: {BASE_URL}/{rid}/ (source: {meta['source']})", ""]
    path = REPORTS / rid / "comments-state.json"
    threads = json.loads(path.read_text())["comments"] if path.exists() else []
    for n, c in enumerate(threads, 1):
        ctx = c.get("context") or {}
        status = " (resolved)" if c.get("resolved") else ""
        lines += [f"## Thread {n}{status}" + (f" — section: {ctx['section']}" if ctx.get("section") else ""),
                  f"> {ctx.get('text') or '(no text under the pin)'}", ""]
        for r in [c, *c["replies"]]:
            edited = f", edited {r['editedAt']}" if r.get("editedAt") else ""
            lines.append(f"- **{r['author']}** ({r['createdAt']}{edited}): {r['text']}")
        lines.append("")
    if not threads:
        lines.append("No comments yet.")
    return "\n".join(lines)


def _first_heading(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


if __name__ == "__main__":
    main()
