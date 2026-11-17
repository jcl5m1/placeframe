"""Project pages served next to the dashboard under /placeframe/ (the reverse proxy strips the prefix):

  /                     project home (public: links only)
  /visualizer/...       the map visualizer's static files (tailnet only)
  /vendor/...           vendored browser libraries (public)
  /reports/             list of reports (tailnet only)
  /reports/<id>/        one report, by its unguessable id (anyone with the link)
  /reports/<id>/comments            GET threads, POST a comment or reply (anyone with the link; no login)
  /reports/<id>/comments.json       every comment with what it refers to, for review tools and agents

Reports are published with ops/publish_report.py into ~/pf_data/reports/<id>/ (report.md and its images; SSD).
Comments are anchored to the text a reader selected (W3C Web Annotation: TextQuoteSelector with prefix/suffix and
TextPositionSelector) plus the heading of the section, and appended to <id>/comments.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import markdown
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIEWER_DIR = Path(os.environ.get("MAPPER_VIEWER_DIR", Path.home() / "pf_x5" / "viewer"))
REPORTS = Path(os.environ.get("MAPPER_REPORTS", Path.home() / "pf_data" / "reports"))
REGISTRY = REPORTS / "registry.json"
REPORT_ID = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
# Anyone with a report's link may comment: keep it bounded.
MAX_BODY, MAX_AUTHOR, MAX_TARGET = 4000, 60, 16_000
RATE_WINDOW_S, RATE_MAX = 600, 30
_SALT = secrets.token_hex(16)  # IPs are kept only as salted hashes (for abuse handling), new salt per process
_recent: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()

router = APIRouter()


def tailnet_only(request: Request, what: str) -> None:
    # Requests through the public Funnel carry this header; tailnet requests don't.
    if request.headers.get("Tailscale-Funnel-Request"):
        raise HTTPException(403, f"{what} is only available on the tailnet")


@router.get("/home")
async def home():
    return FileResponse(HERE / "static" / "home.html", headers={"Cache-Control": "no-cache"})


@router.get("/visualizer")
async def visualizer_redirect():
    return RedirectResponse("visualizer/", status_code=308)


@router.get("/visualizer/{name:path}")
async def visualizer(name: str, request: Request):
    tailnet_only(request, "the map visualizer")
    # Contained by the requested path (not the resolved one): the folder's own symlinks (pf_ui.js) are served.
    path = Path(os.path.normpath(VIEWER_DIR / (name or "index.html")))
    if path.is_dir():
        path = path / "index.html"
    if not path.is_relative_to(VIEWER_DIR) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@router.get("/vendor/{name}")
async def vendor(name: str):
    path = (REPO / "web" / "vendor" / name).resolve()
    if not path.is_relative_to((REPO / "web" / "vendor").resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@router.get("/reports")
async def reports_redirect():
    return RedirectResponse("reports/", status_code=308)


@router.get("/reports/")
async def reports_page(request: Request):
    tailnet_only(request, "the list of reports")
    return FileResponse(HERE / "static" / "reports.html", headers={"Cache-Control": "no-cache"})


@router.get("/reports/api/list")
async def reports_list(request: Request):
    tailnet_only(request, "the list of reports")
    reg = _registry()
    return [{"id": rid, **{k: v for k, v in r.items() if k != "source"}, "comments": _comment_count(rid)}
            for rid, r in sorted(reg.items(), key=lambda kv: kv[1].get("updated", ""), reverse=True)]


@router.get("/reports/{rid}")
async def report_redirect(rid: str):
    _check_id(rid)
    return RedirectResponse(f"{rid}/", status_code=308)


@router.get("/reports/{rid}/")
async def report_page(rid: str):
    _report(rid)
    return FileResponse(HERE / "static" / "report.html", headers={"Cache-Control": "no-cache"})


@router.get("/reports/{rid}/data.json")
async def report_data(rid: str):
    meta = _report(rid)
    text = (REPORTS / rid / "report.md").read_text()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html = md.convert(text)
    return JSONResponse({"id": rid, "title": meta["title"], "created": meta["created"], "updated": meta["updated"],
                         "html": html}, headers={"Cache-Control": "no-cache"})


@router.get("/reports/{rid}/comments")
async def comments(rid: str):
    _report(rid)
    return {"threads": _threads(rid)}


@router.post("/reports/{rid}/comments")
async def add_comment(rid: str, body: dict, request: Request):
    _report(rid)
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0].strip()
    ip_hash = hashlib.sha256((_SALT + ip).encode()).hexdigest()[:16]
    now = time.time()
    with _lock:
        q = _recent[ip_hash]
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_MAX:
            raise HTTPException(429, "too many comments; try again in a few minutes")
        q.append(now)
    text = str(body.get("body") or "").strip()
    author = str(body.get("author") or "").strip()[:MAX_AUTHOR] or "anonymous"
    if not text or len(text) > MAX_BODY:
        raise HTTPException(400, f"a comment needs 1-{MAX_BODY} characters")
    parent = body.get("parent")
    rec = {"type": "comment", "id": secrets.token_urlsafe(9), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "author": author, "body": text, "ip_hash": ip_hash}
    if parent:  # a reply to an existing thread
        if not any(t["thread"] == parent for t in _threads(rid)):
            raise HTTPException(404, "no such thread")
        rec["thread"] = parent
    else:  # a new thread on the selected text
        target = body.get("target")
        if not isinstance(target, dict) or len(json.dumps(target)) > MAX_TARGET:
            raise HTTPException(400, "a new comment needs the selected text (target)")
        rec["thread"] = str(target.get("id") or rec["id"])[-64:]
        rec["target"] = target
        rec["section"] = str(body.get("section") or "")[:200]
    with _lock, (REPORTS / rid / "comments.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return _public(rec)


@router.get("/reports/{rid}/comments.json")
async def comments_export(rid: str):
    meta = _report(rid)
    return {"report": {"id": rid, **{k: v for k, v in meta.items() if k != "source"}}, "threads": _threads(rid)}


@router.get("/reports/{rid}/{name:path}")
async def report_asset(rid: str, name: str):
    _report(rid)
    base = (REPORTS / rid).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.name in ("comments.jsonl", "report.md"):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


def _registry() -> dict:
    try:
        return json.loads(REGISTRY.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _check_id(rid: str) -> None:
    if not REPORT_ID.match(rid):
        raise HTTPException(404)


def _report(rid: str) -> dict:
    _check_id(rid)
    meta = _registry().get(rid)
    if not meta or not (REPORTS / rid / "report.md").is_file():
        raise HTTPException(404, "no such report")
    return meta


def _threads(rid: str) -> list[dict]:
    path = REPORTS / rid / "comments.jsonl"
    if not path.exists():
        return []
    threads: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") != "comment":
            continue
        t = threads.get(r["thread"])
        if t is None and "target" in r:
            t = threads[r["thread"]] = {"thread": r["thread"], "target": r["target"], "section": r.get("section", ""),
                                        "quote": _quote(r["target"]), "comments": []}
        if t is not None:
            t["comments"].append(_public(r))
    return list(threads.values())


def _comment_count(rid: str) -> int:
    return sum(len(t["comments"]) for t in _threads(rid))


def _public(rec: dict) -> dict:
    return {k: rec[k] for k in ("id", "thread", "created", "author", "body") if k in rec}


def _quote(target: dict) -> dict:
    """The quoted text of a W3C annotation (TextQuoteSelector), with its context."""
    targets = target.get("target") or []
    for t in targets if isinstance(targets, list) else [targets]:  # W3C: one target or a list of them
        sel = t.get("selector") or [] if isinstance(t, dict) else []
        for s in sel if isinstance(sel, list) else [sel]:
            if isinstance(s, dict) and s.get("type") == "TextQuoteSelector":
                return {"exact": s.get("exact", ""), "prefix": s.get("prefix", ""), "suffix": s.get("suffix", "")}
    return {"exact": "", "prefix": "", "suffix": ""}
