"""Project pages served next to the dashboard under /placeframe/ (the reverse proxy strips the prefix):

  /                     project home (public: links only)
  /visualizer/...       the map visualizer's static files (tailnet only)
  /vendor/...           vendored browser libraries (public)
  /reports/             list of reports (tailnet only)
  /reports/<id>/        one report, by its unguessable id (anyone with the link)
  /reports/<id>/comments            GET the comments; PUT the page's comment list (anyone with the link; no login)
  /reports/<id>/comments.json       every comment with where it is in the report, for review tools and agents

Reports are published with ops/publish_report.py into ~/pf_data/reports/<id>/ (report.md and its images; SSD).

Comments are pins at any point of the report (pindrop.js in the page): each is anchored to the element under the pin
(a CSS selector and the position within it, as fractions of its size) and carries the section heading and the
element's text, so a reader of comments.json knows what it refers to. Threads have replies; comments are edited,
deleted and resolved in place. There are no accounts: each browser keeps a random participant key, and only the key
that wrote a comment may edit, move or delete it (anyone may resolve or reopen a thread). The page sends its whole
comment list on every change; the server applies only the changes that participant may make, keeps ids, timestamps
and names itself, and returns the stored list. State: <id>/comments-state.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import markdown
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIEWER_DIR = Path(os.environ.get("MAPPER_VIEWER_DIR", Path.home() / "pf_x5" / "viewer"))
REPORTS = Path(os.environ.get("MAPPER_REPORTS", Path.home() / "pf_data" / "reports"))
REGISTRY = REPORTS / "registry.json"
REPORT_ID = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
ITEM_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
PARTICIPANT = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
STATE = "comments-state.json"
# Anyone with a report's link may comment: keep it bounded.
MAX_BODY, MAX_AUTHOR, MAX_SELECTOR = 4000, 60, 1000
MAX_COMMENTS, MAX_REPLIES = 500, 200
RATE_WINDOW_S, RATE_MAX = 600, 60  # changes (new, edited, moved, deleted, resolved) per participant address
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
    return JSONResponse([_public(c) for c in load_comments(rid)], headers={"Cache-Control": "no-cache"})


@router.put("/reports/{rid}/comments")
async def save_comments(rid: str, body: dict, request: Request):
    _report(rid)
    participant = str(body.get("participant") or "")
    if not PARTICIPANT.match(participant):
        raise HTTPException(400, "a participant key is required")
    incoming = body.get("comments")
    if not isinstance(incoming, list) or len(incoming) > MAX_COMMENTS:
        raise HTTPException(400, f"comments must be a list of at most {MAX_COMMENTS}")
    who = _Who(owner=hashlib.sha256(participant.encode()).hexdigest()[:32],
               name=_clip(body.get("name"), MAX_AUTHOR) or "anonymous", ip_hash=_ip_hash(request), now=_now())
    with _lock:
        merged, changes = merge_comments(load_comments(rid), incoming, who)
        if changes:
            q = _recent[who.ip_hash]
            while q and time.time() - q[0] > RATE_WINDOW_S:
                q.popleft()
            if len(q) + changes > RATE_MAX:
                raise HTTPException(429, "too many changes; try again in a few minutes")
            q.extend([time.time()] * changes)
            tmp = REPORTS / rid / (STATE + ".tmp")
            tmp.write_text(json.dumps({"comments": merged}, indent=1))
            tmp.replace(REPORTS / rid / STATE)
    return [_public(c) for c in merged]


@router.get("/reports/{rid}/comments.json")
async def comments_export(rid: str):
    meta = _report(rid)
    return {"report": {"id": rid, **{k: v for k, v in meta.items() if k != "source"}},
            "comments": [export_comment(c) for c in load_comments(rid)]}


@router.get("/reports/{rid}/{name:path}")
async def report_asset(rid: str, name: str):
    _report(rid)
    base = (REPORTS / rid).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.name.startswith("comments") or path.name == "report.md":
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


class _Who:
    def __init__(self, owner: str, name: str, ip_hash: str, now: str) -> None:
        self.owner, self.name, self.ip_hash, self.now = owner, name, ip_hash, now


# The stored list with the changes this participant may make from the list their page sent, and how many changes.
# Each item goes to the page with "base", its stored values, which the page keeps as they are: a field changed on
# the page differs from its base, so a page holding an older copy changes only what it changed itself.
def merge_comments(stored: list[dict], incoming: list[Any], who: _Who) -> tuple[list[dict], int]:
    sent = {str(c.get("id")): c for c in incoming if isinstance(c, dict) and ITEM_ID.match(str(c.get("id", "")))}
    out: list[dict] = []
    changes = 0
    for old in stored:
        new = sent.pop(old["id"], None)
        mine = old["owner"] == who.owner
        if new is None:
            if mine:
                changes += 1  # deleted by its author (with its replies)
            else:
                out.append(old)
            continue
        c = dict(old)
        base = _base(new, old)
        text, anchor, resolved = _clip(new.get("text"), MAX_BODY), _anchor(new.get("anchor")), bool(new.get("resolved"))
        if mine and text and text != base["text"] and text != c["text"]:
            c["text"], c["editedAt"] = text, who.now
        if mine and anchor and anchor != base["anchor"] and anchor != c["anchor"]:
            c["anchor"] = anchor
        if resolved != bool(base["resolved"]) and resolved != c["resolved"]:
            c["resolved"], c["resolvedBy"], c["resolvedAt"] = resolved, who.name if resolved else None, who.now if resolved else None
        c["replies"], n = _merge_replies(old["replies"], new.get("replies"), who)
        if c != old or n:
            c["updatedAt"] = who.now
            changes += 1
        out.append(c)
    for new in sent.values():  # new comments
        text, anchor = _clip(new.get("text"), MAX_BODY), _anchor(new.get("anchor"))
        if not text or not anchor or len(out) >= MAX_COMMENTS:
            continue
        replies, _ = _merge_replies([], new.get("replies"), who)
        out.append({"id": str(new["id"]), "anchor": anchor, "context": _context(new.get("context")), "author": who.name,
                    "text": text, "createdAt": who.now, "updatedAt": who.now, "editedAt": None, "resolved": False,
                    "resolvedBy": None, "resolvedAt": None, "replies": replies, "owner": who.owner, "ip_hash": who.ip_hash})
        changes += 1
    renamed = False  # a participant who changes their name keeps their comments (and can still edit them)
    for item in out + [r for c in out for r in c["replies"]]:
        if item["owner"] == who.owner and item["author"] != who.name:
            item["author"], renamed = who.name, True
    return out, changes + renamed


def _merge_replies(stored: list[dict], incoming: Any, who: _Who) -> tuple[list[dict], int]:
    sent = {str(r.get("id")): r for r in (incoming if isinstance(incoming, list) else [])
            if isinstance(r, dict) and ITEM_ID.match(str(r.get("id", "")))}
    out: list[dict] = []
    changes = 0
    for old in stored:
        new = sent.pop(old["id"], None)
        mine = old["owner"] == who.owner
        if new is None and mine:
            changes += 1
            continue
        r = dict(old)
        text = _clip(new.get("text"), MAX_BODY) if new else ""
        if new and mine and text and text != _base(new, old)["text"] and text != r["text"]:
            r["text"], r["editedAt"], r["updatedAt"] = text, who.now, who.now
            changes += 1
        out.append(r)
    for new in sent.values():
        text = _clip(new.get("text"), MAX_BODY)
        if text and len(out) < MAX_REPLIES:
            out.append({"id": str(new["id"]), "author": who.name, "text": text, "createdAt": who.now,
                        "updatedAt": who.now, "editedAt": None, "owner": who.owner, "ip_hash": who.ip_hash})
            changes += 1
    return out, changes


def load_comments(rid: str) -> list[dict]:
    try:
        return json.loads((REPORTS / rid / STATE).read_text())["comments"]
    except (FileNotFoundError, ValueError, KeyError):
        return []


# A comment for review tools: what it says, who wrote it when, and where it is in the report.
def export_comment(c: dict) -> dict:
    ctx = c.get("context") or {}
    return {"id": c["id"], "author": c["author"], "text": c["text"], "created": c["createdAt"], "edited": c.get("editedAt"),
            "resolved": c["resolved"], "resolved_by": c.get("resolvedBy"),
            "location": {"section": ctx.get("section", ""), "element_text": ctx.get("text", ""),
                         "element": c["anchor"]["selector"],
                         "position_in_element": [round(c["anchor"]["offsetX"], 3), round(c["anchor"]["offsetY"], 3)]},
            "replies": [{"id": r["id"], "author": r["author"], "text": r["text"], "created": r["createdAt"],
                         "edited": r.get("editedAt")} for r in c["replies"]]}


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


def _comment_count(rid: str) -> int:
    return sum(1 + len(c["replies"]) for c in load_comments(rid))


# What the page sees: no owner keys or address hashes; "base" holds the stored values (see merge_comments).
def _public(c: dict) -> dict:
    out = {k: v for k, v in c.items() if k not in ("owner", "ip_hash")}
    out["base"] = {"text": c["text"], "anchor": c["anchor"], "resolved": c["resolved"]}
    out["replies"] = [{**{k: v for k, v in r.items() if k not in ("owner", "ip_hash")}, "base": {"text": r["text"]}}
                      for r in c["replies"]]
    return out


# The values the page started from: its "base", or the stored ones from a page that sent none.
def _base(new: dict, old: dict) -> dict:
    base = new.get("base")
    base = base if isinstance(base, dict) else {}
    return {k: base.get(k, old.get(k)) for k in ("text", "anchor", "resolved")}


def _anchor(a: Any) -> dict | None:
    if not isinstance(a, dict) or not isinstance(a.get("selector"), str) or not 0 < len(a["selector"]) <= MAX_SELECTOR:
        return None
    out: dict[str, Any] = {"selector": a["selector"]}
    for k in ("offsetX", "offsetY", "viewportX", "viewportY"):
        v = a.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or abs(v) > 1e6:
            return None
        out[k] = float(v)
    return out


def _context(ctx: Any) -> dict:
    ctx = ctx if isinstance(ctx, dict) else {}
    return {"section": _clip(ctx.get("section"), 200), "text": _clip(ctx.get("text"), 300)}


def _clip(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _ip_hash(request: Request) -> str:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0].strip()
    return hashlib.sha256((_SALT + ip).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
