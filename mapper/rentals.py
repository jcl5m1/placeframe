"""Rented GPUs for the dashboard: what is running on Vast.ai now, what it costs, and the cost of recent rentals.

  * live: `vastai show instances` (every instance on the account, whoever started it: the fleet broker, a benchmark,
    or by hand), with its price per hour and the cost so far at that price;
  * account: credit left and Vast's total spend (`vastai show user`);
  * recent: the fleet broker's own record of every instance it rented (~/pf_fleet/broker/dispatches.json): GPU, where,
    bid, how long it ran, and an estimated cost (bid × running time + the host's download charge).
Results are cached for a minute: the Vast API is slow and the dashboard polls every few seconds.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

DISPATCHES = Path.home() / "pf_fleet" / "broker" / "dispatches.json"
CACHE_S = 60
RECENT = 12
_lock = threading.Lock()
_cache: dict = {"t": 0.0, "data": None}


def snapshot() -> dict:
    with _lock:
        if _cache["data"] is None or time.time() - _cache["t"] > CACHE_S:
            _cache["data"], _cache["t"] = _collect(), time.time()
        return _cache["data"]


def _collect() -> dict:
    now = time.time()
    out: dict = {"t": now, "error": None, "running": [], "account": None}
    try:
        for i in _vast("show instances --raw") or []:
            start = i.get("start_date") or now
            dph = float(i.get("dph_total") or 0)
            out["running"].append({
                "id": i.get("id"), "gpu": f"{i.get('num_gpus') or 1}× {i.get('gpu_name')}", "where": i.get("geolocation"),
                "status": i.get("actual_status") or i.get("cur_state"), "label": i.get("label") or "",
                "dph": round(dph, 3), "since": start, "cost": round(dph * max(0.0, now - start) / 3600, 2)})
        user = _vast("show user --raw") or {}
        spend = user.get("total_spend")  # Vast reports spend as a negative balance change
        out["account"] = {"credit": user.get("credit"), "total_spend": abs(spend) if spend is not None else None}
    except Exception as e:  # no CLI, no key or the API is down: say so, keep the broker's history
        out["error"] = str(e)[:200]
    out["burn_per_hour"] = round(sum(r["dph"] for r in out["running"]), 3)
    out["recent"], out["totals"] = _history(now)
    return out


def _vast(args: str):
    exe = shutil.which("vastai") or str(Path.home() / ".local/bin/vastai")
    r = subprocess.run([exe, *args.split()], capture_output=True, text=True, timeout=30, check=False)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"vastai exit {r.returncode}")
    return json.loads(r.stdout or "null")


def _history(now: float) -> tuple[list[dict], dict]:
    try:
        jobs = json.loads(DISPATCHES.read_text())
    except (FileNotFoundError, ValueError):
        return [], {}
    rows = []
    for job, d in jobs.items():
        for inst in d.get("instances", []):
            start = inst.get("running_since") or inst.get("launched")
            end = inst.get("ended") or (now if d.get("status") not in ("done", "abandoned", "failed") else start)
            hours = max(0.0, (end or start) - start) / 3600 if start else 0.0
            rows.append({"job": job[:8], "job_status": d.get("status"), "id": inst.get("id"), "gpu": inst.get("gpu"),
                         "where": (inst.get("geo") or "").strip(", "), "bid": inst.get("bid"), "launched": inst.get("launched"),
                         "ended": inst.get("ended"), "hours": round(hours, 2),
                         "cost": round((inst.get("bid") or 0) * hours + (inst.get("download_cost") or 0), 2)})
    rows.sort(key=lambda r: r["launched"] or 0, reverse=True)
    day, week = now - 86400, now - 7 * 86400
    totals = {"day": round(sum(r["cost"] for r in rows if (r["launched"] or 0) >= day), 2),
              "week": round(sum(r["cost"] for r in rows if (r["launched"] or 0) >= week), 2),
              "all": round(sum(r["cost"] for r in rows), 2), "instances": len(rows)}
    return rows[:RECENT], totals
