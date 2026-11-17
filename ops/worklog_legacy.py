#!/usr/bin/env python3
"""Shared worklog + mailbox so several agent sessions (and their subagents) can work on the same
task list without colliding. Stdlib only; state is one SQLite file on the SSD.

    python3 ops/worklog.py status                       # workers, open tasks, leases, your unread count
    python3 ops/worklog.py register NAME --session S --desc "what I do"
    python3 ops/worklog.py add "title" --desc ... --area reconstructor --files a.py,b.py [--prio 1]
    python3 ops/worklog.py claim ID --as NAME           # atomic: fails if someone else holds it
    python3 ops/worklog.py note ID "progress text" --as NAME
    python3 ops/worklog.py done ID "result / commit" --as NAME     (also: release, block, reopen)
    python3 ops/worklog.py lease gpu --as NAME --ttl 3600 --why "throughput benchmark"
    python3 ops/worklog.py unlease gpu --as NAME
    python3 ops/worklog.py send TO|all "message" --as NAME [--task ID]
    python3 ops/worklog.py inbox --as NAME [--peek]     # prints unread, marks read
    python3 ops/worklog.py wait --as NAME [--timeout S] # blocks until a message arrives (run in background)
    python3 ops/worklog.py show ID | list [--all] | log [-n 30]

NAME defaults to $WORKLOG_WORKER. DB: $WORKLOG_DB or ~/pf_worklog/worklog.db. Protocol: ops/WORKLOG.md.
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime

DB = os.environ.get("WORKLOG_DB", os.path.expanduser("~/pf_worklog/worklog.db"))
STALE_S = 2 * 3600  # a worker not seen for this long is shown as stale; its claims may be taken over

SCHEMA = """
create table if not exists workers(name text primary key, session text, descr text,
  registered real, last_seen real);
create table if not exists tasks(id integer primary key, title text not null, descr text default '',
  area text default '', files text default '', prio integer default 2, status text default 'open',
  owner text, created_by text, created real, updated real, result text default '');
create table if not exists notes(id integer primary key, task integer, author text, ts real, text text);
create table if not exists messages(id integer primary key, sender text, recipient text, task integer,
  ts real, text text, read_ts real);
create table if not exists leases(resource text primary key, owner text, why text, since real, expires real);
create table if not exists events(id integer primary key, ts real, who text, what text);
"""
STATUSES = ("open", "claimed", "blocked", "done", "dropped")


def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal")
    c.executescript(SCHEMA)
    return c


class Tx:
    """BEGIN IMMEDIATE: one writer at a time, so check-then-write (claims, leases) is atomic."""

    def __init__(self, c):
        self.c = c

    def __enter__(self):
        self.c.execute("begin immediate")
        return self.c

    def __exit__(self, et, *_):
        self.c.execute("rollback" if et else "commit")


def ts(t):
    return datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else "-"


def die(msg):
    print(f"worklog: {msg}", file=sys.stderr)
    sys.exit(1)


def me(a):
    name = getattr(a, "as_", None) or os.environ.get("WORKLOG_WORKER")
    if not name:
        die("who are you? pass --as NAME or set WORKLOG_WORKER")
    return name


def touch(c, name, what=None):
    now = time.time()
    c.execute("insert into workers(name, registered, last_seen) values(?,?,?) "
              "on conflict(name) do update set last_seen=excluded.last_seen", (name, now, now))
    if what:
        c.execute("insert into events(ts, who, what) values(?,?,?)", (now, name, what))


def task(c, tid):
    r = c.execute("select * from tasks where id=?", (tid,)).fetchone()
    if not r:
        die(f"no task #{tid}")
    return r


def alive(c, name):
    r = c.execute("select last_seen from workers where name=?", (name,)).fetchone()
    return bool(r and time.time() - r["last_seen"] < STALE_S)


def notify(c, sender, recipient, text, tid=None):
    c.execute("insert into messages(sender, recipient, task, ts, text) values(?,?,?,?,?)",
              (sender, recipient, tid, time.time(), text))


# ---------------------------------------------------------------- commands
def cmd_register(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n, "registered")
        c.execute("update workers set session=coalesce(?, session), descr=coalesce(?, descr) where name=?",
                  (a.session, a.desc, n))
    print(f"registered {n}")


def cmd_add(a, c):
    n = me(a)
    now = time.time()
    with Tx(c):
        touch(c, n)
        cur = c.execute("insert into tasks(title, descr, area, files, prio, created_by, created, updated) "
                        "values(?,?,?,?,?,?,?,?)", (a.title, a.desc or "", a.area or "", a.files or "",
                                                     a.prio, n, now, now))
        tid = cur.lastrowid
        c.execute("insert into events(ts, who, what) values(?,?,?)", (now, n, f"added #{tid} {a.title}"))
        if a.owner:
            c.execute("update tasks set status='claimed', owner=? where id=?", (a.owner, tid))
    print(f"#{tid}")


def overlapping(c, t):
    """Other claimed tasks touching the same files/area: a collision warning, not a block."""
    files = {f.strip() for f in (t["files"] or "").split(",") if f.strip()}
    out = []
    for o in c.execute("select * from tasks where status='claimed' and id!=?", (t["id"],)):
        of = {f.strip() for f in (o["files"] or "").split(",") if f.strip()}
        if files & of:
            out.append(f"#{o['id']} ({o['owner']}) shares files {', '.join(sorted(files & of))}")
    return out


def cmd_claim(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n)
        t = task(c, a.id)
        if t["status"] in ("done", "dropped"):
            die(f"#{a.id} is {t['status']}")
        if t["owner"] and t["owner"] != n:
            if alive(c, t["owner"]) and not a.force:
                die(f"#{a.id} is held by {t['owner']} (active). Message them, or --force if they agreed.")
            notify(c, n, t["owner"], f"I took over #{a.id} '{t['title']}' from you "
                   f"({'forced' if a.force else 'you were stale'}).", a.id)
        c.execute("update tasks set status='claimed', owner=?, updated=? where id=?", (n, time.time(), a.id))
        c.execute("insert into events(ts, who, what) values(?,?,?)", (time.time(), n, f"claimed #{a.id}"))
        warn = overlapping(c, t)
    print(f"claimed #{a.id} {t['title']}")
    for w in warn:
        print(f"  warning: {w}")


def set_status(a, c, status, verb):
    n = me(a)
    with Tx(c):
        touch(c, n)
        t = task(c, a.id)
        if t["owner"] and t["owner"] != n and not a.force:
            die(f"#{a.id} is held by {t['owner']}; --force to override")
        owner = None if status in ("open",) else (t["owner"] or n)
        c.execute("update tasks set status=?, owner=?, updated=?, result=case when ?!='' then ? else result end "
                  "where id=?", (status, owner, time.time(), a.text or "", a.text or "", a.id))
        if a.text:
            c.execute("insert into notes(task, author, ts, text) values(?,?,?,?)",
                      (a.id, n, time.time(), f"[{verb}] {a.text}"))
        c.execute("insert into events(ts, who, what) values(?,?,?)", (time.time(), n, f"{verb} #{a.id}"))
        if t["created_by"] and t["created_by"] not in (n, None) and status in ("done", "blocked"):
            notify(c, n, t["created_by"], f"#{a.id} '{t['title']}' {verb}: {a.text or ''}", a.id)
    print(f"#{a.id} -> {status}")


def cmd_note(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n)
        task(c, a.id)
        c.execute("insert into notes(task, author, ts, text) values(?,?,?,?)", (a.id, n, time.time(), a.text))
        c.execute("update tasks set updated=? where id=?", (time.time(), a.id))
    print("noted")


def cmd_lease(a, c):
    n = me(a)
    now = time.time()
    with Tx(c):
        touch(c, n)
        r = c.execute("select * from leases where resource=?", (a.resource,)).fetchone()
        if r and r["owner"] != n and r["expires"] > now:
            die(f"{a.resource} leased by {r['owner']} until {ts(r['expires'])} ({r['why']})")
        c.execute("insert or replace into leases values(?,?,?,?,?)", (a.resource, n, a.why or "", now, now + a.ttl))
        c.execute("insert into events(ts, who, what) values(?,?,?)", (now, n, f"leased {a.resource} ({a.why})"))
    print(f"leased {a.resource} until {ts(now + a.ttl)}")


def cmd_unlease(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n)
        r = c.execute("select * from leases where resource=?", (a.resource,)).fetchone()
        if r and r["owner"] != n and not a.force:
            die(f"{a.resource} is leased by {r['owner']}")
        c.execute("delete from leases where resource=?", (a.resource,))
        c.execute("insert into events(ts, who, what) values(?,?,?)", (time.time(), n, f"released {a.resource}"))
    print(f"released {a.resource}")


def cmd_send(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n)
        if a.to == "all":
            rs = [r["name"] for r in c.execute("select name from workers where name!=?", (n,))]
        else:
            if not c.execute("select 1 from workers where name=?", (a.to,)).fetchone():
                die(f"no worker {a.to} (see `status`)")
            rs = [a.to]
        for r in rs:
            notify(c, n, r, a.text, a.task)
    sess = [r["session"] for r in c.execute(
        f"select session from workers where name in ({','.join('?' * len(rs))}) and session is not null", rs)]
    print(f"sent to {', '.join(rs) or 'nobody'}")
    if sess:
        print(f"  (to wake an idle session now, also SendMessage to: {', '.join(sess)})")


def show_msgs(rows):
    for m in rows:
        t = f" re #{m['task']}" if m["task"] else ""
        print(f"[{ts(m['ts'])}] {m['sender']}{t}: {m['text']}")


def cmd_inbox(a, c):
    n = me(a)
    with Tx(c):
        touch(c, n)
        rows = c.execute("select * from messages where recipient=? and read_ts is null order by id", (n,)).fetchall()
        if not a.peek:
            c.execute("update messages set read_ts=? where recipient=? and read_ts is null", (time.time(), n))
    show_msgs(rows) if rows else print("(no unread messages)")


def cmd_wait(a, c):
    n = me(a)
    end = time.time() + a.timeout
    while time.time() < end:
        with Tx(c):
            touch(c, n)
            rows = c.execute("select * from messages where recipient=? and read_ts is null order by id",
                             (n,)).fetchall()
            if rows:
                c.execute("update messages set read_ts=? where recipient=? and read_ts is null", (time.time(), n))
        if rows:
            show_msgs(rows)
            return
        time.sleep(a.poll)
    print("(timeout, no messages)")


def cmd_list(a, c):
    q = "select * from tasks" + ("" if a.all else " where status not in ('done','dropped')") + \
        " order by status='done', prio, id"
    for t in c.execute(q):
        own = t["owner"] or ""
        flag = "" if not own or alive(c, own) else " (stale)"
        print(f"#{t['id']:<3} p{t['prio']} {t['status']:<8} {own + flag:<26} [{t['area']}] {t['title']}")


def cmd_show(a, c):
    t = task(c, a.id)
    for k in ("title", "status", "owner", "prio", "area", "files", "created_by", "descr", "result"):
        if t[k] not in (None, ""):
            print(f"{k:>10}: {t[k]}")
    print(f"{'created':>10}: {ts(t['created'])}   updated: {ts(t['updated'])}")
    for nt in c.execute("select * from notes where task=? order by id", (a.id,)):
        print(f"  [{ts(nt['ts'])}] {nt['author']}: {nt['text']}")


def cmd_status(a, c):
    now = time.time()
    print("Workers:")
    for w in c.execute("select * from workers order by last_seen desc"):
        st = "active" if now - w["last_seen"] < STALE_S else "stale"
        print(f"  {w['name']:<24} {st:<6} seen {ts(w['last_seen'])}  session={w['session'] or '-'}  {w['descr'] or ''}")
    print("Leases:")
    ls = c.execute("select * from leases where expires>? order by resource", (now,)).fetchall()
    for l in ls:
        print(f"  {l['resource']:<12} {l['owner']} until {ts(l['expires'])}  {l['why']}")
    if not ls:
        print("  (none)")
    print("Open tasks:")
    cmd_list(argparse.Namespace(all=False), c)
    n = getattr(a, "as_", None) or os.environ.get("WORKLOG_WORKER")
    if n:
        k = c.execute("select count(*) from messages where recipient=? and read_ts is null", (n,)).fetchone()[0]
        print(f"Unread for {n}: {k}")


def cmd_log(a, c):
    for e in reversed(c.execute("select * from events order by id desc limit ?", (a.n,)).fetchall()):
        print(f"[{ts(e['ts'])}] {e['who']}: {e['what']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as", dest="as_", help="worker name (default $WORKLOG_WORKER)")
    s = p.add_subparsers(dest="cmd", required=True)

    def sub(name, fn, *args):
        sp = s.add_parser(name)
        sp.add_argument("--as", dest="as_", default=argparse.SUPPRESS)
        for spec in args:
            spec(sp)
        sp.set_defaults(fn=fn)
        return sp

    tid = lambda sp: sp.add_argument("id", type=int)  # noqa: E731
    text = lambda sp: sp.add_argument("text", nargs="?", default="")  # noqa: E731
    force = lambda sp: sp.add_argument("--force", action="store_true")  # noqa: E731

    sub("register", cmd_register, lambda sp: sp.add_argument("--session"), lambda sp: sp.add_argument("--desc"))
    sp = sub("add", cmd_add, lambda sp: sp.add_argument("title"))
    for f in ("--desc", "--area", "--files", "--owner"):
        sp.add_argument(f)
    sp.add_argument("--prio", type=int, default=2, help="1 high .. 3 low")
    sub("claim", cmd_claim, tid, force)
    sub("note", cmd_note, tid, lambda sp: sp.add_argument("text"))
    for name, status, verb in (("done", "done", "done"), ("release", "open", "released"),
                               ("block", "blocked", "blocked"), ("reopen", "open", "reopened"),
                               ("drop", "dropped", "dropped")):
        sub(name, lambda a, c, st=status, v=verb: set_status(a, c, st, v), tid, text, force)
    sp = sub("lease", cmd_lease, lambda sp: sp.add_argument("resource"))
    sp.add_argument("--ttl", type=int, default=3600)
    sp.add_argument("--why")
    sub("unlease", cmd_unlease, lambda sp: sp.add_argument("resource"), force)
    sp = sub("send", cmd_send, lambda sp: sp.add_argument("to"), lambda sp: sp.add_argument("text"))
    sp.add_argument("--task", type=int)
    sub("inbox", cmd_inbox, lambda sp: sp.add_argument("--peek", action="store_true"))
    sp = sub("wait", cmd_wait)
    sp.add_argument("--timeout", type=float, default=6 * 3600)
    sp.add_argument("--poll", type=float, default=10)
    sub("list", cmd_list, lambda sp: sp.add_argument("--all", action="store_true"))
    sub("show", cmd_show, tid)
    sub("status", cmd_status)
    sub("log", cmd_log, lambda sp: sp.add_argument("-n", type=int, default=30))

    a = p.parse_args()
    a.fn(a, db())


if __name__ == "__main__":
    main()
