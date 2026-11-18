#!/usr/bin/env python3
"""Moved: the shared worklog is now provided by the Discord agent harness.

Every agent on the Discord server, on every machine, shares one worklog through
the harness's `wl` command (~/github/jcl5m1/discord_agent/bin/wl; agents the
harness runs already have it on their PATH). This file forwards the old
commands to it, so existing instructions keep working. The old per-project
SQLite version is kept as ops/worklog_legacy.py; its data stays in
~/pf_worklog/worklog.db, but nothing is written there any more.
"""
import os
import sys

WL = os.path.expanduser("~/github/jcl5m1/discord_agent/bin/wl")
args = sys.argv[1:]
# The old `register NAME ...` took the worker name as an argument.
if args[:1] == ["register"] and len(args) > 1 and not args[1].startswith("-"):
    args = ["register", "--as", args[1], *args[2:]]
os.execv(sys.executable, [sys.executable, WL, *args])
