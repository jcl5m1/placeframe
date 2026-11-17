# Shared worklog (multi-agent coordination)

The worklog moved: the Discord agent harness now provides one shared worklog for
every agent on the Discord server, on every machine. Use `wl` (agents the harness
runs have it on their PATH, as worker `<instance>@<machine>`); from a terminal
session use `~/github/jcl5m1/discord_agent/bin/wl --as <your-name>`.
`python3 ops/worklog.py ...` still works: it forwards to `wl`.

Quick reference (`wl help` lists everything):
- `wl list`, then `wl claim <id> "<what you're starting, up to 10 words>"`; if it
  isn't listed, `wl add "<title>" --files a.py,b.py` first.
- While you hold a claim: `wl checkin <id> "<up to 10 words>"` at least every
  30 minutes, or the claim is released. Longer notes: `wl note <id> "..."`.
- `wl done <id> "<result>"`, `wl block <id> "<need>"`, `wl release <id>`.
- Shared resources: `wl lease gpu --ttl 3600 --why "..."`, `wl unlease gpu`.
- Messages: `wl send <worker|all> "text" [--task <id>]`; agents the harness runs
  get them delivered, terminal sessions use `wl inbox` or `wl wait`.

The change log is the #worklog channel on Discord; `/worklog` there shows the status.
Full description: the harness README, "Shared worklog".
