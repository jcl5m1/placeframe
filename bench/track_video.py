"""Render a replayed tracked session as the phone would show it: camera frame (left), top-down view (right) with the
map slice, the path registered to the map, the localizations that agreed, and the status line.

  python bench/track_video.py REPLAY_DIR CANDIDATES.jsonl POINTS3D.txt ~/pf_data/videos/<folder>/<name>.mp4
        [--steps 2] [--speed 2]
Videos under ~/pf_data/videos appear in the dashboard's "Generated videos" page (a <name>.txt beside one is its
description). The tracker is the current tracker.py run over the saved candidates.

The top-down view:
  * is drawn in a frame fixed to the map: the map's orientation never changes, and the view only translates to keep
    the phone centred, at a fixed scale;
  * places every path point with the first confident map alignment after it that was known at that moment, so when
    the tracker recovers from drift the path before the recovery snaps back onto the map; the path breaks only at
    odometry jumps and where it leaves the height slice (another floor);
  * shows, between localizations (--steps - 1 frames per interval), the pose carried by odometry alone.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
import track_analyze as A  # noqa: E402

T = A.T
W, H, LEFT = 1280, 720, 540
SHOW_P = 0.8
SPAN_M = 14.0  # metres across the top-down view
SLICE_M = 1.0  # height window around the phone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("candidates")
    ap.add_argument("points", type=Path)
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=2, help="rendered frames per localization interval (1 = no odometry frames)")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed relative to the recording")
    a = ap.parse_args()
    session = json.loads((a.replay / "session.json").read_text())
    frames = [json.loads(line) for line in (a.replay / a.candidates).open()]
    pts = np.array([[float(v) for v in ln.split()[1:4]] for ln in a.points.open() if not ln.startswith("#")])
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)

    odom = [np.array(f["odom"]) for f in frames]
    times = [A.datetime.fromisoformat(f["t"]).timestamp() for f in frames]
    track = run_tracker(frames, odom, times)
    first = next((i for i, s in enumerate(track) if s["confident"]), None)
    if first is None:
        raise SystemExit("the tracker never became confident")
    display_from_map = np.eye(4)
    display_from_map[:3, :3] = track[first]["A"][:3, :3]  # gravity-up (odometry axes), fixed for the whole video
    map_display = (display_from_map[:3, :3] @ pts.T).T
    jumps = {i for i in range(1, len(odom)) if odometry_jump(odom[i - 1], odom[i], times[i] - times[i - 1])}

    tmp = Path(tempfile.mkdtemp())
    n = 0
    for i in range(len(frames)):
        for k in range(a.steps):
            if k and i + 1 >= len(frames):
                break
            w = k / a.steps
            pose = interpolate(odom[i], odom[i + 1], w) if k else odom[i]
            img = Image.new("RGB", (W, H), (14, 16, 20))
            cam = Image.open(a.replay / "img" / session[i]["image"]).rotate(-90, expand=True)  # RIGHT_TOP: upright
            cam.thumbnail((LEFT, H))
            img.paste(cam, ((LEFT - cam.width) // 2, (H - cam.height) // 2))
            draw_top_down(img, map_display, display_from_map, track, odom, jumps, i, pose)
            status(img, track[i], i, len(frames), k, font, small)
            img.save(tmp / f"{n:05d}.png")
            n += 1
    fps = a.steps * a.speed * (len(frames) - 1) / (times[-1] - times[0])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.3f}", "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
    print(f"wrote {a.out}: {n} frames at {fps:.2f} fps ({a.speed:g}x the recording)")


def run_tracker(frames, odom, times) -> list[dict]:
    tracker = T.Tracker()
    out = []
    for f, o, t in zip(frames, odom, times):
        use_prior = tracker.predicted_map_from_camera(o) is not None
        cands = [T.Candidate(np.array(c["m"]), c["conf"], c["inl"], c["src"])
                 for c in f["plain"] + (f["with_prior"] if use_prior else [])]
        res = tracker.update(o, cands, now=t)
        b = tracker.best
        p = b.probability if b else 0.0
        out.append({"A": b.odom_from_map.copy() if b else None, "p": p, "confident": b is not None and p >= SHOW_P,
                    "agreed": res.agreed, "selected": res.selected, "candidates": len(cands),
                    "places": len(tracker.hypotheses)})
    return out


def odometry_jump(a, b, dt) -> bool:
    step = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    return step > T.JUMP_MIN_M and step / max(dt, 0.05) > T.JUMP_SPEED_M_S


def interpolate(a, b, w):
    m = np.eye(4)
    m[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(w).as_matrix()
    m[:3, 3] = (1 - w) * a[:3, 3] + w * b[:3, 3]
    return m


def alignments_as_of(track, now: int) -> list:
    """For every frame j <= now: the first confident alignment at or after j known by frame `now` (hindsight), else
    the latest one before j; None before the first."""
    out = [None] * (now + 1)
    nxt = None
    for j in range(now, -1, -1):
        if track[j]["confident"]:
            nxt = track[j]["A"]
        out[j] = nxt
    last = None
    for j in range(now + 1):
        if track[j]["confident"]:
            last = track[j]["A"]
        if out[j] is None:
            out[j] = last
    return out


def alignment_for(track, j: int, now: int):
    return alignments_as_of(track, now)[j]


def draw_top_down(frame, map_display, display_from_map, track, odom, jumps, now, pose):
    # Drawn on its own canvas and pasted, so nothing drawn can spill over the camera image.
    size = H - 84
    img = Image.new("RGB", (size, size), (21, 25, 30))
    frame.paste(img_panel(img, size, map_display, display_from_map, track, odom, jumps, now, pose), (LEFT + 10, 74))


def img_panel(img, size, map_display, display_from_map, track, odom, jumps, now, pose):
    x0 = y0 = 0
    d = ImageDraw.Draw(img)
    aligned = alignments_as_of(track, now)
    current = aligned[now]
    if current is None:
        d.text((x0 + 12, y0 + size // 2), "waiting for the first confident localization", fill=(200, 200, 200))
        return img

    def to_display(alignment, p):
        q = display_from_map @ np.linalg.inv(alignment) @ np.append(p, 1.0)
        return q[:3]

    here = to_display(current, pose[:3, 3])
    s = size / SPAN_M

    def scr(q):
        return (x0 + size / 2 + (q[0] - here[0]) * s, y0 + size / 2 + (q[2] - here[2]) * s)

    def inside(u, v, margin=0):
        return x0 + margin <= u < x0 + size - margin and y0 + margin <= v < y0 + size - margin

    near = map_display[np.abs(map_display[:, 1] - here[1]) <= SLICE_M]
    px = img.load()
    for q in near[::2]:
        u, v = scr(q)
        if inside(u, v):
            px[int(u), int(v)] = (120, 130, 140)
    # Each path point with the alignment that best places it as of now (hindsight within the frames seen so far).
    seg = []
    for j in range(now + 1):
        al = aligned[j]
        q = to_display(al, odom[j][:3, 3]) if al is not None else None
        ok = q is not None and abs(q[1] - here[1]) <= SLICE_M and j not in jumps
        if not ok and len(seg) > 1:
            d.line(seg, fill=(66, 165, 245), width=3)
        if not ok:
            seg = []
        if q is not None and abs(q[1] - here[1]) <= SLICE_M:
            seg.append(scr(q))
    seg.append(scr(here))
    if len(seg) > 1:
        d.line(seg, fill=(66, 165, 245), width=3)
    for j in range(now + 1):
        if track[j]["agreed"] and track[j]["confident"]:
            q = to_display(aligned[j], odom[j][:3, 3])
            u, v = scr(q)
            if abs(q[1] - here[1]) <= SLICE_M and inside(u, v, 4):
                d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(102, 187, 106))
    fwd = (display_from_map @ np.linalg.inv(current))[:3, :3] @ pose[:3, :3][:, 2]
    nrm = np.hypot(fwd[0], fwd[2]) or 1
    u, v = x0 + size / 2, y0 + size / 2
    d.line([(u, v), (u + fwd[0] / nrm * 24, v + fwd[2] / nrm * 24)], fill=(255, 255, 255), width=3)
    d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(255, 255, 255))
    d.line([(x0 + 12, y0 + size - 14), (x0 + 12 + 2 * s, y0 + size - 14)], fill=(200, 200, 200), width=2)
    d.text((x0 + 18 + 2 * s, y0 + size - 22), "2 m", fill=(200, 200, 200))
    return img


def status(img, st, i, n, k, font, small):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 64], fill=(0, 0, 0))
    p = st["p"]
    if k:
        line = f"Between localizations: odometry pose · {100 * p:.0f}%"
    else:
        line = f"Localized · {100 * p:.0f}%" if st["confident"] else (f"Localizing… {100 * p:.0f}%" if st["A"] is not None else "Not localized")
    d.text((16, 8), line, font=font, fill=(105, 240, 174) if st["confident"] and not k else (255, 255, 255))
    sel = st["selected"]
    detail = (f"frame {i + 1}/{n} · {st['candidates']} candidates · "
              + (f"agreed: conf {sel.confidence:.2f}, {sel.num_inliers} inliers ({sel.source})" if st["agreed"] and sel else "none agreed")
              + f" · places tracked {st['places']}")
    d.text((16, 38), detail, font=small, fill=(200, 200, 200))


if __name__ == "__main__":
    main()
