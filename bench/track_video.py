"""Render a replayed tracked session as the phone would show it: camera frame (left), top-down view (right) with the
map slice, the path registered to the map and the localizations that agreed, and the status line.

  python bench/track_video.py REPLAY_DIR CANDIDATES.jsonl POINTS3D.txt ~/pf_data/videos/<folder>/<name>.mp4 [--fps 4]
Videos under ~/pf_data/videos appear in the dashboard's "Generated videos" page (a <name>.txt beside one is its
description).
The tracker is the current tracker.py run over the saved candidates.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import track_analyze as A  # noqa: E402

T = A.T
W, H, LEFT = 1280, 720, 540
SHOW_P = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("candidates")
    ap.add_argument("points", type=Path)
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=4)
    a = ap.parse_args()
    session = json.loads((a.replay / "session.json").read_text())
    frames = [json.loads(line) for line in (a.replay / a.candidates).open()]
    pts = np.array([[float(v) for v in ln.split()[1:4]] for ln in a.points.open() if not ln.startswith("#")])
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)

    tracker = T.Tracker()
    display = None  # display_from_map, fixed at the first confident frame
    aligns, path, fixes = [], [], []  # (frame index, display_from_odom); odometry positions; fixes (index, pos)
    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):
        odom = np.array(f["odom"])
        t = A.datetime.fromisoformat(f["t"]).timestamp()
        use_prior = tracker.predicted_map_from_camera(odom) is not None
        cands = [T.Candidate(np.array(c["m"]), c["conf"], c["inl"], c["src"])
                 for c in f["plain"] + (f["with_prior"] if use_prior else [])]
        res = tracker.update(odom, cands, now=t)
        best = tracker.best
        p = best.probability if best else 0.0
        path.append((i, odom[:3, 3]))
        if best is not None and p >= SHOW_P:
            if display is None or (aligns and np.linalg.norm((aligns[-1][1] @ odom)[:3, 3]
                                                             - (display @ np.linalg.inv(best.odom_from_map) @ odom)[:3, 3]) > 1.0):
                display, aligns, fixes = best.odom_from_map.copy(), [], []
            aligns.append((i, display @ np.linalg.inv(best.odom_from_map)))
            if res.agreed:
                fixes.append((i, odom[:3, 3]))
        img = Image.new("RGB", (W, H), (14, 16, 20))
        cam = Image.open(a.replay / "img" / session[i]["image"]).rotate(-90, expand=True)  # RIGHT_TOP: turn upright
        cam.thumbnail((LEFT, H))
        img.paste(cam, ((LEFT - cam.width) // 2, (H - cam.height) // 2))
        draw_top_down(img, pts, display, aligns, path, fixes, odom, i)
        d = ImageDraw.Draw(img)
        status = f"Localized · {100 * p:.0f}%" if p >= SHOW_P else (f"Localizing… {100 * p:.0f}%" if best else "Not localized")
        d.rectangle([0, 0, W, 64], fill=(0, 0, 0))
        d.text((16, 8), status, font=font, fill=(105, 240, 174) if p >= SHOW_P else (255, 255, 255))
        sel = res.selected
        detail = (f"frame {i + 1}/{len(frames)} · {len(cands)} candidates · "
                  + (f"agreed: conf {sel.confidence:.2f}, {sel.num_inliers} inliers ({sel.source})" if res.agreed and sel
                     else "none agreed") + f" · places tracked {len(tracker.hypotheses)} · odometry jumps {tracker.jumps}")
        d.text((16, 38), detail, font=small, fill=(200, 200, 200))
        img.save(tmp / f"{i:05d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out], check=True)
    print("wrote", a.out)


def draw_top_down(img, pts, display, aligns, path, fixes, odom, now):
    x0, size = LEFT + 10, H - 84
    y0 = 74
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x0 + size, y0 + size], fill=(21, 25, 30))

    def to_display(i, p):
        if not aligns:
            return p
        j = next((k for k, (fi, _) in enumerate(aligns) if fi >= i), len(aligns) - 1)
        m = aligns[j][1]
        return m[:3, :3] @ p + m[:3, 3]

    here = to_display(now, odom[:3, 3])
    pp = [to_display(i, p) for i, p in path]
    near = [q for q in pp if abs(q[1] - here[1]) <= 1.0]
    span = max(8.0, np.ptp(np.array(near)[:, [0, 2]], axis=0).max() + 4) if len(near) > 1 else 8.0
    c = np.mean(np.array(near)[:, [0, 2]], axis=0) if near else here[[0, 2]]
    s = size / span

    def scr(q):
        return (x0 + size / 2 + (q[0] - c[0]) * s, y0 + size / 2 + (q[2] - c[1]) * s)

    if display is not None:
        md = (display[:3, :3] @ pts.T).T + display[:3, 3]
        md = md[np.abs(md[:, 1] - here[1]) <= 1.0]
        px = img.load()
        for q in md[::2]:
            u, v = scr(q)
            if x0 <= u < x0 + size and y0 <= v < y0 + size:
                px[int(u), int(v)] = (120, 130, 140)
    seg = []
    for q in pp + [None]:
        if q is not None and abs(q[1] - here[1]) <= 1.0:
            seg.append(scr(q))
            continue
        if len(seg) > 1:
            d.line(seg, fill=(66, 165, 245), width=3)
        seg = []
    d.rectangle([x0 + size, y0, W, H], fill=(14, 16, 20))  # keep the path inside the panel
    for i, p in fixes:
        q = to_display(i, p)
        u, v = scr(q)
        if abs(q[1] - here[1]) <= 1.0 and x0 + 4 <= u <= x0 + size - 4 and y0 + 4 <= v <= y0 + size - 4:
            d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(102, 187, 106))
    u, v = scr(here)
    m = aligns[-1][1] if aligns else np.eye(4)
    fwd = m[:3, :3] @ odom[:3, :3][:, 2]
    n = np.hypot(fwd[0], fwd[2]) or 1
    d.line([(u, v), (u + fwd[0] / n * 22, v + fwd[2] / n * 22)], fill=(255, 255, 255), width=3)
    d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(255, 255, 255))


if __name__ == "__main__":
    main()
