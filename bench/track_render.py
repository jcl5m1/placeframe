"""Draw a replayed tracked session the way the app does: map points (cut at walking height) in a display frame fixed
at the first confident fix, raw odometry path, and the path registered with the tracker's alignments.

  python bench/track_render.py REPLAY_DIR MAP_SFM_DIR OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

d, sfm, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
frames = [json.loads(line) for line in (d / "candidates.jsonl").open()]
odom = [np.array(f["odom"]) for f in frames]
conf = [(i, np.array(f["best_A"])) for i, f in enumerate(frames) if f["best_A"] and f["best_p"] >= 0.8]
D = conf[0][1]  # display_from_map
pts = np.array([[float(v) for v in line.split()[1:4]] for line in (sfm / "points3D.txt").open() if not line.startswith("#")])
pd = (D[:3, :3] @ pts.T).T + D[:3, 3]
times = [i for i, _ in conf]
raw, reg = [], []
for i, o in enumerate(odom):
    j = next((k for k, t in enumerate(times) if t >= i), len(times) - 1)
    raw.append(o[:3, 3])
    reg.append((D @ np.linalg.inv(conf[j][1]) @ o)[:3, 3])
raw, reg = np.array(raw), np.array(reg)
h = np.median(reg[:, 1])
pd = pd[np.abs(pd[:, 1] - h) < 1.0]
lo = np.minimum(reg[:, [0, 2]].min(0), raw[:, [0, 2]].min(0)) - 2
hi = np.maximum(reg[:, [0, 2]].max(0), raw[:, [0, 2]].max(0)) + 2
S = 1000 / (hi - lo).max()
img = Image.new("RGB", (1000, 1000), (18, 20, 26))
px = img.load()
for x, _, z in pd[::2]:
    u, v = int((x - lo[0]) * S), int((z - lo[1]) * S)
    if 0 <= u < 1000 and 0 <= v < 1000:
        px[u, v] = (150, 160, 170)
dr = ImageDraw.Draw(img)
dr.line([((x - lo[0]) * S, (z - lo[1]) * S) for x, _, z in raw], fill=(230, 120, 60), width=2)
dr.line([((x - lo[0]) * S, (z - lo[1]) * S) for x, _, z in reg], fill=(66, 165, 245), width=3)
dr.text((10, 10), "orange: raw odometry   blue: registered with tracked alignments   grey: map", fill=(255, 255, 255))
img.save(out)
print(len(conf), "confident frames of", len(frames))
