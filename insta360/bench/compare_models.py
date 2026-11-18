#!/usr/bin/env python3
"""Compare two COLMAP text models frame by frame (camera0 = rig pose): similarity-align B to A on shared frames,
then report per-rig position / rotation differences, plus point and reprojection stats.

  python compare_models.py A/images.txt B/images.txt
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R


def frames(path):
    out = {}
    lines = [ln for ln in open(path) if ln.strip() and not ln.startswith("#")]
    for ln in lines[0::2]:
        f = ln.split()
        rig, cam, stem = f[9].split("/")
        if cam != "camera0":
            continue
        q = [float(x) for x in f[1:5]]
        t = np.array([float(x) for x in f[5:8]])
        Rcw = R.from_quat([q[1], q[2], q[3], q[0]])
        out[(rig, stem)] = (Rcw.inv(), -(Rcw.inv().apply(t)))
    return out


def umeyama(a, b):
    ma, mb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((a - ma) ** 2).sum(1).mean()
    return s, Rm, mb - s * Rm @ ma


A, B = frames(sys.argv[1]), frames(sys.argv[2])
common = sorted(set(A) & set(B))
print(f"frames: A {len(A)}, B {len(B)}, shared {len(common)}; only in A {len(set(A) - set(B))}, only in B {len(set(B) - set(A))}")
pa = np.array([B[k][1] for k in common]); pb = np.array([A[k][1] for k in common])
s, Rm, t = umeyama(pa, pb)
Ralign = R.from_matrix(Rm)
by = defaultdict(list)
for k in common:
    p = s * Rm @ B[k][1] + t
    dpos = np.linalg.norm(p - A[k][1])
    drot = np.degrees((A[k][0].inv() * (Ralign * B[k][0])).magnitude())
    by[k[0]].append((dpos, drot))
print(f"similarity B->A: scale {s:.4f}")
for rig, v in sorted(by.items()):
    v = np.array(v)
    print(f"  {rig}: n={len(v):4d}  position diff p50 {np.median(v[:,0]):.3f} m  p90 {np.percentile(v[:,0],90):.3f} m   "
          f"rotation diff p50 {np.median(v[:,1]):.2f} deg  p90 {np.percentile(v[:,1],90):.2f} deg")
