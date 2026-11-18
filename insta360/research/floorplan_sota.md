# Floor/Wall Extraction, Vector Floor Plans and Walkable Area from 360 + Sparse SfM

Status: literature/web survey, Nov 2026. Nothing was run. Legend: **[V]** means verified from the paper, repo or official project page (URL inline). **[I]** means my inference or recollection that was not checked against a primary source.

## 0. What our data does and doesn't support

- **Sparse SfM points (30k–300k, metric, gravity-aligned, y-down).** Floors and white walls have little texture, so SfM puts few points on them. Points cluster on furniture, edges, door frames and decor. A raw density map from this cloud therefore looks very different from the dense RGB-D/LiDAR density maps that all learned floor-plan models were trained on (Structured3D, SceneCAD/ScanNet). [I]
- **What we get almost for free.** Gravity and metric scale. Camera height above the floor is roughly constant (the camera is handheld or on a pole), and we have the trajectory, which is a set of positions known to be walkable. Both are strong priors. [I]
- **What matters most: the 360 panoramas.** Panorama layout networks need only an equirectangular image plus gravity alignment. They do not depend on point density at all. We already have metric poses, which removes the hardest step in the multi-view layout fusion papers (scale and pose recovery). [I]

## 1. Classical geometry on the point cloud

| Step | Input | Output | Notes |
|---|---|---|---|
| Floor/ceiling heights | Point y-values, trajectory | Floor and ceiling planes | Floor is about `cam_y + h_cam`. Use a histogram peak or a RANSAC plane near that height, done per level. Robust even when the cloud is sparse, because the trajectory gives the prior. [I] |
| Wall plane detection | Points between floor+0.3 m and ceiling−0.3 m | Vertical planes and line segments | RANSAC or region growing with a normal ⟂ gravity constraint. Open3D `segment_plane` and PCL are permissive (MIT/BSD). [I] |
| Density / height maps | Points binned in XZ at 2–5 cm | 2D images: count, max height, wall-ness | This is the standard input for every learned method in §2. [V: RoomFormer README, https://github.com/ywyue/RoomFormer] |
| Manhattan/Atlanta regularization | Wall segments | Snapped dominant orientations | Histogram of segment angles, then snap to 1–3 dominant directions. Atlanta world allows several horizontal frames. [I] |
| Room segmentation | Free-space map | Room labels | Options are distance-transform watershed, Voronoi/morphological methods, or graph cuts over a cell complex built from wall lines (Floor-SP style). [I] |
| Polygonization | Room masks or cell complex | Room polygons, wall polylines | Alpha shapes, then Douglas–Peucker, then orientation snapping. Alternatively, a cell-complex arrangement of wall lines followed by labelling cells inside/outside. [I] |

- **Fit:** high. It runs in seconds to a minute on CPU for a house. The weak point is **wall evidence**: textureless walls can be almost empty in SfM. Mitigations are (a) also projecting points from the *rays* (free-space carving: each camera-to-point ray marks the cells it passes through as empty), and (b) adding monocular depth (§4). Free-space carving with SfM rays is cheap and especially powerful here, because track length gives a confidence weight per point. [I]

## 2. Learned floor-plan reconstruction (density map / point cloud to polygons)

All of these were trained on Structured3D (synthetic, dense, 256×256 density maps) or SceneCAD (ScanNet). Structured3D numbers below are Room/Corner/Angle F1, taken from the PolyRoom paper's Table 1 [V: https://arxiv.org/html/2407.10439v1].

| Method | Input → Output | S3D Room / Corner / Angle F1 | Code / license | Notes |
|---|---|---|---|---|
| MonteFloor (ICCV'21) | Density map → polygons via MCTS | 95.0 / 82.5 / 80.5 [V] | — | Slow search. Superseded. |
| HEAT (CVPR'22) | Density map → corner/edge graph | 95.4 / 82.5 / 78.3 [V] | GPL-3.0, non-commercial models [V: https://github.com/woodfrog/heat] | Outputs a graph rather than room polygons. |
| RoomFormer (CVPR'23) | Density map → set of room polygons (two-level DETR queries) | 97.3 / 87.2 / 81.2 [V] | MIT [V: https://github.com/ywyue/RoomFormer] | Claims "significantly faster inference" than earlier methods (no number in the README) [V]. Our best-licensed learned option. |
| SLIBO-Net (NeurIPS'23) | Density map → polygons (slicing boxes) | 98.4 / 85.4 / 84.4 [V via PolyRoom table] | Code status unverified [I] | Strongest on room and angle accuracy. |
| PolyRoom (ECCV'24) | Density map + Mask R-CNN room init → polygons | 98.3 / 90.2 / 85.2 [V]. SceneCAD: Room F1 91.2, Corner F1 78.0 [V] | Checkpoints on Google Drive. **No license file visible** [V: https://github.com/3dv-casia/PolyRoom] | Current best corner accuracy. Built on RoomFormer and MMDetection. |
| FRI-Net (ECCV'24) | Density map → room-wise implicit SDF → polygons | Paper claims SOTA. Aggregator lists Room F1 96.5 (binary input). **Not verified** [https://arxiv.org/abs/2407.10687] | Unverified | The paper itself notes weakness on fine internal walls. |
| PolyDiffuse (NeurIPS'23) | Density map + initial guess → diffusion-refined polygons | Not verified | Unverified | Works as a refiner on top of RoomFormer. Iterative and slower. [I] |
| SceneScript (Meta, ECCV'24) | Points or images → CAD-like token language (walls, doors, windows) | — | Weights are research-only, gated behind a signup [V: https://projectaria.com/scenescript] | Trained on Aria Synthetic Environments. |
| SpatialLM 1.1 (2025) | Axis-aligned point cloud (z-up) → walls, doors, windows + 3D boxes | Structured3D layout F1 at 0.25/0.5 IoU: 94.3/93.5 (fine-tuned) vs RoomFormer 83.4/81.4 [V: https://github.com/manycore-research/SpatialLM] | Code Apache-2.0. **Weights CC-BY-NC-4.0** [V] | 0.5B–1B LLM plus the Sonata encoder. Its demos use MASt3R-SLAM/SLAM3R clouds from video, which are denser than COLMAP. Expect it to fit on a 3090 [I]. |

- **Fit for us:** medium. The density-map domain gap is the main risk. Mitigations [I]:
  1. Build the density map from *densified* points (MVS, or monocular depth back-projection, §4), not raw SfM.
  2. Render a "wall-likelihood" map from free-space carving so it looks closer to the training distribution.
  3. Fine-tune RoomFormer/PolyRoom on our data using pseudo-labels from the classical baseline (§8).
- **Compute:** all of these are small DETR-style models on 256×256 inputs, so inference takes well under a second on a 3090. [I]
- **Licensing:** the commercial-safe set is RoomFormer (MIT) plus our own code. HEAT, SceneScript and the SpatialLM weights are non-commercial. PolyRoom's license is unclear.

## 3. Panorama room layout estimation (per frame) and multi-view fusion

Single-panorama models predict, for each image column, the floor and ceiling boundary (plus corners). Given camera height, this becomes a metric 2D room polygon centred on the camera.

| Method | MatterportLayout 2D/3D IoU | ZInD-All 2D/3D IoU | Code / license |
|---|---|---|---|
| HorizonNet (CVPR'19) | Cuboid: PanoContext+S2D3D 3DIoU 83.9 [V] | weights exist for ST3D, MP3D, ZInD | MIT. Weights carry the training dataset terms [V: https://github.com/sunset1995/HorizonNet] |
| LED2-Net (CVPR'21) | 82.37 / 80.05 [V] | 82.31 / 80.28 [V] | Repo exists, license unverified [I] |
| LGT-Net (CVPR'22) | 83.52 / 81.11 [V] | 84.07 / 82.09 [V] | MIT. Optional Manhattan post-processing [V: https://github.com/zhigangjiang/LGT-Net] |
| DOP-Net (2024) | 84.11 / 81.70 [V] | 83.92 / 81.87 [V] | Unverified |
| Bi-Layout (CVPR'24) | 84.56 / 82.05 (equivalent branch) [V] | 84.90 / 82.96 [V] | https://github.com/LIAGM/Bi_Layout, license unverified [V link] |

All IoU numbers are from the Bi-Layout paper, Table 1a [V: https://arxiv.org/html/2404.09993].

- **Bi-Layout** predicts two layouts per panorama: an "enclosed" one that stops at openings, and an "extended" one that covers all visible space [V: https://liagm.github.io/Bi_Layout]. That split maps directly onto our needs: enclosed gives room polygons, extended gives visible floor (walkable candidate).

### Fusion with known poses

- **360-DFPE** (RA-L'22) builds a whole floor plan from a 360 sequence. It registers each per-frame layout to VSLAM poses, recovers scale, and clusters frames into rooms [V: https://arxiv.org/abs/2112.06180, code https://github.com/EnriqueSolarte/direct_360_FPE, old dependency pins]. We already have metric poses, so the scale-recovery part is unnecessary. [I]
- **360-MLC** (NeurIPS'22) projects layout boundaries from many views into a shared frame. It then re-projects them as uncertainty-weighted pseudo-labels to self-train a layout model on an unlabeled new domain [V: https://arxiv.org/abs/2210.12935]. This is exactly the recipe for adapting a layout model to our homes without labels.
- **Simple fusion recipe [I].** For each frame (subsample to about 0.5 fps):
  1. Gravity-align the equirectangular image using our pose. This avoids running HorizonNet's vanishing-point preprocessing.
  2. Run LGT-Net or Bi-Layout.
  3. Convert the floor boundary to metric XZ, using camera height from the SfM floor plane.
  4. Transform into the world frame.
  5. Accumulate rays into a 2D occupancy grid: pixels between the camera and the boundary are free, and the boundary itself is wall evidence.
  6. Polygonize as in §1.

  Doing the fusion in the occupancy domain tolerates occlusion and multi-room panoramas far better than intersecting polygons.
- **Compute:** LGT-Net/HorizonNet run in tens of milliseconds per 1024×512 pano on a GPU [I]. About 200 frames takes well under a minute.
- **Fit:** high. These models are the cheapest *learned* source of wall geometry, and they are independent of SfM sparsity. Known weaknesses are cluttered and non-Manhattan rooms, open-plan spaces (ambiguous boundaries, which is the problem Bi-Layout targets), and handheld camera-height variation. That last one we can measure from SfM.

## 4. Image semantic segmentation lifted to 3D

| Model | Output | License | Notes |
|---|---|---|---|
| OneFormer (ADE20K) | 150 classes incl. wall and floor | MIT. Swin-L 57.0–57.4, DiNAT-L 58.3 mIoU single-scale [V: https://github.com/SHI-Labs/OneFormer] | Strong out-of-the-box wall/floor/ceiling/door classes. |
| Mask2Former (ADE20K) | Same | MIT [I] | Slightly weaker. Easier to fine-tune (detectron2). |
| SAM2 / Grounded-SAM / OpenSeeD | Masks from prompts | SAM2 Apache-2.0 [I] | Useful for refining boundaries and for open-vocabulary prompts like "rug", "stairs". |
| Depth Anything V2 | Relative or metric depth | Small Apache-2.0. Base/Large CC-BY-NC [V: https://github.com/DepthAnything/Depth-Anything-V2] | Indoor metric variants exist (see the `metric_depth` folder). |
| UniDepth V2 | Metric depth + intrinsics | CC-BY-NC-4.0 [V: https://github.com/lpiccinelli-eth/UniDepth] | |
| Metric3D v2 | Metric depth + normals | BSD-2 code, but the authors ask commercial users to contact them [V-ish: https://huggingface.co/zachL1/Metric3D] | Normals help separate wall from floor. |
| MoGe-2 | Metric point map. Has an experimental `infer_panorama` | License unverified [V feature: https://raw.githubusercontent.com/microsoft/MoGe/main/README.md] | Natural fit for our equirect frames. |

- **Lifting recipe [I].**
  1. Run segmentation on the six 90° cubemap views. ADE-trained models expect perspective images, so these work better than the equirect.
  2. Project every SfM point into its observing views and take a majority vote weighted by track length. This gives per-point floor/wall/other labels.
  3. Densify: back-project floor/wall pixels using monocular depth, scale- and shift-aligned per frame to the SfM points visible in that frame (least squares on inverse depth). The result is dense floor and wall points, which close the textureless gap.
- **Compute:** OneFormer Swin-L is about 0.2–0.5 s per 1024² image on a 3090 [I]. 400 frames × 6 views gives about 10–20 minutes, so subsample frames or use a smaller backbone for the fast path.
- **Fit:** high. This is the best way to put explicit *wall* and *floor* labels into the map. Wall/floor confusion around baseboards and glass is the typical failure.

## 5. 3D point cloud semantic segmentation

- **Point Transformer V3 / Sonata (Pointcept)** is SOTA on ScanNet/S3DIS. The Sonata weights' licensing is inconsistent across sources (CC-BY-NC is reported for the official release) [V conflict: https://github.com/facebookresearch/sonata]. **OneFormer3D** and **Swin3D** are similar (dense RGB-D training).
- **Fit:** low for raw SfM. These networks learn local geometric context on dense, uniformly sampled surfaces (2 cm voxels). Our clouds are 10–100× sparser and biased toward textured objects. Large floor and wall regions have no points at all, so there is nothing to label. Expect poor transfer without densification [I]. They become relevant only after densification (§4) or MVS, and even then 2D-segmentation-then-lift is cheaper and probably better for just walls and floors [I]. SpatialLM's use of Sonata on video-SLAM clouds is the main evidence that this can work with denser reconstructions [V].

## 6. Walkable area / free space

- **Floor-plane occupancy [I].** Start from a 2D grid on the floor plane (5 cm cells):
  - **free:** floor-labelled points, or cells carved by the floor part of panorama layouts (§3 "extended" region).
  - **occupied:** any point at height 0.1–1.8 m above the floor (furniture, walls).
  - **unknown:** everything else.

  Then remove free cells closer than the agent radius (about 0.25 m) to occupied cells, and keep the connected component that contains the trajectory.
- **Navmesh.** Recast (Zlib license) voxelizes a *triangle mesh* and outputs a navmesh with agent radius, height, climb and slope parameters [V: https://github.com/recastnavigation/recastnavigation]. You would need a mesh first: the floor polygon extruded with furniture boxes, or a Poisson/TSDF mesh from densified depth. That is overkill unless multi-level or stair navigation is needed. A 2D grid matches Recast for single-floor houses [I].
- **Trajectory as label [I].**
  - The walking path is a *positive-only* label for walkable floor. Dilate it by about 0.3 m for body width, and project the camera position down to the floor height.
  - Cells crossed by camera-to-point rays below 1.5 m are free space, but not necessarily walkable (e.g. under tables).
  - Points within ±3 cm of the floor plane directly under or near the trajectory are reliable floor samples. Use them to fit and validate the floor plane and to seed region growing.
  - Treat this as PU (positive-unlabeled) learning: the path covers only part of the floor, so never treat off-path area as negative.

## 7. Recommended pipeline

### A. Cheap robust baseline (CPU/GPU, about 2–5 minutes per house) [I]

1. **Floor/ceiling.** Floor height = median(camera y) + camera height. Refine with RANSAC on points within ±10 cm, using trajectory floor points as seeds. Ceiling = histogram peak above the camera. Handle multiple levels by clustering trajectory heights.
2. **Free-space carving.** Rasterize SfM camera-to-point rays (track-length weighted) into a 5 cm XZ grid, using only rays inside the 0.2–2.0 m height band. Accumulate hit counts (walls/objects) and pass-through counts (free).
3. **Pano layouts.** LGT-Net (MIT, MP3D or ZInD weights) on gravity-aligned equirects, 1 frame in 4. Fuse the floor boundaries into the same grid as free and wall evidence. This is the step that rescues textureless walls.
4. **Walls.** Take high hit-count cells with a low pass-through count, run RANSAC/Hough lines, then snap to Manhattan/Atlanta directions.
5. **Rooms.** Build a cell complex from the wall lines and label cells inside/outside via graph cut, with data term = free evidence and smoothness = wall evidence. Use distance-transform watershed for room splitting, then Douglas–Peucker for room polygons and wall polylines (with thickness).
6. **Walkable.** Free cells minus obstacle band (0.1–1.8 m) minus agent-radius erosion, keeping the component containing the trajectory. Export as GeoJSON polygons.

Why this baseline: every step is deterministic, debuggable and license-clean. It exploits our unique assets (metric poses, gravity, trajectory, 360) and doesn't depend on point density.

### B. Learned upgrade path, in order of expected value per compute [I]

1. **OneFormer floor/wall/door segmentation on cubemap faces, lifted to SfM points.** Feed the labels into the grid, which gives cleaner walls and furniture/floor separation. Add monocular-depth densification (Depth Anything V2-Small Apache, or MoGe-2 if its license allows) for dense floor and wall points.
2. **Swap LGT-Net for Bi-Layout.** Use the enclosed output for rooms and the extended output for floor visibility.
3. **RoomFormer (MIT) as the polygonizer, run on our densified density map.** Fine-tune on pseudo-labels from baseline A. If licensing permits, try PolyRoom for better corners.
4. **Research comparator only (non-commercial weights):** SpatialLM 1.1 on the densified cloud, converted to z-up. It is also the only option here that outputs doors and windows directly.

## 8. Future work: fine-tuning

| Model | Why it would improve | Cheap labels | Rough data need [I] |
|---|---|---|---|
| Pano layout (LGT-Net/Bi-Layout) | Domain shift: Insta360 stitching seams, handheld height, cluttered homes | 360-MLC-style multi-view consistency pseudo-labels from our own poses [V method]. Fused baseline polygons re-projected into each pano. | 20–50 houses (≈5–10k panos) of self-training. 360-MLC shows gains with no GT [V] |
| OneFormer/Mask2Former wall/floor/ceiling | ADE20K is generic. Our concern is a 3–5-class problem | Pseudo-labels: pixels whose SfM points lie on the fitted floor plane = floor. Pixels on RANSAC wall planes = wall. Refine with SAM2. | 2–5k images. A few hundred hand-corrected images is enough to tune [I] |
| RoomFormer/PolyRoom | Density-map domain gap | Baseline-A polygons (filtered by confidence) plus 20–30 hand-fixed plans for validation. Augment with Structured3D maps that are sparsified/degraded to mimic SfM. | 100+ houses of pseudo-labels, or synthetic degradation of Structured3D's 3.5k scenes |
| Walkable segmentation (2D grid UNet or point head) | Learning furniture footprints and under-table space | Trajectory as positive (PU learning). Floor points under the path. Obstacle points as negatives. | Tens of houses. The labels are noisy but plentiful (every walk) |
| Mono depth (scale/shift head) | Better metric alignment indoors | SfM points as sparse metric depth supervision | Thousands of frames (we already have them) |

**Self-training loop [I].** Run the baseline, keep high-confidence regions (where the pano-layout, SfM-carving and segmentation sources agree), then train. Run the improved model, re-fuse, and repeat. Always hold out 5–10 hand-annotated houses to measure room/corner F1 with the Structured3D/MonteFloor evaluation scripts in the RoomFormer repo.

## 9. Verification gaps

- **Not verified:** FRI-Net, SLIBO-Net, PolyDiffuse and DOP-Net code licenses; FRI-Net's own benchmark table (the PDF was not machine-readable); per-model runtime and VRAM (none of the repos state them); MoGe-2, SAM2 and Mask2Former licenses (from memory); and the conflicting Sonata weight license.
- **Inferences:** all fit assessments and compute estimates for sparse SfM are mine and should be validated on 2–3 of our houses before committing.
