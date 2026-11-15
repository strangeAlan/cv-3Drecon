# Experiment log

Each entry: hypothesis, setup (config, manifest, seed), result, verified facts /
inferences / open issues.

## 10/27 Setup

- Data: DA3-BENCH `eth3d.zip` from hf-mirror (15.1 GB, sha256 `2a12b961…6917` matches
  the HF LFS oid), extracted to `/home/hsy/datasets/da3-bench/eth3d` (35 GB), zip deleted.
  11 scenes (DA3 benchmark list; terrace and meadow excluded upstream). Images are the
  original distorted DSLR JPGs (THIN_PRISM_FISHEYE GT calibration), GT = combined mesh,
  GT depth + masks per image.
- Code: VGGT official commit `a288dd0` (43 files, git-blob sha1 verified via GitHub API);
  DA3 repo at `/home/hsy/references/Depth-Anything-3` (commit `3d835ec`), benchmark code used unmodified.
- Weights: local VGGT-1B (`model.pt`), DA3NESTED-GIANT-LARGE (the original, not `-1.1`).
- Env: standalone CPython 3.11.17 + `.venv`, packages via uv from aliyun PyPI.
- Reference numbers (DA3 paper, ETH3D, all views ≤100): VGGT AUC3 26.3 / AUC30 80.8 /
  F1 57.2 (pred. pose); DA3-Giant 48.4 / 91.2 / 79.0. ETH3D is listed as evaluation-only
  data in the DA3 paper.

Protocol decisions:
- Feed-forward models take the distorted images directly and are fused with their
  predicted pinhole intrinsics, exactly as in the DA3 benchmark.
- COLMAP (SIMPLE_RADIAL, camera mode AUTO, i.e. defaults) produces depth on its
  undistorted grid; `fuse_colmap` maps the GT mask through COLMAP's own estimated
  camera, then uses the same alignment / TSDF / sampling as DA3.
- Unregistered COLMAP views are excluded from fusion; `posefail_*` counts their pairs
  as failures.
- `fscore_vis`: recall against GT points visible in the case's own views, because the
  whole-scene GT makes recall a coverage measure for small view counts.

Environment pitfalls found during setup (all fixed, pinned in `env/requirements.txt`):
- open3d 0.20 + DA3 TSDF code -> empty mesh, and DA3's `sample_points_from_mesh`
  silently substitutes 1M uniform random points in [-1,1]^3. First VGGT eval gave
  F1 0.015 on courtyard because of this. Pinned open3d 0.19.0 (verified: 3 views ->
  86k TSDF points, non-empty mesh) and `evaluate.py` now raises on empty meshes
  instead of scoring random points.
- open3d needs libusb (absent on server): preloaded from the `libusb-package` wheel.
- opencv-python needs libgthread: use opencv-python-headless 4.11.0.86.
- DA3 needs `moviepy.editor` -> moviepy 1.0.3.
- pip download from aliyun ran at ~60 KB/s while curl got 11 MB/s; installed with uv.

## 10/27 First observations (1027-repro, 1027-prelim; partial)

Verified:
- VGGT repro, 10/11 scenes evaluated: means AUC3 ≈ 0.31, AUC30 ≈ 0.84, F1@0.25 m ≈ 0.63
  vs paper 26.3 / 80.8 / 57.2 (11 scenes). Same order, slightly higher; final check
  once all 11 scenes are in.
- F1@0.25 m saturates on indoor scenes (kicker 0.997, office 0.992): the DA3 threshold
  cannot separate methods there -> tighter thresholds needed (Codex task 1).
- COLMAP registration, random subsets: office 3/5, 6/10, 26/26 (763 s); pipes 2/5.
  VGGT/DA3 run 0.3-12 s per case.

Fixes:
- `fscore_vis` first used precision against visible GT-depth points; GT depth only covers
  scanned pixels, so correct predictions outside it counted as wrong (pipes 0.83 -> 0.49).
  Now precision stays whole-mesh, only recall uses visible GT.
- Added `sampling: neighbor` (random anchor + nearest views by camera distance and axis
  angle; uses GT poses only to choose inputs). facade, 5 views: camera spread 1.5 m vs
  8-9 m for random subsets. Random subsets keep their old directory names; neighbor
  subsets are `nb_v<k>_s<seed>`.

## 10/27 Evaluation protocol: what changed and why (verified)

1. Repro check (1027-repro, 11 scenes, all views, DA3 protocol): DA3 48.4/91.1/78.2
   (paper DA3-Giant 48.4/91.2/79.0), VGGT 29.2/81.7/63.2 (paper 26.3/80.8/57.2). Pipeline OK.
2. DA3 GT mesh `combined_mesh.ply` is incomplete: rays through GT-depth pixels of
   courtyard/DSC_0307 hit it only 6% of the time; GT-depth points are 17 cm (median) from it.
   -> Geometry GT for our metrics = GT depth of all views back-projected (stride 2, 5 mm voxels,
   cached in /home/hsy/datasets/da3-bench/eth3d_gtcloud). Self-consistency (one view vs. the
   rest, stride 4): office median 2 mm (100% < 2 cm); courtyard 8 mm (86%), facade 7 mm (92%).
   So 2 cm outdoors is near the GT floor; 5/10 cm are the main outdoor thresholds.
   GT depth is z-depth (ray-length reading is off by ~1 m); the THIN_PRISM_FISHEYE model vs a
   plain pinhole changes the back-projection by ~2 mm only (lens is close to rectilinear).
3. DA3 TSDF fusion is unfair to COLMAP: PatchMatch geometric depth is sparse (10-28% valid
   pixels), TSDF extracts little surface from it (pipes all views: recall 0.07 TSDF vs 0.77
   direct points). Raw depth maps instead carry outliers (pipes: 79% of points < 25 cm).
   -> COLMAP/hybrid are evaluated on their standard output, the stereo-fusion cloud
   (min_num_pixels 2 / 3 / 5 for <=5 / <=10 / more views); VGGT/DA3 on back-projected depth
   resampled to a common 1600 px grid (bilinear, no interpolation across >10% depth jumps).
4. Camera-center Sim(3) alignment (DA3) is ill-conditioned for neighbor subsets (cameras within
   1-2 m, scene depth >10 m): COLMAP courtyard neighbor 10 views scored F@5cm 0.065 < 5 views
   0.242. -> Point-cloud metrics now align on per-pixel correspondences (predicted point vs
   GT-depth point of the same pixel, 20k per view, Umeyama trimmed at 3x median, 5 iters), as
   in point-map evaluations. Inlier fraction ~0.97-0.99 for VGGT/DA3, ~0.72-0.78 for COLMAP
   depth maps (outliers). Pose metrics are relative (alignment-free) and unaffected.
   DA3-protocol columns are kept only for the repro comparison.
5. COLMAP with <3 registered views: pose metrics computed, reconstruction counted as failed.
6. PatchMatch src images now explicitly 20 (COLMAP CLI default; pycolmap passes -1).

## 10/27 Hybrid method, first exploration (1027-hybridtest: office, courtyard x neighbor 5/10, seed 0)

Observed (point-cloud metrics, point alignment):
- COLMAP: most precise (acc median 5-14 mm) but incomplete (comp median 62-450 mm; office
  recall@5cm ~0.2). VGGT/DA3: acc 12-32 mm, comp 8-23 mm, F@5cm 0.74-0.98.
- With neighbor subsets COLMAP registers all views (office 5/5); registration failures in the
  prelim run came from random (low-overlap) subsets.
- hybrid_vggt_ba (FF init + triangulation + BA + PatchMatch + fusion) ~ COLMAP: similar
  precision, same holes. hybrid_vggt_none (no BA): PatchMatch with FF cameras gives few
  consistent points (office 0.01-0.03M fused) -> BA is needed.
- First completion design (fill PatchMatch holes before COLMAP fusion, only near MVS support)
  raised office-5 recall@5cm 0.16 -> 0.23 only: fills far from support were not made and
  COLMAP's 1% depth-consistency check rejected most fills. Replaced by: fused MVS cloud +
  FF depth in its holes (scale + guided-filter residual correction), no 1% check.

## 10/27 Hybrid method: closed (negative result; user decision: not required by the course)

Final exploration (1027-hybridtest, point alignment + similarity ICP to GT for every method):
- ICP refinement (Tanks and Temples style) applied to all methods' final clouds; last-step
  rotation ~0.01 deg, so converged. It mainly helps VGGT/DA3 (courtyard-5 DA3 F@2cm 0.458 -> 0.577).
- F@2cm, DA3 vs best hybrid (DA3 init + BA + PatchMatch + fused cloud + aligned DA3 points in
  its holes): courtyard-5 0.577 / 0.577, courtyard-10 0.509 / 0.470, office-5 0.905 / 0.880,
  office-10 0.823 / 0.799. Guided-filter residual correction hurt in every case.
- Hybrid MVS part is less precise than COLMAP's own (courtyard-5 precision@5cm 0.71 vs 0.85):
  FF init + BA on few SIFT points gives worse cameras than full incremental SfM.
- Why the combination cannot help here: on the surface COLMAP reconstructs, COLMAP vs DA3
  accuracy median is 4.2-6.4 mm vs 6.8-9.5 mm (share < 1 cm 0.78-0.84 vs 0.53-0.71; both 100%
  < 2 cm). The classical advantage is ~3 mm, below the outdoor GT floor (~7 mm), while its
  incompleteness costs most of the F-score.
Code kept (src/run_hybrid.py, src/depth_completion.py) for the report's exploratory section;
not part of the main experiment.

## 10/27 Main sweep launched: 1027-main

11 scenes x {5, 10, 20} views x {neighbor, random} x seeds {0, 1, 2}, seed-major order,
181 cases (20 views skipped for pipes/relief; relief_2 20 = all). COLMAP sharded over both GPUs.

Cleanup 10/27: 1027-prelim keeps raw outputs and info.json only (its COLMAP run predates stereo
fusion and its metrics used camera-center alignment; registration counts quoted in the proposal
come from these info.json files). Stale per-case metrics, TSDF caches and logs removed.
Report: report/proposal.tex and report/report.tex (CVPR template, TinyTeX pdfLaTeX via
report/build.sh). Figures: scripts/make_figures.py; tables: scripts/summarize.py.
scripts/finish_1027_main.sh evaluates the main sweep automatically when it finishes.

## 10/28 Proposal final (collaborators' rewrite) -> follow-up experiments it commits to

Title now "Multi-View 3D Reconstruction from Sparse to Dense Views". The proposal promises:
1. all-views setting for every scene (main sweep has it only for relief_2) -> run COLMAP/VGGT/DA3 on all views;
2. peak GPU memory for all methods -> add COLMAP GPU memory monitoring (nvidia-smi sampling);
3. "end-to-end" runtime = excluding model loading, including image reading -> VGGT timer must include
   load_and_preprocess_images (DA3 timer already includes its own image loading).

## 10/28 Main sweep 1027-main: results (181 cases x 3 methods, results/1027-main/summary.txt)

Fix: evaluate.py icp_sim3 crashed (UnboundLocalError) when the first ICP iteration had <100 pairs;
now keeps the initial alignment in that case.

Verified (means over scenes x seeds, same cases for all methods; F = point-cloud F-score):
- COLMAP registration: neighbor 0.92/0.97/0.96 (5/10/20 views), random 0.43/0.62/0.81; 12 of 33
  random-5 cases fail completely. playground is the weakest scene even with neighbor views (0.70).
- F@5cm neighbor: DA3 0.80/0.80/0.74 > VGGT 0.77/0.72/0.61 > COLMAP 0.37/0.47/0.45.
  random: DA3 0.54/0.69/0.68 > VGGT 0.43/0.53/0.52 > COLMAP 0.11/0.28/0.36.
- COLMAP is the most precise (precision@5cm 0.85 vs DA3 0.75 at neighbor 10/20; accuracy median
  15 mm vs 27-36 mm) but least complete (completeness median 0.21-0.45 m vs 0.015-0.018 m).
- Pose AUC@5 with failures counted, neighbor: COLMAP 0.64/0.70/0.64, DA3 0.68/0.64/0.57,
  VGGT 0.64/0.53/0.41 -> where COLMAP registers, its cameras are competitive or better.
Unexpected, not yet explained:
- Feed-forward F-scores drop with more neighbor views (VGGT F@5cm 0.77 -> 0.61, precision 0.74 ->
  0.54 from 5 to 20 views) and COLMAP does not improve from 10 to 20. Candidate (inference only):
  20-view subsets span a larger area/depth range, so absolute errors grow; check per scene and
  with errors normalized by scene depth before writing anything.
- Runtime in this run: feed-forward = inference only (old timer); end-to-end numbers come from
  1028-efficiency / 1028-allviews.

## 10/28 Follow-up runs started (proposal commitments)
- Measurement: NVML process-level peak GPU memory for all methods (common.GpuMemSampler, includes
  CUDA context and model weights); feed-forward time is end-to-end (image reading included, model
  loading excluded, one warm-up pass); VGGT also logs reading vs inference: reading 6-27 s for
  5-20 views (official loader decodes 24 MP JPEGs sequentially), inference 0.6-1.6 s. COLMAP logs
  match statistics (keypoints, verified pairs, inliers, largest component of the pair graph).
- 1028-allviews (11 scenes, all images), 1028-efficiency (courtyard/office/facade x neighbor
  5/10/20, seed 0), then scripts/match_stats.py on 1027-main for the failure analysis.

## 10/31 Follow-up runs: status (all finished, verified from files)
- 1028-allviews (11 scenes, results/1028-allviews/summary.txt), means: F@5cm DA3 0.765 > VGGT 0.613 >
  COLMAP 0.518; COLMAP reg 0.989, precision@5cm 0.825 (highest), recall@5cm 0.400; pose AUC@5 COLMAP 0.704,
  DA3 0.631, VGGT 0.435; time COLMAP 1154 s, VGGT 29.9 s, DA3 4.8 s (end-to-end).
- 1028-efficiency (courtyard/office/facade x neighbor 5/10/20, seed 0; from info.json, not evaluated for
  geometry): mean end-to-end time COLMAP 118/272/666 s, VGGT 6.0/11.4/24.3 s (mostly image reading),
  DA3 2.1/3.7/6.4 s; max NVML peak GPU memory COLMAP 5.8 GB, VGGT 9.8/11.3/11.8 GB, DA3 11.8/13.7/14.6 GB.
- Match statistics for 1027-main: results/1027-main/match_stats_{0,1}of2.csv (181 cases), not yet analysed.

## 10/31 Analyses for the report (verified) and report draft v1

1. Same views, more context (scripts/nested_views.py -> results/1027-main/nested_views.csv). Neighbor subsets are
   nested (5 ⊂ 10 ⊂ 20 for the same seed; checked 81/81 on the manifest). The 5 shared views are cut out of the
   10/20-view predictions and evaluated like a 5-view case; 24 (scene, seed) pairs with a 20-view run:
   - F@5cm on the same 5 views: DA3 78.5 / 81.1 / 80.5, VGGT 74.0 / 74.7 / 72.2 (5 / 10 / 20 input views)
     -> feed-forward geometry does not degrade with more views; the drop in the main table comes from the
     additional parts of larger subsets (median view-angle 20 -> 57 deg, depth p95 8.9 -> 13.6 m, from GT).
   - AUC@5 on the same 5 views: COLMAP 62.6 / 78.4 / 78.6 (BA + more images registered), DA3 71.4 / 65.9 / 66.1,
     VGGT 62.3 / 62.3 / 48.6.
2. COLMAP failures (match_stats): 13/180 subsets fail completely (12 random-5, 1 playground neighbor-5); 9/13 have a
   disconnected verified-match graph. In random subsets 308 images unregistered, only 59 outside the largest
   connected component -> most unregistered images are matched but not registrable. playground nb_v5_s1: 10/10
   pairs verified, median 987 inliers, "No good initial image pair found" (log). Baseline/depth ratio does not
   predict neighbor failures (checked, no trend).
3. delivery_area all views: COLMAP 40/40 registered but 306/780 verified pairs; camera-center error after Sim(3)
   grows smoothly to 0.8 m along an 11 m path (distortion, not a mis-registered cluster) -> F@5cm 0.17.
4. Pose over COLMAP-registered pairs only (pose_auc05): COLMAP best in every setting (0.62-0.75 vs DA3 0.45-0.68);
   different pair sets per method, stated as such in the report.
5. Indoor (kicker, office, pipes, relief, relief_2, terrains) vs outdoor: median GT depth 2.7 vs 5.7 m;
   neighbor F@5 DA3 0.90/0.66, VGGT 0.82/0.59, COLMAP 0.48/0.37. office: 5.1k SIFT keypoints vs ~10k elsewhere.
6. COLMAP time: PatchMatch 70-91% of the total (1028 runs). COLMAP peak GPU 5.8 GB in every run.

Report: report/report.tex all sections written (v1, 10/31); tables generated by scripts/make_tables.py into
report/tables/, qualitative figure by scripts/make_qualitative.py (courtyard / office, nb_v10_s0). Runtime figure
dropped in favour of tab_efficiency (fig_runtime used the old inference-only timer).

## 11/13 Back to development: DA3-initialised bundle adjustment (DA3-BA), branch dev-v2

Why (user 11/13): the course prefers more workload and some own innovation. Report v2 will be written by me
before the authors edit in Overleaf again.

Hypothesis: COLMAP's cameras are the most accurate where it registers, DA3 never fails; a refinement that
starts from DA3 and uses SIFT tracks should get both. Evidence before building: per-case oracle
max(COLMAP, DA3) is 2.5-8 AUC@5 points above the better method, but a simple rule (COLMAP if all registered,
else DA3) gains only 0-1 points -> needs a real method.

Why the 10/27 hybrid failed on poses (verified): it triangulated SIFT matches with the DA3 cameras; COLMAP's
triangulation keeps only matches within a few px, DA3 pose errors are hundreds of px at 6048 px, so ~300
points per case survived and BA made the cameras worse than DA3 (courtyard nb5 AUC@5 0.96 -> 0.66).

Method (src/run_da3ba.py): SIFT + exhaustive matching (baseline settings) -> tracks from verified inlier
matches -> every observation lifted to 3D with the DA3 depth/pose (median per track, 5% depth tolerance) ->
coarse-to-fine robust BA (Cauchy 16/4/1 px, drop obs > 128/32/8 px) -> COLMAP triangulation with the refined
cameras -> 2 x BA (1 px, drop > 4 px). Views with < 30 observations keep their DA3 pose (Sim(3)-mapped).
Dense outputs: DA3 depth with the refined cameras (da3ba: one global scale; da3ba_ds: per-view scale from
the view's SfM points); --dense: COLMAP PatchMatch + fusion on the refined cameras (hybrid_da3ba) plus
DA3-BA fill of the holes (hybrid_da3ba_fill).

Exploration set (manifests/1113-da3ba-explore.json, 13 cases from 1027-main, fixed before any run):
colmap_fails (6: random subsets with COLMAP reg < 60%, plus playground nb5 s1 init failure and courtyard
random-5 s1 total failure), colmap_better (4: COLMAP AUC@5 >> DA3), da3_good (3: DA3 AUC@5 > 0.75, where
the 10/27 hybrid hurt).

Findings, in order (pose AUC@5 with failures counted, mean over the 13 cases; COLMAP 0.479, DA3 0.581):
1. SIMPLE_RADIAL (COLMAP default) BA: 0.672. Wins where SfM is well conditioned, loses on low-overlap sets and
   on courtyard nb5 (0.96 -> 0.58): one view off by 22-30 deg in translation direction, focal 3.7% short.
   Fixing the intrinsics to DA3's: 0.372, refining focal only: 0.350 -> DA3 focal not accurate enough.
2. Uncertainty gating via BA covariance (pycolmap estimate_ba_covariance, relative pose std): predicted std
   0.02-0.2 deg vs actual errors 1-10 deg, Spearman 0.24 -> errors are systematic, not noise. Dropped.
3. DA3 camera-centre priors in BA (pycolmap pose-prior BA, std 5% of camera spacing): relative poses
   unchanged (only the gauge moved); thousands of pixel residuals dominate. Dropped.
4. Cause (verified): SIMPLE_RADIAL cannot represent the ETH3D lenses. Fit to the GT THIN_PRISM_FISHEYE model
   over the image (scripts/make_camera_model_fig.py, results/camera_model_fit.csv, all 11 scenes):
   SIMPLE_RADIAL RMS 10.9 px / max 61 px / focal -2.2%; RADIAL 1.1 / 7.5 px / -0.25%; OPENCV 0.8 / 5.8 px.
   BA trades this model error for biased focal lengths and poses.
5. RADIAL BA: 0.892 (colmap_better 0.949, colmap_fails 0.807, da3_good 0.987); better than DA3 in 12/13 cases,
   the exception is delivery_area random-20 (0.447 vs 0.471), the least stable case (two SIMPLE_RADIAL
   runs gave 0.435 and 0.246; other cases differ by <= 0.035 between runs: GPU SIFT/matching nondeterminism).
6. Same refinement on VGGT (ff_method: vggt): 0.507 -> 0.869.
7. Geometry without dense matching does not improve: F@2cm DA3 0.500, da3ba 0.485, da3ba_ds 0.501 (DA3's
   per-view depth errors dominate) -> testing the dense stage on the refined cameras.

Fairness consequence: the COLMAP baseline is handicapped by the same model -> colmap_radial (COLMAP defaults,
camera model RADIAL) runs on all 1027-main subsets and all views (configs/1113-main.yaml, 1113-allviews.yaml,
started 11/13 ~16:00 UTC, scripts/launch_1113_colmap_radial.sh). DA3-BA / VGGT-BA sparse stage on the same
cases: scripts/launch_1113_ba.sh.

## 11/14 05:45 OOM re-run (graph-mining GPU conflict)
- A graph-mining job (~36 GB on GPU1 from 03:43) made COLMAP-RADIAL shard 1 hit CUDA OOM in SIFT extraction at 03:54-03:56.
  False failures (0 registered): 1113-main playground random 20 views s2, and 1113-allviews relief_2, pipes, office, electro,
  delivery_area (all views). No OOM lines anywhere else in the shard logs; shard 0 clean.
- Failed outputs moved to runs/_oom_failed_20261114/; finish_1113.sh paused with SIGSTOP (stage-1 main evaluation kept running);
  scripts/rerun_1114_oom_cases.sh re-runs the 6 cases (main on GPU0, allviews on GPU1, logs colmap_radial_rerun_1114.log)
  and sends SIGCONT to finish_1113.sh afterwards, so the allviews evaluation, the dense stage and eval_all use the re-run results.
- The stage-1 main evaluation started before the move, so the stage-1 summary still counts playground v20_s2 as a failure;
  eval_all at the end recomputes it.
- COLMAP-RADIAL / BA sparse runtimes after 11/13 ~17:25 shared the GPUs with graph-mining jobs (wall time possibly inflated;
  peak_gpu_gb is per-process and unaffected).

## 11/15 Results of the 11/13 runs (results/1113-summary.md; verified from metrics.csv)

All sparse methods complete: 180 subsets + 11 all-view cases for colmap_radial, da3ba(_ds), vggtba(_ds); the 6 OOM
cases were re-run and are in the final metrics (e.g. playground random-20 s2: 12/20 registered).
Pose AUC@5 (failures counted), means over all 180 subsets / all views:
- COLMAP 46.6 / 70.4, COLMAP-RADIAL 65.4 / 95.2, VGGT 43.4 / 43.5, VGGT-BA 72.8 / 92.1, DA3 57.4 / 63.1, DA3-BA 81.8 / 95.3.
- Per case, DA3-BA vs DA3: better 152, tie 13, worse 15 (of 180); 11/11 with all views. VGGT-BA vs VGGT 154/15/11.
  DA3-BA vs COLMAP-RADIAL: 103/57/20 on subsets, 3/4/4 with all views (mean +0.1): with dense views the classical
  pipeline with the right camera model is as good; DA3-BA's gain is for sparse / low-overlap inputs.
- Neighbor subsets: DA3-BA 95-96 vs COLMAP-RADIAL 86.5-92.7. Random-5: DA3-BA 49.6 vs DA3 44.7 (smallest gain).
Geometry:
- COLMAP-RADIAL vs COLMAP: F@5 33.0 -> 50.9 (subsets), 51.8 -> 75.6 (all views); precision@5 95%. With all views
  COLMAP-RADIAL has the best F@2 (59.7 vs DA3 51.1) and ties DA3 at F@5 (75.6 vs 76.5): the v1 conclusion "DA3 best
  geometry everywhere" holds only against default COLMAP and must be revised.
- DA3-BA depth: F@2/5 unchanged (47.3/70.8 vs DA3 45.7/70.8 subsets; da3ba_ds 53.3/79.3 vs 51.1/76.5 all views).
  VGGT-BA (global scale) loses F@2 (36.7 -> 31.9); per-view scale (vggtba_ds) recovers it with all views (40.7).
- Dense stage on DA3-BA cameras (negative): hybrid_da3ba (MVS only) ~ COLMAP-RADIAL (all views F@2 59.2 vs 59.7);
  hybrid_da3ba_fill vs DA3 F@2: better 36 / worse 84 of 123 subsets (mean -4.9), 4/7 with all views (-6.7).
  Dense stage incomplete on subsets (124/181): one shard aborted on a COLMAP CHECK (PatchMatch without depth range,
  empty sparse model), the other on an Open3D voxel error in the completion; not re-run (result already negative).
- Runtimes of the 11/13 runs are inflated (GPUs shared with graph-mining jobs from 11/13 ~17:25); use 1028-efficiency
  for runtime, and DA3-BA's own time_s breakdown (features_matching, ba) only as an order of magnitude.
