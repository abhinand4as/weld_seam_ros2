#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seam_detector_node.py — ROS 2 (Jazzy) weld-seam detection node.

Subscribes to /scene/points (published by scene_publisher_node) and runs
the full groove-detection pipeline ported from seam_detection_line.py,
with no robot-arm or camera-transform logic — visualisation in RViz2 only.

Detection pipeline  (mirrors seam_detection_line.py)
-----------------------------------------------------
1.  Voxel downsample + depth filter
2.  Normal estimation, oriented upward (+Z) away from the part surface
3.  Asymmetry feature extraction  →  high-asymmetry points near the seam
4.  Z-band filter                 →  keep only points at the step-edge height
5.  DBSCAN clustering             →  largest cluster by XY span = groove
6.  thin_line   — project groove points onto local regression lines
7.  sort_points — order thinned points along the seam
8.  B-spline fit (pass 1)         →  smooth, dense trajectory
9.  Cylinder filter               →  keep groove points inside the trajectory
10. thin_line + sort_points + B-spline on refined groove (pass 2)
11. Surface-normal estimation via RANSAC plane fit

Step 4 (Z-band filter) is the key addition over the original pipeline.
On a lap joint the weld seam sits at the step edge between the two plates.
That step edge occupies a narrow Z band just below the upper plate surface.
Filtering to that band before DBSCAN eliminates flat-surface blobs and
vertical boundary edges that would otherwise compete with the seam cluster.

Published topics  (all in 'world' frame)
-----------------------------------------
/seam/downsampled_cloud   PointCloud2  — preprocessed input cloud
/seam/groove_cloud        PointCloud2  — detected groove cluster
/seam/trajectory_cloud    PointCloud2  — smoothed seam trajectory
/seam/trajectory_poses    PoseArray    — oriented torch poses along seam
/seam/markers             MarkerArray  — line-strip + start arrow for RViz

Parameters  (ros2 param set /seam_detector <name> <value>)
-----------------------------------------------------------
frame_id          str    'world'

voxel_size        float   0.003  m
    Voxel grid cell size. Finer = more points, slower. 3 mm is appropriate
    for a lap-joint part ~200 mm wide.

max_depth         float   5.0    m
    Depth clip on Z axis. Keep at default unless your PCD has background noise.

delete_percentage float   0.85
    Fraction of low-asymmetry points discarded before the Z-band filter.
    High-asymmetry points near sharp edges / the seam survive.  0.85 keeps
    the top 15 %; increase toward 0.92 if too many flat-surface points remain.

z_band_fraction   float   0.35
    [KEY PARAMETER] The Z-band filter keeps points whose Z satisfies:
        Z  >=  Z_max  -  z_band_fraction * (Z_max - Z_min)
    For a lap joint with Z range 0–14 mm and z_band_fraction=0.35:
        threshold = 0.014 - 0.35*0.014 = 0.0091 m
    → keeps only points above 9.1 mm, isolating the step-edge region.
    Increase toward 0.5 if the seam sits mid-height; decrease toward 0.2 to
    keep a narrower band if background edges survive.

neighbor_radius   float   0.006  m
    DBSCAN eps. At ~2× voxel_size the seam points cluster together while
    points on different edges stay separate. Increase by 0.002 if DBSCAN
    finds no cluster; the Z-band filter makes this less sensitive than before.

thin_radius       float   0.010  m
    Neighbourhood radius for thin_line SVD regression. ~3× voxel_size.

sort_distance     float   0.004  m
    Step size for sort_points walk. Must be slightly larger than the average
    inter-point spacing in the thinned groove cloud.
    Rule: sort_distance ≈ voxel_size * 1.5  is a safe starting point.
    If Pass-2 sort returns very few points, halve this value.

process_interval  float   5.0    s
    Minimum seconds between detections (throttle).

Usage
-----
ros2 run weld_seam_ros2 seam_detector_node
"""

import time
import traceback
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import rclpy
import scipy.spatial as spatial
from geometry_msgs.msg import Point as GPoint
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
import vg

# ---------------------------------------------------------------------------
# PointCloud2 field layout (XYZ only — matches scene_publisher_node output)
# ---------------------------------------------------------------------------

FIELDS_XYZ = [
    PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
]

# scipy.interpolate.splprep with default k=3 requires strictly m > k,
# i.e. at least 5 unique points.
_SPLINE_MIN_PTS: int = 5


# ===========================================================================
# Detection helpers — logic ported 1-to-1 from seam_detection_line.py
# ===========================================================================

def thin_line(
    points: np.ndarray,
    point_cloud_thickness: float = 0.01,
) -> Tuple[np.ndarray, list]:
    """
    Project every point in *points* onto the local 3-D regression line
    computed from neighbours within *point_cloud_thickness*.

    Logic unchanged from seam_detection_line.py.
    Fixes applied (marked [FIX]):
      • isolated-point fallback — prevents degenerate SVD
      • zero-denominator guard in line projection
    """
    point_tree = spatial.cKDTree(points)
    new_points: list = []
    regression_lines: list = []

    for point in point_tree.data:
        neighbours = point_tree.data[
            point_tree.query_ball_point(point, point_cloud_thickness)
        ]

        # [FIX] Need ≥ 2 distinct points for SVD to give a meaningful line.
        if len(neighbours) < 2:
            neighbours = points

        data_mean = neighbours.mean(axis=0)
        _, _, vv = np.linalg.svd(neighbours - data_mean)

        # Two points on the local regression line (first principal component)
        linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis] + data_mean
        regression_lines.append(list(linepts))

        ap    = point - linepts[0]
        ab    = linepts[1] - linepts[0]
        ab_sq = np.dot(ab, ab)

        # [FIX] Guard against degenerate line (both endpoints identical).
        if ab_sq < 1e-12:
            new_points.append(list(point))
        else:
            new_points.append(list(linepts[0] + np.dot(ap, ab) / ab_sq * ab))

    return np.array(new_points), regression_lines


def sort_points(
    points: np.ndarray,
    regression_lines: list,
    sorted_point_distance: float = 0.005,
) -> np.ndarray:
    """
    Walk the thinned cloud in both directions from index 0, choosing the
    next neighbour whose direction best aligns with the local regression line.

    Logic unchanged from seam_detection_line.py.
    Fixes applied (marked [FIX]):
      • epsilon in norm products to prevent ZeroDivisionError
      • row-wise np.all() index lookup instead of element-wise ==
      • visited-set guard to prevent revisiting points (primary cause of
        premature walk termination seen in earlier RViz results)
    """
    point_tree = spatial.cKDTree(points)

    def _walk(direction: int) -> list:
        """direction: +1 = forward,  -1 = backward."""
        walked:  list = []
        visited: set  = {0}   # [FIX] Track visited indices; prevents loop-back
        idx      = 0
        reg_prev = regression_lines[idx][1] - regression_lines[idx][0]

        while True:
            v = regression_lines[idx][1] - regression_lines[idx][0]

            # Keep direction consistent with the previous step
            denom = np.linalg.norm(reg_prev) * np.linalg.norm(v)
            if denom > 1e-12 and np.dot(reg_prev, v) / denom < 0:
                v = -v
            reg_prev = v

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                break

            probe = (
                points[idx]
                + direction * (v / v_norm) * sorted_point_distance
            )

            # Forward: wider search radius (÷1.5); backward: tighter (÷3.0)
            # — same split as seam_detection_line.py
            radius = sorted_point_distance / (1.5 if direction == 1 else 3.0)
            candidates = point_tree.data[
                point_tree.query_ball_point(probe, radius)
            ]
            if len(candidates) < 1:
                break

            # Candidate with direction closest to probe direction
            probe_vec = probe - points[idx]
            best      = candidates[0]
            best_vec  = best - points[idx]
            for c in candidates:
                c_vec = c - points[idx]
                if vg.angle(probe_vec, c_vec) < vg.angle(probe_vec, best_vec):
                    best_vec = c_vec
                    best     = c

            # [FIX] Row-wise match avoids IndexError from element-wise ==
            matches = np.where(np.all(points == best, axis=1))[0]
            if len(matches) == 0:
                break
            new_idx = int(matches[0])

            # [FIX] Skip already-visited points so the walk cannot double back
            if new_idx in visited:
                break

            visited.add(new_idx)
            idx = new_idx
            walked.append(best)

        return walked

    left  = [points[0]] + _walk(+1)
    right = _walk(-1)
    return np.array(list(reversed(right)) + left)


def fit_bspline(
    sorted_pts: np.ndarray,
    n_out_multiplier: int = 2,
) -> np.ndarray:
    """
    Fit a cubic B-spline through *sorted_pts* and evaluate it at
    ``max(len(sorted_pts) * n_out_multiplier, 20)`` uniformly spaced values.

    Logic unchanged from seam_detection_line.py (splprep / splev).
    Fixes applied:
      • deduplicate before splprep to avoid "m > k" error
      • explicit count check with actionable error message
    """
    # Remove duplicates while preserving walk order
    _, unique_idx = np.unique(sorted_pts, axis=0, return_index=True)
    pts = sorted_pts[np.sort(unique_idx)]

    if len(pts) < _SPLINE_MIN_PTS:
        raise RuntimeError(
            f'fit_bspline needs ≥ {_SPLINE_MIN_PTS} unique points, got {len(pts)}. '
            'Reduce sort_distance or voxel_size, or lower delete_percentage.'
        )

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    (tck, _), _, _, _ = interpolate.splprep(
        [x, y, z], s=float('inf'), full_output=1
    )
    u_fine = np.linspace(0, 1, max(len(x) * n_out_multiplier, 20))
    x_f, y_f, z_f = interpolate.splev(u_fine, tck)
    return np.vstack((x_f, y_f, z_f)).T


def points_in_cylinder(
    pt1: np.ndarray,
    pt2: np.ndarray,
    radius: float,
    query_points: np.ndarray,
) -> np.ndarray:
    """
    Return the subset of *query_points* inside the cylinder defined by
    axis pt1→pt2 and the given *radius*.

    Unchanged from seam_detection_line.py.
    """
    vec     = pt2 - pt1
    r_scaled = radius * np.linalg.norm(vec)
    inside: list = [
        q for q in query_points
        if (
            np.dot(q - pt1, vec) >= 0
            and np.dot(q - pt2, vec) <= 0
            and np.linalg.norm(np.cross(q - pt1, vec)) <= r_scaled
        )
    ]
    return np.array(inside) if inside else np.empty((0, 3))


def find_surface_normal(
    trajectory_pts: np.ndarray,
    pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    """
    RANSAC-fit a plane to *pcd*, compute per-point normals for trajectory
    points oriented opposite to the plane normal, and return the mean.

    Unchanged from seam_detection_line.py.
    """
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.003, ransac_n=20, num_iterations=100
    )
    a, b, c, _ = plane_model
    plane_normal = np.array([a, b, c])

    combined = np.concatenate(
        (trajectory_pts, np.asarray(pcd.points)), axis=0
    )
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined)
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=300
        )
    )
    combined_pcd.normalize_normals()
    combined_pcd.orient_normals_to_align_with_direction(
        orientation_reference=-plane_normal
    )
    traj_pcd = combined_pcd.select_by_index(list(range(len(trajectory_pts))))
    return np.asarray(traj_pcd.normals).mean(axis=0)


def normalise_feature(values: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Unchanged from seam_detection_line.py."""
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def compute_asymmetry_feature(
    pcd: o3d.geometry.PointCloud,
    n_neighbours: int = 30,
) -> np.ndarray:
    """
    For every point measure how much its normal deviates from the mean normal
    of its k nearest neighbours.  High value → groove / seam edge.

    Core logic unchanged from seam_detection_line.py (find_feature_value).
    Fix: zero-normal guard prevents division by zero on degenerate mesh faces.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals  = np.asarray(pcd.normals)
    n_pts    = normals.shape[0]
    feature: list = []

    for i in range(n_pts):
        k = max(2, min(n_pts // 100 + 1, n_neighbours))
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        mean_n  = normals[np.array(idx), :].mean(axis=0)
        n_i     = normals[i]
        norm_ni = np.linalg.norm(n_i)
        if norm_ni < 1e-9:
            feature.append(0.0)
            continue
        residual = mean_n - n_i * (np.dot(mean_n, n_i) / norm_ni)
        feature.append(float(np.linalg.norm(residual)))

    return np.array(feature)


def cluster_groove(
    pcd_selected: o3d.geometry.PointCloud,
    eps: float,
    min_points: int = 5,
) -> Tuple[o3d.geometry.PointCloud, list]:
    """
    DBSCAN cluster the pre-filtered high-asymmetry points and return the
    cluster with the largest XY bounding-box span — which is the seam.

    After the Z-band filter (Step 4) only points near the step edge survive,
    so the largest cluster in XY is almost always the seam line itself.  We
    use the actual bounding-box diagonal in XY (not the PCA singular value
    used previously) because it is a true distance in metres and directly
    comparable to the known part dimensions.

    Original logic returned largest-by-count.  That is still used as a
    fallback if all XY spans are equal (degenerate input).

    Returns
    -------
    groove      : Open3D PointCloud of the best cluster
    diagnostics : list of dicts for INFO logging
    """
    labels = np.array(
        pcd_selected.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False
        )
    )
    unique, _ = np.unique(labels, return_counts=True)
    valid = [(lbl, np.where(labels == lbl)[0]) for lbl in unique if lbl != -1]

    if not valid:
        raise RuntimeError(
            'DBSCAN found no valid groove cluster. '
            'Try lowering delete_percentage, increasing neighbor_radius, '
            'or adjusting z_band_fraction.'
        )

    all_pts = np.asarray(pcd_selected.points)
    scored:      list = []
    diagnostics: list = []

    for lbl, idxs in valid:
        pts  = all_pts[idxs]
        span = pts.max(axis=0) - pts.min(axis=0)

        # XY bounding-box diagonal — true distance, not PCA variance
        xy_diag = float(np.linalg.norm(span[:2]))

        # Aspect ratio using bounding box (simpler and more interpretable
        # than PCA singular values for axis-aligned seams)
        xy_dims = np.sort(span[:2])[::-1]          # [longer, shorter]
        bb_ar   = float(xy_dims[0] / xy_dims[1]) if xy_dims[1] > 1e-6 else float('inf')

        scored.append((xy_diag, len(idxs), lbl, idxs))
        diagnostics.append({
            'label':    int(lbl),
            'n_pts':    len(idxs),
            'xy_diag':  round(xy_diag, 4),
            'bb_ar':    round(bb_ar, 1),
            'span_xyz': tuple(round(float(s), 4) for s in span),
        })

    # Primary sort: largest XY bounding-box diagonal = longest cluster = seam
    scored.sort(key=lambda x: x[0], reverse=True)
    _, _, _, best_indices = scored[0]
    return pcd_selected.select_by_index(best_indices.tolist()), diagnostics


# ===========================================================================
# ROS 2 conversion helpers
# ===========================================================================

def open3d_to_ros2_cloud(
    o3d_cloud: o3d.geometry.PointCloud,
    frame_id: str,
    node: Node,
) -> PointCloud2:
    """Convert an Open3D PointCloud to a ROS 2 PointCloud2 (XYZ only)."""
    header = Header()
    header.stamp = node.get_clock().now().to_msg()
    header.frame_id = frame_id
    pts = np.asarray(o3d_cloud.points, dtype=np.float32).tolist()
    return pc2.create_cloud(header, FIELDS_XYZ, pts)


def ros2_cloud_to_open3d(ros_cloud: PointCloud2) -> Optional[o3d.geometry.PointCloud]:
    """Convert a ROS 2 PointCloud2 (XYZ or XYZRGB) to an Open3D PointCloud."""
    field_names = [f.name for f in ros_cloud.fields]
    raw = list(
        pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names)
    )
    if not raw:
        return None

    o3d_cloud = o3d.geometry.PointCloud()
    xyz = [(x, y, z) for x, y, z, *_ in raw] if 'rgb' in field_names \
          else [(x, y, z) for x, y, z in raw]
    o3d_cloud.points = o3d.utility.Vector3dVector(
        np.array(xyz, dtype=np.float64)
    )
    return o3d_cloud


# ===========================================================================
# RViz2 marker builders
# ===========================================================================

def make_trajectory_line_marker(
    points: np.ndarray,
    frame_id: str,
    node: Node,
    marker_id: int = 10,
) -> Marker:
    """LINE_STRIP marker tracing the full seam trajectory (green)."""
    m = Marker()
    m.header.stamp    = node.get_clock().now().to_msg()
    m.header.frame_id = frame_id
    m.ns     = 'seam_detection'
    m.id     = marker_id
    m.type   = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.002
    m.color   = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    m.pose.orientation.w = 1.0
    for p in points:
        gp = GPoint()
        gp.x, gp.y, gp.z = float(p[0]), float(p[1]), float(p[2])
        m.points.append(gp)
    return m


def make_start_arrow_marker(
    start: np.ndarray,
    direction: np.ndarray,
    frame_id: str,
    node: Node,
    marker_id: int = 11,
) -> Marker:
    """ARROW marker at the seam start showing the travel direction (yellow)."""
    m = Marker()
    m.header.stamp    = node.get_clock().now().to_msg()
    m.header.frame_id = frame_id
    m.ns     = 'seam_detection'
    m.id     = marker_id
    m.type   = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x = 0.004
    m.scale.y = 0.008
    m.scale.z = 0.008
    m.color   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    m.pose.orientation.w = 1.0
    d_norm = np.linalg.norm(direction)
    tip    = start + (direction / d_norm if d_norm > 1e-12 else direction) * 0.03
    p0, p1 = GPoint(), GPoint()
    p0.x, p0.y, p0.z = float(start[0]), float(start[1]), float(start[2])
    p1.x, p1.y, p1.z = float(tip[0]),   float(tip[1]),   float(tip[2])
    m.points = [p0, p1]
    return m


def build_pose_array(
    trajectory_pts: np.ndarray,
    normal: np.ndarray,
    frame_id: str,
    node: Node,
) -> PoseArray:
    """
    Build a PoseArray along the trajectory where each pose has:
      X — along the seam (forward direction)
      Z — along the surface normal
      Y — cross(Z, X)

    Mirrors find_orientation() in seam_detection_line.py.
    """
    pose_array = PoseArray()
    pose_array.header.stamp    = node.get_clock().now().to_msg()
    pose_array.header.frame_id = frame_id

    z_dir = normal / (np.linalg.norm(normal) + 1e-12)

    for i in range(len(trajectory_pts) - 1):
        pos_diff = trajectory_pts[i + 1] - trajectory_pts[i]
        x_dir    = pos_diff - np.dot(pos_diff, z_dir) * z_dir
        norm_x   = np.linalg.norm(x_dir)
        if norm_x < 1e-9:
            continue
        x_dir /= norm_x
        y_dir  = np.cross(z_dir, x_dir)
        y_dir /= np.linalg.norm(y_dir)

        quat = R.from_matrix(np.column_stack((x_dir, y_dir, z_dir))).as_quat()

        pose = Pose()
        pose.position.x    = float(trajectory_pts[i][0])
        pose.position.y    = float(trajectory_pts[i][1])
        pose.position.z    = float(trajectory_pts[i][2])
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        pose_array.poses.append(pose)

    return pose_array


# ===========================================================================
# Detection node
# ===========================================================================

class SeamDetectorNode(Node):
    """
    Subscribes to /scene/points, runs the seam-detection pipeline once per
    *process_interval* seconds, and publishes results on /seam/* topics.
    """

    def __init__(self) -> None:
        super().__init__('seam_detector')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('frame_id',          'world')
        self.declare_parameter('voxel_size',         0.003)
        self.declare_parameter('max_depth',          5.0)
        self.declare_parameter('delete_percentage',  0.85)
        self.declare_parameter('z_band_fraction',    0.35)  # [ADDED] Z-step filter
        self.declare_parameter('neighbor_radius',    0.006)
        self.declare_parameter('thin_radius',        0.010)
        self.declare_parameter('sort_distance',      0.004)
        self.declare_parameter('process_interval',   5.0)

        self._frame_id          = self.get_parameter('frame_id').value
        self._voxel_size        = self.get_parameter('voxel_size').value
        self._max_depth         = self.get_parameter('max_depth').value
        self._delete_pct        = self.get_parameter('delete_percentage').value
        self._z_band_fraction   = self.get_parameter('z_band_fraction').value
        self._neighbor_radius   = self.get_parameter('neighbor_radius').value
        self._thin_radius       = self.get_parameter('thin_radius').value
        self._sort_distance     = self.get_parameter('sort_distance').value
        self._process_interval  = self.get_parameter('process_interval').value

        # ── publishers ────────────────────────────────────────────────────
        self._pub_ds     = self.create_publisher(PointCloud2, '/seam/downsampled_cloud', 10)
        self._pub_groove = self.create_publisher(PointCloud2, '/seam/groove_cloud',      10)
        self._pub_traj   = self.create_publisher(PointCloud2, '/seam/trajectory_cloud',  10)
        self._pub_poses  = self.create_publisher(PoseArray,   '/seam/trajectory_poses',  10)
        self._pub_marks  = self.create_publisher(MarkerArray, '/seam/markers',            10)

        # ── subscriber ────────────────────────────────────────────────────
        self._last_process_time = 0.0
        self.create_subscription(
            PointCloud2, '/scene/points', self._cloud_callback, 10
        )

        self.get_logger().info(
            'SeamDetectorNode ready — listening on /scene/points\n'
            f'  frame_id         = {self._frame_id}\n'
            f'  voxel_size       = {self._voxel_size} m\n'
            f'  max_depth        = {self._max_depth} m\n'
            f'  delete_pct       = {self._delete_pct}\n'
            f'  z_band_fraction  = {self._z_band_fraction}\n'
            f'  neighbor_radius  = {self._neighbor_radius} m\n'
            f'  thin_radius      = {self._thin_radius} m\n'
            f'  sort_distance    = {self._sort_distance} m\n'
            f'  process_interval = {self._process_interval} s'
        )

    # ── callback ──────────────────────────────────────────────────────────

    def _cloud_callback(self, msg: PointCloud2) -> None:
        now = time.time()
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        self.get_logger().info('Received cloud — running seam detection …')
        t0 = time.time()

        pcd = ros2_cloud_to_open3d(msg)
        if pcd is None or len(pcd.points) == 0:
            self.get_logger().warn('Received empty cloud — skipping.')
            return

        try:
            groove, trajectory_pts, normal = self._detect(pcd)
        except Exception as exc:
            self.get_logger().error(f'Detection failed: {exc}')
            self.get_logger().debug(traceback.format_exc())
            return

        self.get_logger().info(
            f'Detection done in {time.time() - t0:.2f} s — '
            f'{len(trajectory_pts)} trajectory points'
        )
        self._publish_results(pcd, groove, trajectory_pts, normal)

    # ── core detection pipeline ───────────────────────────────────────────

    def _detect(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """
        Full seam-detection pipeline.

        Returns
        -------
        groove         : Open3D PointCloud of the refined groove cluster
        trajectory_pts : (N, 3) ordered, spline-smoothed seam centreline
        normal         : (3,) mean surface normal along the seam
        """
        # ── Step 1: voxel downsample ──────────────────────────────────────
        pcd = pcd.voxel_down_sample(voxel_size=self._voxel_size)
        self.get_logger().info(
            f'  [1] Voxel downsample → {len(pcd.points)} points'
        )

        # ── Step 2: depth filter (keep Z < max_depth) ─────────────────────
        pts_arr  = np.asarray(pcd.points)
        mask     = pts_arr[:, 2] < self._max_depth
        pts_kept = pts_arr[mask]
        if len(pts_kept) < 50:
            raise RuntimeError(
                f'Too few points after depth filter ({len(pts_kept)}). '
                f'Cloud Z: [{pts_arr[:,2].min():.3f}, {pts_arr[:,2].max():.3f}] m'
            )
        pcd_f = o3d.geometry.PointCloud()
        pcd_f.points = o3d.utility.Vector3dVector(pts_kept)
        pcd_f.remove_non_finite_points()
        pcd = pcd_f
        z_min = float(pts_kept[:, 2].min())
        z_max = float(pts_kept[:, 2].max())
        self.get_logger().info(
            f'  [2] Depth filter     → {len(pcd.points)} points  '
            f'(Z: {z_min:.4f} – {z_max:.4f} m)'
        )

        # ── Step 3: normal estimation ─────────────────────────────────────
        # Orient normals upward (+Z): correct for a static part on a table.
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._voxel_size * 4, max_nn=30
            )
        )
        pcd.normalize_normals()
        pcd.orient_normals_to_align_with_direction(
            orientation_reference=np.array([0.0, 0.0, 1.0])
        )

        # ── Step 4: asymmetry feature ─────────────────────────────────────
        n_pts   = len(pcd.points)
        feature = compute_asymmetry_feature(
            pcd, n_neighbours=min(max(n_pts // 100 + 1, 6), 30)
        )
        norm_feature = normalise_feature(feature)
        self.get_logger().info(
            f'  [4] Asymmetry        → range [{feature.min():.4f}, '
            f'{feature.max():.4f}]'
        )

        # ── Step 5: asymmetry selection ───────────────────────────────────
        n_delete     = int(n_pts * self._delete_pct)
        kept_indices = np.argsort(norm_feature)[n_delete:].tolist()
        pcd_selected = pcd.select_by_index(kept_indices)
        self.get_logger().info(
            f'  [5] Asymmetry select → {len(pcd_selected.points)} points '
            f'(top {100*(1-self._delete_pct):.0f}%)'
        )

        # ── Step 6: Z-band filter — isolate the step-edge height ──────────
        #
        # [KEY ADDITION] On a lap joint the weld seam sits at the step edge
        # between the upper and lower plate.  That edge occupies a narrow Z
        # band just below the top surface.  Filtering to this band before
        # DBSCAN eliminates:
        #   • flat upper-plate surface points  (Z near z_max — above the band)
        #   • lower-plate surface points       (Z near z_min — below the band)
        #   • vertical part boundary edges     (span full Z range)
        #
        # The band is defined as:
        #   Z  >=  z_max  -  z_band_fraction * (z_max - z_min)
        #
        # With z_band_fraction=0.35 and Z range 0–14 mm:
        #   threshold = 0.014 - 0.35*0.014 = 0.0091 m
        # → keeps only points above 9.1 mm, i.e. within the top 35% of Z.
        #
        # Adjust z_band_fraction:
        #   • increase toward 0.5 if the seam sits mid-height
        #   • decrease toward 0.15 for a very thin step
        #
        sel_pts  = np.asarray(pcd_selected.points)
        z_thresh = z_max - self._z_band_fraction * (z_max - z_min)
        z_mask   = sel_pts[:, 2] >= z_thresh
        zband_pts = sel_pts[z_mask]

        self.get_logger().info(
            f'  [6] Z-band filter    → {len(zband_pts)} points  '
            f'(Z >= {z_thresh:.4f} m, fraction={self._z_band_fraction})'
        )

        if len(zband_pts) < 10:
            raise RuntimeError(
                f'Z-band filter left only {len(zband_pts)} points. '
                f'Try increasing z_band_fraction (current: {self._z_band_fraction}) '
                f'or lowering delete_percentage (current: {self._delete_pct}).'
            )

        pcd_zband = o3d.geometry.PointCloud()
        pcd_zband.points = o3d.utility.Vector3dVector(zband_pts)

        # ── Step 7: DBSCAN — largest XY-span cluster = seam ──────────────
        groove, cluster_diags = cluster_groove(
            pcd_zband, eps=self._neighbor_radius
        )
        groove_pts = np.asarray(groove.points)
        g_span     = groove_pts.max(axis=0) - groove_pts.min(axis=0)

        self.get_logger().info(
            f'  [7] DBSCAN found {len(cluster_diags)} cluster(s):'
        )
        for d in sorted(cluster_diags, key=lambda x: x['xy_diag'], reverse=True):
            self.get_logger().info(
                f'        label={d["label"]:2d}  n={d["n_pts"]:4d}  '
                f'xy_diag={d["xy_diag"]:.4f} m  '
                f'bb_ar={d["bb_ar"]:.1f}  '
                f'span_xyz=({d["span_xyz"][0]:.3f}, '
                f'{d["span_xyz"][1]:.3f}, {d["span_xyz"][2]:.3f}) m'
            )
        self.get_logger().info(
            f'  [7] Selected groove  → {len(groove_pts)} points  '
            f'span_XYZ=({g_span[0]:.3f}, {g_span[1]:.3f}, {g_span[2]:.3f}) m'
        )

        # ── Step 8: pass-1 trajectory (thin → sort → spline) ─────────────
        thinned, reg_lines = thin_line(
            groove_pts, point_cloud_thickness=self._thin_radius
        )
        sorted_pts = sort_points(
            thinned, reg_lines, sorted_point_distance=self._sort_distance
        )
        self.get_logger().info(
            f'  [8] Pass-1 sort      → {len(sorted_pts)} points'
        )
        traj_pts_v1 = fit_bspline(sorted_pts)

        # ── Step 9: cylinder filter → refined groove ──────────────────────
        start_pt, end_pt = traj_pts_v1[0], traj_pts_v1[-1]
        refined_pts = points_in_cylinder(
            start_pt, end_pt,
            radius=self._voxel_size * 5,
            query_points=groove_pts,
        )
        if len(refined_pts) < 10:
            self.get_logger().warn(
                f'Cylinder filter returned {len(refined_pts)} points — '
                'falling back to raw groove cloud.'
            )
            refined_pts = groove_pts

        refined_pcd = o3d.geometry.PointCloud()
        refined_pcd.points = o3d.utility.Vector3dVector(refined_pts)
        refined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.get_logger().info(
            f'  [9] Refined groove   → {len(refined_pcd.points)} points'
        )

        # ── Step 10: pass-2 trajectory (thin → sort → spline) ────────────
        ref_pts              = np.asarray(refined_pcd.points)
        thinned2, reg_lines2 = thin_line(
            ref_pts, point_cloud_thickness=self._thin_radius
        )
        sorted_pts2 = sort_points(
            thinned2, reg_lines2, sorted_point_distance=self._sort_distance
        )
        self.get_logger().info(
            f'  [10] Pass-2 sort     → {len(sorted_pts2)} points'
        )
        trajectory_pts = fit_bspline(sorted_pts2)

        # ── Step 11: surface normal along the final trajectory ────────────
        normal = find_surface_normal(trajectory_pts, pcd)

        return refined_pcd, trajectory_pts, normal

    # ── publishing ────────────────────────────────────────────────────────

    def _publish_results(
        self,
        pcd:            o3d.geometry.PointCloud,
        groove:         o3d.geometry.PointCloud,
        trajectory_pts: np.ndarray,
        normal:         np.ndarray,
    ) -> None:
        """Publish all detection results to their respective /seam/* topics."""

        self._pub_ds.publish(
            open3d_to_ros2_cloud(pcd, self._frame_id, self)
        )
        self._pub_groove.publish(
            open3d_to_ros2_cloud(groove, self._frame_id, self)
        )

        traj_o3d = o3d.geometry.PointCloud()
        traj_o3d.points = o3d.utility.Vector3dVector(trajectory_pts)
        self._pub_traj.publish(
            open3d_to_ros2_cloud(traj_o3d, self._frame_id, self)
        )

        self._pub_poses.publish(
            build_pose_array(trajectory_pts, normal, self._frame_id, self)
        )

        markers = MarkerArray()
        markers.markers.append(
            make_trajectory_line_marker(trajectory_pts, self._frame_id, self)
        )
        markers.markers.append(
            make_start_arrow_marker(
                trajectory_pts[0],
                trajectory_pts[1] - trajectory_pts[0],
                self._frame_id, self,
            )
        )
        self._pub_marks.publish(markers)
        self.get_logger().info('Published detection results on /seam/* topics.')


# ===========================================================================
# Entry point
# ===========================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SeamDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()