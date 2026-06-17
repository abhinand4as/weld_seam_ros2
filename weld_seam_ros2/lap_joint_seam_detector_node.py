#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lap_joint_seam_detector_node.py — ROS 2 (Jazzy) lap-joint weld-seam detector.

Exploits the bimodal Z distribution of a lap joint (two flat plates at
different heights) to locate the seam without surface normals or asymmetry
features. The algorithm is robust to sensor noise on flat surfaces — where
normal-deviation methods typically fail.

Algorithm
---------
1. Voxel downsample the input cloud; clip points above max_depth.
2. Split into upper / lower height groups via 1-D two-means clustering on Z.
3. For each group, retain points whose XY position lies within edge_radius
   of any point in the opposite group — these are step-edge candidates.
4. DBSCAN on the candidates; keep the largest valid cluster as the seam.

Which plate's edge is published is controlled by the seam_group parameter:
  'lower'  — lower-plate surface at the foot of the step (default).
             This is the correct torch path for a lap-joint fillet weld.
  'upper'  — upper-plate edge only.
  'both'   — union of both edges (useful for visualisation / debugging).

Published topics
----------------
/seam/downsampled_cloud   sensor_msgs/PointCloud2   Filtered input cloud
/seam/groove_cloud        sensor_msgs/PointCloud2   Detected seam cluster

Parameters  (ros2 param set /seam_detector <name> <value>)
----------------------------------------------------------
frame_id          str    'world'     Fixed frame for all published messages
voxel_size        float   0.002  m   Voxel grid spacing for downsampling
max_depth         float   0.7    m   Z clip — discard points above this height
edge_radius       float   0.003  m   XY distance threshold for step-edge detection.
                                     Rule of thumb: ~ voxel_size for a clean seam.
                                     Increase if no candidates are found;
                                     decrease if the whole plate surface is selected.
min_cluster_pts   int     10         DBSCAN minimum cluster size (noise rejection)
seam_group        str    'lower'     Which height group to return (lower/upper/both)
process_interval  float   5.0    s   Minimum seconds between successive detections
verbose           bool    False       Log per-step point counts for parameter tuning

Usage
-----
ros2 launch weld_seam_ros2 detector.launch.py detector:=lap_joint
"""

import time
from typing import Tuple

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header


# XYZ-only PointCloud2 field layout — matches scene_publisher_node output.
_FIELDS_XYZ = [
    PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _ros_to_open3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """Convert a ROS 2 PointCloud2 to an Open3D PointCloud (XYZ only)."""
    pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=-1).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def _open3d_to_ros(pcd: o3d.geometry.PointCloud, header: Header) -> PointCloud2:
    """Convert an Open3D PointCloud to a ROS 2 PointCloud2 (XYZ only)."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pc2.create_cloud(header, _FIELDS_XYZ, pts.tolist())


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _two_means_1d(values: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """
    One-dimensional two-means clustering.

    Iteratively assigns each value to the nearer of two centroids until
    convergence. Initialises centroids at the global min and max so the
    algorithm always separates the two extreme modes — appropriate for
    a lap joint whose Z distribution has two well-separated peaks.

    Parameters
    ----------
    values   : 1-D array of scalar values (Z coordinates).
    max_iter : Maximum refinement iterations before declaring convergence.

    Returns
    -------
    mask : Boolean array; True marks the upper (higher-Z) cluster.
    """
    c_low, c_high = values.min(), values.max()
    mask = np.zeros(values.shape, dtype=bool)

    for _ in range(max_iter):
        new_mask = np.abs(values - c_high) < np.abs(values - c_low)
        if new_mask.sum() == 0 or (~new_mask).sum() == 0:
            break
        if np.array_equal(new_mask, mask):
            break
        mask   = new_mask
        c_low  = values[~mask].mean()
        c_high = values[ mask].mean()

    return mask


def _largest_dbscan_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> o3d.geometry.PointCloud:
    """
    Run DBSCAN on *pcd* and return the largest non-noise cluster.

    Noise points (label == -1) are always excluded. If every point is
    classified as noise the largest label by count is returned as a
    last resort so the caller always gets a non-empty cloud.
    """
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    best  = next((values[i] for i in order if values[i] != -1), values[order[0]])
    return pcd.select_by_index(np.where(labels == best)[0].tolist())


def _detect_seam(
    pcd: o3d.geometry.PointCloud,
    edge_radius: float,
    min_cluster_pts: int,
    seam_group: str,
) -> Tuple[o3d.geometry.PointCloud, int, int, int]:
    """
    Core detection: height split → XY proximity filter → DBSCAN.

    Parameters
    ----------
    pcd            : Pre-filtered (downsampled, depth-clipped) input cloud.
    edge_radius    : XY distance threshold for step-edge candidates.
    min_cluster_pts: DBSCAN minimum cluster size.
    seam_group     : 'lower', 'upper', or 'both' — see module docstring.

    Returns
    -------
    seam_pcd    : Detected seam cluster as an Open3D PointCloud.
    n_upper     : Number of points in the upper height group.
    n_lower     : Number of points in the lower height group.
    n_candidates: Seam candidate count before DBSCAN.
    """
    points = np.asarray(pcd.points)

    upper_mask = _two_means_1d(points[:, 2])
    upper = points[ upper_mask]
    lower = points[~upper_mask]

    if len(upper) == 0 or len(lower) == 0:
        raise RuntimeError(
            'Z distribution appears unimodal — cannot split into two height '
            'groups. Check that the cloud contains both plates of the lap joint.'
        )

    # Project both groups onto the XY plane and find step-edge candidates by
    # cross-group proximity. Points far from the step have no opposite-group
    # neighbour within edge_radius; points at the step do.
    dist_upper_to_lower, _ = cKDTree(lower[:, :2]).query(upper[:, :2])
    dist_lower_to_upper, _ = cKDTree(upper[:, :2]).query(lower[:, :2])

    upper_edge = upper[dist_upper_to_lower < edge_radius]
    lower_edge = lower[dist_lower_to_upper < edge_radius]

    group_map = {
        'lower': [lower_edge],
        'upper': [upper_edge],
        'both':  [upper_edge, lower_edge],
    }
    parts = [p for p in group_map.get(seam_group, [lower_edge]) if len(p) > 0]

    if not parts:
        raise RuntimeError(
            f'No seam candidates found for seam_group={seam_group!r}. '
            'Try increasing edge_radius.'
        )

    candidates = np.vstack(parts)
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidates)

    seam_pcd = _largest_dbscan_cluster(
        candidate_pcd, eps=4.0 * edge_radius, min_points=min_cluster_pts
    )
    return seam_pcd, len(upper), len(lower), len(candidates)


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class LapJointSeamDetectorNode(Node):
    """
    Subscribes to /scene/points and periodically runs the lap-joint seam
    detection pipeline, publishing the downsampled cloud and the detected
    seam cluster for visualisation in RViz2.

    Detection is throttled to once every process_interval seconds so that
    continuous publishers (e.g. scene_publisher_node at 1 Hz) do not cause
    the compute-heavy pipeline to run on every message.
    """

    def __init__(self) -> None:
        super().__init__('seam_detector')

        # ── parameters ──────────────────────────────────────────────────────
        self.declare_parameter('frame_id',          'world')
        self.declare_parameter('voxel_size',         0.002)
        self.declare_parameter('max_depth',          0.7)
        self.declare_parameter('edge_radius',        0.003)
        self.declare_parameter('min_cluster_pts',    10)
        self.declare_parameter('seam_group',        'lower')
        self.declare_parameter('process_interval',   5.0)
        self.declare_parameter('verbose',            False)

        # ── publishers ───────────────────────────────────────────────────────
        self._pub_ds   = self.create_publisher(PointCloud2, '/seam/downsampled_cloud', 10)
        self._pub_seam = self.create_publisher(PointCloud2, '/seam/groove_cloud',      10)

        # ── subscriber ───────────────────────────────────────────────────────
        self._last_process_time = 0.0
        self.create_subscription(PointCloud2, '/scene/points', self._cloud_callback, 10)

        self.get_logger().info(
            'LapJointSeamDetectorNode ready — listening on /scene/points\n'
            f'  voxel={self.get_parameter("voxel_size").value} m  '
            f'edge_radius={self.get_parameter("edge_radius").value} m  '
            f'seam_group={self.get_parameter("seam_group").value!r}'
        )

    # ── callback ─────────────────────────────────────────────────────────────

    def _cloud_callback(self, msg: PointCloud2) -> None:
        interval = self.get_parameter('process_interval').value
        now = time.time()
        if now - self._last_process_time < interval:
            return
        self._last_process_time = now

        self.get_logger().info('Received cloud — running lap-joint seam detection …')
        t0 = time.time()

        try:
            ds_pcd, seam_pcd = self._pipeline(msg)
        except Exception as exc:
            self.get_logger().error(f'Detection failed: {exc}')
            return

        self.get_logger().info(
            f'Detection done in {time.time() - t0:.2f} s — '
            f'{len(seam_pcd.points)} seam points'
        )
        self._publish(msg.header.frame_id, ds_pcd, seam_pcd)

    # ── pipeline ─────────────────────────────────────────────────────────────

    def _pipeline(
        self, msg: PointCloud2
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Preprocess the cloud and run core detection."""
        voxel_size   = self.get_parameter('voxel_size').value
        max_depth    = self.get_parameter('max_depth').value
        edge_radius  = self.get_parameter('edge_radius').value
        min_cluster  = self.get_parameter('min_cluster_pts').value
        seam_group   = self.get_parameter('seam_group').value
        verbose      = self.get_parameter('verbose').value

        # Step 1 — convert and sanitise
        pcd = _ros_to_open3d(msg)
        pcd.remove_non_finite_points()

        # Step 2 — depth clip (Z in world frame = height above table)
        pts = np.asarray(pcd.points)
        pts = pts[pts[:, 2] < max_depth]
        if len(pts) == 0:
            raise RuntimeError('No points remain after depth filter (max_depth too low?).')
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Step 3 — voxel downsample
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd.points) == 0:
            raise RuntimeError('No points remain after voxel downsampling.')

        # Step 4 — height-split + edge proximity + DBSCAN
        seam_pcd, n_upper, n_lower, n_cand = _detect_seam(
            pcd, edge_radius, min_cluster, seam_group
        )

        if verbose:
            self.get_logger().info(
                f'  {len(pcd.points)} pts → {n_upper}/{n_lower} upper/lower '
                f'→ {n_cand} candidates → {len(seam_pcd.points)} seam pts'
            )

        return pcd, seam_pcd

    # ── publish ───────────────────────────────────────────────────────────────

    def _publish(
        self,
        source_frame: str,
        ds_pcd: o3d.geometry.PointCloud,
        seam_pcd: o3d.geometry.PointCloud,
    ) -> None:
        frame_id = self.get_parameter('frame_id').value
        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        self._pub_ds.publish(_open3d_to_ros(ds_pcd, header))
        self._pub_seam.publish(_open3d_to_ros(seam_pcd, header))

        self.get_logger().info('Published seam detection results.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LapJointSeamDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
