# weld_seam_ros2

Publishes a static weld inspection scene for RViz2: a table, the
lap-joint test part as a mesh sitting on the table, and that part's
captured point cloud overlapping the mesh in the same fixed frame.

## Layout

| Path | Contents |
|---|---|
| `weld_seam_ros2/scene_publisher_node.py` | Node that publishes the table marker, the part mesh marker and the point cloud |
| `launch/view_scene.launch.py` | Launches the node + RViz2 with `rviz/scene.rviz` |
| `rviz/scene.rviz` | RViz2 config: Grid, MarkerArray, PointCloud2 |
| `assets/` | 3D assets: `lap_joint_test.stl` (mesh) and the captured point clouds (`.pcd`) |
| `reference/` | Background reading |

## Dependencies

```bash
sudo apt install ros-jazzy-sensor-msgs-py ros-jazzy-rviz2
pip install open3d numpy
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select weld_seam_ros2
source install/setup.bash
```

## Run

```bash
ros2 launch weld_seam_ros2 view_scene.launch.py
```

This opens RViz2 (Fixed Frame `world`) showing:
- a generated table (gray-brown box, top surface at `Z=0`)
- `assets/lap_joint_test.stl` resting on the table
- `assets/lap_joint_base_frame.pcd` published on `/scene/points`, overlapping the mesh

## Topics

| Topic | Type | Description |
|---|---|---|
| `/scene/markers` | `visualization_msgs/MarkerArray` | Table (CUBE) + part mesh (MESH_RESOURCE) |
| `/scene/points` | `sensor_msgs/PointCloud2` | Part point cloud, same frame as the markers |

## Parameters (`scene_publisher` node)

| Parameter | Default | Description |
|---|---|---|
| `frame_id` | `world` | Fixed frame for all published data |
| `mesh_path` | `assets/lap_joint_test.stl` | STL mesh of the part |
| `pcd_path` | `assets/lap_joint_base_frame.pcd` | Point cloud to overlay on the mesh |
| `table_width` | `0.6` | Table size along X (m) |
| `table_depth` | `0.6` | Table size along Y (m) |
| `table_height` | `0.05` | Table thickness (m); top surface is always at `Z=0` |
| `publish_rate` | `2.0` | Publish rate (Hz) |

The part mesh and the default point cloud already share the same local
frame with the part's bottom face at `Z=0`, so keeping the table top at
`Z=0` makes the cloud line up with the mesh without any extra offset.

To try the alternate captured cloud that includes the table surface in
the scan itself, run:

```bash
ros2 launch weld_seam_ros2 view_scene.launch.py \
  --ros-args -p pcd_path:=<share_dir>/assets/lap_joint_with_table.pcd
```

(launch arguments aren't wired up yet — for now, override the node directly
with `ros2 run weld_seam_ros2 scene_publisher --ros-args -p pcd_path:=...`)
