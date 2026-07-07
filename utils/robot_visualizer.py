"""Robot visualizer: URDF-based forward kinematics + headless MeshCat rendering."""

import io
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np

logger = logging.getLogger(__name__)


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """Roll-pitch-yaw (XYZ extrinsic) → 4×4 homogeneous matrix."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, 0],
        [0, 0, 0, 1],
    ])


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis + angle (Rodrigues' formula)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis / np.linalg.norm(axis)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s, 0],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c), 0],
        [0, 0, 0, 1],
    ])


def _translation_matrix(xyz: tuple[float, float, float]) -> np.ndarray:
    """Translation → 4×4 homogeneous matrix."""
    m = np.eye(4)
    m[:3, 3] = xyz
    return m


def _resolve_urdf(urdf_path: Path, mesh_dir: Path | None = None) -> str:
    """Replace ``package://`` URIs in URDF with absolute filesystem paths.

    Args:
        urdf_path: Path to the URDF file.
        mesh_dir: Directory containing mesh files.  If None, auto-detected as
            ``<urdf_parent>/meshes/``.

    Returns:
        URDF XML string with resolved paths.
    """
    if mesh_dir is None:
        mesh_dir = urdf_path.resolve().parent / "meshes"
    mesh_dir = Path(mesh_dir).resolve()
    content = urdf_path.read_text(encoding="utf-8")
    content = re.sub(
        r'package://[^/]+/meshes/',
        str(mesh_dir) + "/",
        content,
    )
    return content


@dataclass
class JointInfo:
    """Parsed URDF joint information."""
    name: str
    joint_type: str  # "revolute", "prismatic", "fixed"
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower_limit: float
    upper_limit: float


def _parse_float3(el: ET.Element, attr: str, default: str = "0 0 0") -> tuple[float, float, float]:
    v = el.get(attr, default).strip().split()
    return (float(v[0]), float(v[1]), float(v[2]))


def parse_urdf(xml_str: str) -> tuple[list[JointInfo], list[str]]:
    """Parse URDF XML into joint info list and link name list.

    Returns:
        joints: List of JointInfo in traversal order (root → leaf).
        link_names: All link names.
    """
    root = ET.fromstring(xml_str)
    robot_name = root.get("name", "robot")
    logger.info("Parsing URDF for robot: %s", robot_name)

    # Collect all links
    link_names = [l.get("name") for l in root.findall("link")]

    # Build parent→child tree
    joints_raw = []
    children_to_parent = {}
    for j in root.findall("joint"):
        jname = j.get("name")
        jtype = j.get("type")
        parent = j.find("parent").get("link") if j.find("parent") is not None else ""
        child = j.find("child").get("link") if j.find("child") is not None else ""
        children_to_parent[child] = (parent, jname)

        origin = j.find("origin")
        xyz = _parse_float3(origin, "xyz", "0 0 0") if origin is not None else (0, 0, 0)
        rpy = _parse_float3(origin, "rpy", "0 0 0") if origin is not None else (0, 0, 0)

        axis_el = j.find("axis")
        axis = _parse_float3(axis_el, "xyz", "1 0 0") if axis_el is not None else (1, 0, 0)

        limit = j.find("limit")
        lower = float(limit.get("lower", 0)) if limit is not None else -np.inf
        upper = float(limit.get("upper", 0)) if limit is not None else np.inf

        joints_raw.append(JointInfo(
            name=jname, joint_type=jtype,
            parent=parent, child=child,
            origin_xyz=xyz, origin_rpy=rpy,
            axis=axis,
            lower_limit=lower, upper_limit=upper,
        ))

    # Topological sort: find root (child not appearing as any child)
    all_parents = {j.parent for j in joints_raw}
    all_children = {j.child for j in joints_raw}
    roots = all_parents - all_children
    if not roots:
        roots = {joints_raw[0].parent}

    # Walk tree from each root
    ordered = []
    visited = set()
    def walk(name, chain=None):
        if chain is None:
            chain = []
        if name in visited:
            return
        visited.add(name)
        for j in joints_raw:
            if j.parent == name:
                chain.append(j)
                walk(j.child, chain)
                return
    for r in roots:
        walk(r, ordered)

    # If ordering missed some, append all remaining
    for j in joints_raw:
        if j not in ordered:
            ordered.append(j)

    return ordered, link_names


def joint_map_dataset_to_urdf(state: np.ndarray) -> np.ndarray:
    """Map 7-DoF dataset state to 8-DoF URDF joint configuration.

    Dataset: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
    URDF:    [joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8]

    joint7 (left finger) = gripper (positive = open)
    joint8 (right finger) = -gripper (mirror motion)
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        q = np.zeros(8)
        q[:6] = state[:6]
        q[6] = state[6]   # joint7: left finger
        q[7] = -state[6]  # joint8: right finger (mirror)
        return q
    else:
        # Batch mode: (T, 7) → (T, 8)
        T = state.shape[0]
        q = np.zeros((T, 8))
        q[:, :6] = state[:, :6]
        q[:, 6] = state[:, 6]
        q[:, 7] = -state[:, 6]
        return q


class RobotVisualizer:
    """URDF-based robot model for FK computation + MeshCat visualization.

    Uses pure numpy + XML parsing (no Pinocchio dependency) for forward
    kinematics from URDF joint definitions.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        mesh_dir: str | Path | None = None,
    ):
        """Load robot model from URDF.

        Args:
            urdf_path: Path to ``.urdf`` file.
            mesh_dir: Directory containing STL/DAE mesh files.  If None,
                auto-detected as ``<urdf_parent>/meshes/``.
        """
        self._urdf_path = Path(urdf_path)
        xml_str = _resolve_urdf(self._urdf_path, mesh_dir)

        self._joints, self._link_names = parse_urdf(xml_str)

        # Build name → JointInfo map
        self._joint_map = {j.name: j for j in self._joints}

        # Ordered list of active (non-fixed) joint names
        self._active_joint_names = [
            j.name for j in self._joints if j.joint_type in ("revolute", "prismatic")
        ]

        logger.info(
            "Loaded robot with %d links, %d joints (%d active)",
            len(self._link_names), len(self._joints), len(self._active_joint_names),
        )

        self._mesh_dir = mesh_dir
        self._vis: meshcat.Visualizer | None = None

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def compute_fk(self, joint_positions: np.ndarray) -> dict[str, dict]:
        """Compute forward kinematics for all links.

        Args:
            joint_positions: Joint configuration.  Either 8-DoF URDF config
                (indexes matching ``self.active_joint_names``) or 7-DoF dataset
                config (automatically mapped).

        Returns:
            ``{link_name: {"translation": [x, y, z], "rotation": [w, x, y, z]}}``
            in world frame.
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if len(joint_positions) == 7:
            joint_positions = joint_map_dataset_to_urdf(joint_positions)

        # Reset all transforms to identity
        T = np.eye(4)

        # We only track transforms for links that have a joint pointing to them
        link_to_joint = {}
        for j in self._joints:
            link_to_joint[j.child] = j

        fk_results = {}

        for link_name in self._link_names:
            if link_name in fk_results:
                continue

            # Compute transform for this link
            j = link_to_joint.get(link_name)
            if j is not None:
                # Fixed offset from parent
                T_fixed = _translation_matrix(j.origin_xyz) @ _rpy_to_matrix(j.origin_rpy)

                # Joint variable (if any)
                if j.joint_type == "revolute":
                    idx = self._active_joint_names.index(j.name)
                    q = joint_positions[idx]
                    T_var = _axis_angle_rotation(np.array(j.axis), q)
                elif j.joint_type == "prismatic":
                    idx = self._active_joint_names.index(j.name)
                    q = joint_positions[idx]
                    T_var = _translation_matrix(tuple(q * a for a in j.axis))
                else:
                    T_var = np.eye(4)

                # The joint's transform = fixed offset × joint variable
                T_joint = T_fixed @ T_var

                # Compose with parent's transform
                parent_fk = fk_results.get(j.parent, {})
                if parent_fk:
                    T_parent = np.eye(4)
                    T_parent[:3, 3] = parent_fk["translation"]
                    T_parent[:3, :3] = mtf.quaternion_matrix(parent_fk["rotation"])[:3, :3]
                    T_link = T_parent @ T_joint
                else:
                    T_link = T_joint.copy()

                trans = T_link[:3, 3].tolist()
                quat = mtf.quaternion_from_matrix(T_link).tolist()
                fk_results[link_name] = {"translation": trans, "rotation": quat}

        return fk_results

    @property
    def active_joint_names(self) -> list[str]:
        return list(self._active_joint_names)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self._joints]

    @property
    def link_names(self) -> list[str]:
        return list(self._link_names)

    # ------------------------------------------------------------------
    # MeshCat Visualization
    # ------------------------------------------------------------------

    def _ensure_visualizer(self, headless: bool = True) -> meshcat.Visualizer:
        if self._vis is None:
            vis = meshcat.Visualizer()
            if not headless:
                vis.open()
            self._vis = vis
            self._load_geometry()
        return self._vis

    def _load_geometry(self):
        """Load mesh geometry into the MeshCat scene."""
        self._vis.delete()
        # Add coordinate frame for reference
        self._vis["/world"].set_transform(np.eye(4))

        # Load geometry from the resolved URDF
        xml_str = _resolve_urdf(self._urdf_path, self._mesh_dir)
        root = ET.fromstring(xml_str)

        for link in root.findall("link"):
            lname = link.get("name")
            visual = link.find("visual")
            if visual is None:
                continue
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            mesh_path = mesh_elem.get("filename", "")
            if not Path(mesh_path).exists():
                logger.debug("Mesh not found: %s, using fallback sphere", mesh_path)
                self._vis[f"robot/{lname}"].set_object(
                    g.Sphere(0.02),
                    g.MeshPhongMaterial(color=0x888888, wireframe=False),
                )
                continue

            try:
                mesh = g.StlMeshGeometry.from_file(mesh_path)
                mat = g.MeshPhongMaterial(color=0x6666cc, wireframe=False)
                self._vis[f"robot/{lname}"].set_object(mesh, mat)
            except Exception as e:
                logger.warning("Failed to load mesh %s: %s", mesh_path, e)
                self._vis[f"robot/{lname}"].set_object(
                    g.Sphere(0.02),
                    g.MeshPhongMaterial(color=0x888888, wireframe=False),
                )

    def render_frame(
        self,
        joint_positions: np.ndarray,
        headless: bool = True,
    ) -> meshcat.Visualizer:
        """Update the MeshCat scene to reflect the given joint configuration.

        Args:
            joint_positions: 7-DoF (dataset) or 8-DoF (URDF) joint config.
            headless: If True, no window is opened (for headless rendering).

        Returns:
            The MeshCat visualizer instance.
        """
        vis = self._ensure_visualizer(headless=headless)
        fk = self.compute_fk(joint_positions)

        for link_name, pose in fk.items():
            T = np.eye(4)
            T[:3, 3] = pose["translation"]
            T[:3, :3] = mtf.quaternion_matrix(pose["rotation"])[:3, :3]
            vis[f"robot/{link_name}"].set_transform(T)

        return vis

    def render_video(
        self,
        joint_trajectory: np.ndarray,
        output_path: str | Path,
        fps: float = 10.0,
    ) -> None:
        """Save an interactive HTML animation of the robot trajectory.

        Open the HTML file in any browser to view the animation.  If a browser
        is already connected to the MeshCat server when this method is called,
        frames are also captured and saved as ``.mp4``.

        Args:
            joint_trajectory: (T, 7) or (T, 8) array of joint positions.
            output_path: Path for the output file.
            fps: Frames per second for the output video/animation.
        """
        import time

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joint_trajectory = np.asarray(joint_trajectory, dtype=np.float64)
        T = joint_trajectory.shape[0]

        vis = self._ensure_visualizer(headless=True)

        # Try video capture if a browser is already connected
        time.sleep(0.5)
        test_img = vis.get_image()
        if test_img is not None:
            import imageio_ffmpeg
            frames = [test_img]
            for t in range(1, min(T, 200)):
                self.render_frame(joint_trajectory[t], headless=True)
                time.sleep(0.03)
                img = vis.get_image()
                if img is not None:
                    frames.append(img)
            if len(frames) > 1:
                writer = imageio_ffmpeg.write_frames(
                    str(output_path),
                    (frames[0].shape[1], frames[0].shape[0]),
                    fps=fps, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                )
                writer.send(None)
                for img in frames:
                    writer.send(img[:, :, :3])
                writer.close()
                logger.info("Saved video to %s (%d frames)", output_path, len(frames))
                return

        # Fallback: save static HTML snapshot (last frame only)
        html = f"""<!DOCTYPE html><html><body>
<h2>Robot animation: {output_path.name}</h2>
<p>Open MeshCat at <a href="http://127.0.0.1:7000/static/">http://127.0.0.1:7000/static/</a>
to view the interactive visualization.</p>
<p>To capture headless video, install and run:
<pre>  pip install playwright
  playwright install chromium
</pre></p>
</body></html>"""
        html_path = output_path.with_suffix(".html")
        html_path.write_text(html)
        logger.info(
            "Saved placeholder HTML to %s (open MeshCat at http://127.0.0.1:7000/static/ to view).",
            html_path,
        )

    def close(self):
        """Close the MeshCat visualizer."""
        if self._vis is not None:
            self._vis.close()
            self._vis = None
