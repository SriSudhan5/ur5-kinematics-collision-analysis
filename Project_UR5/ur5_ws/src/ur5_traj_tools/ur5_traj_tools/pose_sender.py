import os
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState

from eaik.IK_URDF import UrdfRobot

UR5_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ---------- small helpers (no ur5_planner dependency) ----------

def wrap_to_pi(x):
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def edge_cost(q_from, q_to):
    dq = wrap_to_pi(np.asarray(q_to) - np.asarray(q_from))
    return float(dq @ dq)

def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n <= 0:
        raise ValueError("Bad quaternion norm")
    x, y, z, w = x/n, y/n, z/n, w/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R

def parse_pose_line(line: str):
    parts = line.strip().split()
    if len(parts) == 3:
        x, y, z = map(float, parts)
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    elif len(parts) == 7:
        x, y, z, qx, qy, qz, qw = map(float, parts)
    else:
        raise ValueError("Need 3 values (x y z) or 7 values (x y z qx qy qz qw)")
    return np.array([x, y, z], dtype=float), np.array([qx, qy, qz, qw], dtype=float)

# ------------------------- node -------------------------

class PoseSender(Node):
    def __init__(self):
        super().__init__("ur5_pose_sender")

        self.declare_parameter("service", "/check_state_validity")
        self.srv_name = self.get_parameter("service").get_parameter_value().string_value

        urdf_path = os.environ.get("UR5_URDF_PATH", "")
        if not urdf_path or not os.path.isfile(urdf_path):
            raise RuntimeError(f"Set UR5_URDF_PATH to a URDF file. Got: {urdf_path}")
        self.bot = UrdfRobot(urdf_path)

        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.cli = self.create_client(GetStateValidity, self.srv_name)

        self.last_q = None

        self.get_logger().info(
            "Type pose:\n"
            "  x y z\n"
            "or\n"
            "  x y z qx qy qz qw\n"
            "Type 'q' to quit."
        )

    def pick_solution(self, Qs):
        if len(Qs) == 0:
            return None
        if self.last_q is None:
            return Qs[0]
        costs = [edge_cost(self.last_q, q) for q in Qs]
        return Qs[int(np.argmin(costs))]

    def publish_q(self, q: np.ndarray):
        msg = JointState()
        msg.name = UR5_JOINT_NAMES
        msg.position = [float(x) for x in q]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.last_q = np.asarray(q, dtype=float)

    def check_validity(self, q: np.ndarray):
        if not self.cli.service_is_ready():
            return None

        req = GetStateValidity.Request()
        rs = RobotState()
        rs.joint_state.name = UR5_JOINT_NAMES
        rs.joint_state.position = [float(x) for x in q]
        req.robot_state = rs
        # If needed for your MoveIt config:
        # req.group_name = "ur_manipulator"

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done():
            return None
        resp = future.result()
        return bool(resp.valid)

    def run_console(self):
        while rclpy.ok():
            try:
                line = input("\npose> ").strip()
            except EOFError:
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if not line:
                continue

            try:
                xyz, quat = parse_pose_line(line)
                qx, qy, qz, qw = quat.tolist()

                T = np.eye(4, dtype=float)
                T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
                T[:3, 3] = xyz

                sol = self.bot.IK(T)
                Qs = [np.asarray(x, dtype=float).reshape(6,) for x in sol.Q]
                q = self.pick_solution(Qs)
                if q is None:
                    print("❌ No IK solution")
                    continue

                self.publish_q(q)
                print("Published q =", np.round(q, 4))

                ok = self.check_validity(q)
                if ok is None:
                    print("⚠️  validity service not ready (start MoveIt for collision checks)")
                elif ok:
                    print("✅ VALID (collision-free)")
                else:
                    print("❌ COLLISION")

                rclpy.spin_once(self, timeout_sec=0.0)

            except Exception as e:
                print("Error:", e)

def main():
    rclpy.init()
    node = PoseSender()
    try:
        node.run_console()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
