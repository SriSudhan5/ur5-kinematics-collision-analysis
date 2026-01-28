import os
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState


DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def discover_state_validity_service(node: Node) -> str | None:
    for name, types in node.get_service_names_and_types():
        if "moveit_msgs/srv/GetStateValidity" in types:
            return name
    return None


class TrajPlayback(Node):
    def __init__(self):
        super().__init__("ur5_traj_playback")

        self.declare_parameter("traj_file", "trajectory_eaik.npz")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("loop", True)

        self.declare_parameter("check_collision", True)
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("state_validity_service", "")

        traj_file = os.path.expanduser(str(self.get_parameter("traj_file").value))
        data = np.load(traj_file)
        self.t = np.asarray(data["t"], dtype=float).reshape(-1)
        self.Q = np.asarray(data["Q"], dtype=float)
        if self.Q.shape[0] != self.t.shape[0]:
            raise RuntimeError("t and Q length mismatch")

        self.joint_names = list(self.get_parameter("joint_names").value)
        if len(self.joint_names) != self.Q.shape[1]:
            raise RuntimeError(f"joint_names length {len(self.joint_names)} != Q dof {self.Q.shape[1]}")

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.check_collision = bool(self.get_parameter("check_collision").value)
        self.group_name = str(self.get_parameter("group_name").value)
        srv_name = str(self.get_parameter("state_validity_service").value).strip()

        self.sv_client = None
        if self.check_collision:
            if not srv_name:
                srv_name = discover_state_validity_service(self) or ""
            if not srv_name:
                self.get_logger().warn("No GetStateValidity service found. Collision checks disabled.")
                self.check_collision = False
            else:
                self.get_logger().info(f"Using GetStateValidity service: {srv_name}")
                self.sv_client = self.create_client(GetStateValidity, srv_name)

        self.k = 0
        rate = float(self.get_parameter("rate").value)
        self.dt = 1.0 / rate
        self.loop = bool(self.get_parameter("loop").value)
        self.timer = self.create_timer(self.dt, self.on_tick)

        self.get_logger().info(f"Loaded {self.Q.shape[0]} points from {traj_file}. Publishing at {rate} Hz.")

    def on_tick(self):
        if self.k >= self.Q.shape[0]:
            if self.loop:
                self.k = 0
            else:
                self.get_logger().info("Done.")
                rclpy.shutdown()
                return

        q = self.Q[self.k]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        self.pub.publish(msg)

        if self.check_collision and self.sv_client is not None and self.sv_client.service_is_ready():
            req = GetStateValidity.Request()
            rs = RobotState()
            rs.joint_state = msg
            req.robot_state = rs
            req.group_name = self.group_name

            future = self.sv_client.call_async(req)

            def _done_cb(fut):
                try:
                    resp = fut.result()
                    if not resp.valid:
                        self.get_logger().warn(f"COLLISION at k={self.k}, t={self.t[self.k]:.3f}s")
                except Exception as e:
                    self.get_logger().error(f"GetStateValidity failed: {e}")

            future.add_done_callback(_done_cb)

        self.k += 1


def main():
    rclpy.init()
    node = TrajPlayback()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

