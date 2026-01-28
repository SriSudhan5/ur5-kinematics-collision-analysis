import os
import csv
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

class TrajCheck(Node):
    def __init__(self):
        super().__init__("ur5_traj_check")

        self.declare_parameter("traj_file", "trajectory_eaik.npz")
        self.declare_parameter("csv_out", "collision_report.csv")
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("state_validity_service", "")

        traj_file = os.path.expanduser(str(self.get_parameter("traj_file").value))
        csv_out = os.path.expanduser(str(self.get_parameter("csv_out").value))

        data = np.load(traj_file)
        self.t = np.asarray(data["t"], dtype=float).reshape(-1)
        self.Q = np.asarray(data["Q"], dtype=float)

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.group_name = str(self.get_parameter("group_name").value)

        srv_name = str(self.get_parameter("state_validity_service").value).strip()
        if not srv_name:
            srv_name = discover_state_validity_service(self) or ""
        if not srv_name:
            raise RuntimeError("No GetStateValidity service found. Start MoveIt first.")
        self.get_logger().info(f"Using GetStateValidity service: {srv_name}")

        self.client = self.create_client(GetStateValidity, srv_name)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for GetStateValidity service...")

        self.csv_out = csv_out
        self.k = 0
        self.results = []

        self.timer = self.create_timer(0.001, self.step)  # run as fast as possible

    def step(self):
        if self.k >= self.Q.shape[0]:
            # write CSV and exit
            with open(self.csv_out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["k", "t", "valid"])
                w.writerows(self.results)
            self.get_logger().info(f"Wrote report: {self.csv_out}")
            rclpy.shutdown()
            return

        q = self.Q[self.k]
        msg = JointState()
        msg.name = self.joint_names
        msg.position = q.tolist()

        req = GetStateValidity.Request()
        rs = RobotState()
        rs.joint_state = msg
        req.robot_state = rs
        req.group_name = self.group_name

        future = self.client.call_async(req)

        k_now = self.k
        t_now = float(self.t[self.k])

        def _done(fut):
            try:
                resp = fut.result()
                self.results.append([k_now, t_now, int(resp.valid)])
                if not resp.valid:
                    self.get_logger().warn(f"COLLISION at k={k_now}, t={t_now:.3f}s")
            except Exception as e:
                self.get_logger().error(str(e))

        future.add_done_callback(_done)
        self.k += 1


def main():
    rclpy.init()
    node = TrajCheck()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

