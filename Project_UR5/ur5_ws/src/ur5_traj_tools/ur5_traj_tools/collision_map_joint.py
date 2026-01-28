import os
import csv
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState

JOINT_NAMES_UR5 = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class JointSpaceCollisionMap(Node):
    def __init__(self):
        super().__init__("ur5_collision_map_joint")

        self.declare_parameter("service", "/check_state_validity")
        self.declare_parameter("group_name", "ur_manipulator")

        self.declare_parameter("q1_min", -np.pi)
        self.declare_parameter("q1_max",  np.pi)
        self.declare_parameter("q1_N", 81)

        self.declare_parameter("q2_min", -np.pi)
        self.declare_parameter("q2_max",  np.pi)
        self.declare_parameter("q2_N", 81)

        # fixed q3..q6
        self.declare_parameter("q_fixed", [-1.57, 1.57, 0.0, 0.0])

        self.declare_parameter("out_csv", "collision_map_q1_q2.csv")
        self.declare_parameter("out_npz", "collision_map_q1_q2.npz")
        self.declare_parameter("out_png", "collision_map_q1_q2.png")

        self.srv_name = str(self.get_parameter("service").value)
        self.group = str(self.get_parameter("group_name").value)

        q_fixed = np.asarray(self.get_parameter("q_fixed").value, dtype=float).reshape(-1)
        if q_fixed.size != 4:
            raise RuntimeError("q_fixed must be 4 values: [q3,q4,q5,q6]")
        self.q_fixed = q_fixed

        q1_min = float(self.get_parameter("q1_min").value)
        q1_max = float(self.get_parameter("q1_max").value)
        q1_N = int(self.get_parameter("q1_N").value)

        q2_min = float(self.get_parameter("q2_min").value)
        q2_max = float(self.get_parameter("q2_max").value)
        q2_N = int(self.get_parameter("q2_N").value)

        self.q1_vals = np.linspace(q1_min, q1_max, q1_N)
        self.q2_vals = np.linspace(q2_min, q2_max, q2_N)
        self.valid = np.zeros((q2_N, q1_N), dtype=np.uint8)  # rows=q2, cols=q1

        self.out_csv = os.path.expanduser(str(self.get_parameter("out_csv").value))
        self.out_npz = os.path.expanduser(str(self.get_parameter("out_npz").value))
        self.out_png = os.path.expanduser(str(self.get_parameter("out_png").value))

        self.client = self.create_client(GetStateValidity, self.srv_name)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for {self.srv_name} ...")

        total = self.q1_vals.size * self.q2_vals.size
        self.get_logger().info(
            f"Sampling q1 x q2 = {self.q1_vals.size} x {self.q2_vals.size} = {total} states, fixed q3..q6={self.q_fixed.tolist()}"
        )

        # run sequentially in constructor (simple + reliable)
        self.run_all_checks()
        self.write_outputs()
        self.get_logger().info("Done.")
        rclpy.shutdown()

    def check_one(self, q: np.ndarray) -> bool:
        js = JointState()
        js.name = JOINT_NAMES_UR5
        js.position = q.tolist()

        req = GetStateValidity.Request()
        rs = RobotState()
        rs.joint_state = js
        req.robot_state = rs
        req.group_name = self.group

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        resp = future.result()
        if resp is None:
            return False
        return bool(resp.valid)

    def run_all_checks(self):
        total = self.q1_vals.size * self.q2_vals.size
        k = 0
        for jj, q2 in enumerate(self.q2_vals):
            for ii, q1 in enumerate(self.q1_vals):
                q = np.zeros(6, dtype=float)
                q[0] = q1
                q[1] = q2
                q[2:] = self.q_fixed

                ok = self.check_one(q)
                self.valid[jj, ii] = 1 if ok else 0

                k += 1
                if k % 200 == 0 or k == total:
                    self.get_logger().info(f"Progress: {k}/{total}")

    def write_outputs(self):
        np.savez(self.out_npz, q1=self.q1_vals, q2=self.q2_vals, valid=self.valid)
        self.get_logger().info(f"Wrote {self.out_npz}")

        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["q1", "q2", "valid"])
            for jj, q2 in enumerate(self.q2_vals):
                for ii, q1 in enumerate(self.q1_vals):
                    w.writerow([float(q1), float(q2), int(self.valid[jj, ii])])
        self.get_logger().info(f"Wrote {self.out_csv}")

        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(
                self.valid,
                origin="lower",
                aspect="auto",
                extent=[self.q1_vals[0], self.q1_vals[-1], self.q2_vals[0], self.q2_vals[-1]],
            )
            plt.xlabel("q1 [rad]")
            plt.ylabel("q2 [rad]")
            plt.title("Collision-free map (1=valid, 0=collision)")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.out_png, dpi=200)
            self.get_logger().info(f"Wrote {self.out_png}")
        except Exception as e:
            self.get_logger().warn(f"Plot skipped (matplotlib missing in ROS python). Reason: {e}")

def main():
    rclpy.init()
    JointSpaceCollisionMap()

if __name__ == "__main__":
    main()
