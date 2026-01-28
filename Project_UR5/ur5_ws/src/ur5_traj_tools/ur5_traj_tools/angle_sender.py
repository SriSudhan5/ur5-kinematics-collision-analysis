# cat > ~/Project_UR5_1/Project_UR5_1/ur5_ws/src/ur5_traj_tools/ur5_traj_tools/angle_sender.py <<'PY'
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


def parse_angles(line: str) -> np.ndarray:
    """
    Accept exactly 6 joint values.

    Examples (radians):
      0 -1.57 1.57 0 0 0

    Examples (degrees):
      0deg -90deg 90deg 0deg 0deg 0deg
      0 -90deg 90deg 0 0 0   (mix allowed; if any token has 'deg', all treated as deg)
    """
    s = line.strip().replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 6:
        raise ValueError("Need exactly 6 values.")

    use_deg = any(p.lower().endswith("deg") for p in parts)
    vals = []
    for p in parts:
        p2 = p.lower().replace("deg", "")
        vals.append(float(p2))

    q = np.array(vals, dtype=float)
    if use_deg:
        q = np.deg2rad(q)
    return q


class AngleSender(Node):
    def __init__(self):
        super().__init__("ur5_angle_sender")

        self.declare_parameter("service", "/check_state_validity")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("publish_repeats", 5)

        self.srv_name = str(self.get_parameter("service").value)
        self.group = str(self.get_parameter("group_name").value)
        self.repeats = int(self.get_parameter("publish_repeats").value)

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.client = self.create_client(GetStateValidity, self.srv_name)
        # Don't hard-fail if not available yet; user might start MoveIt after.
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"Service {self.srv_name} not ready. Start MoveIt for collision checks."
            )

        self.get_logger().info(
            "Type 6 joint angles.\n"
            "  radians: 0 -1.57 1.57 0 0 0\n"
            "  degrees: 0deg -90deg 90deg 0deg 0deg 0deg\n"
            "Type 'q' to quit.\n"
        )

    def publish_joint_state(self, q: np.ndarray) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES_UR5
        msg.position = q.tolist()
        self.pub.publish(msg)
        return msg

    def check_collision(self, js: JointState):
        if not self.client.service_is_ready():
            print("⚠️  MoveIt validity service not ready (start MoveIt).")
            return

        req = GetStateValidity.Request()
        rs = RobotState()
        rs.joint_state = js
        req.robot_state = rs
        req.group_name = self.group

        fut = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
        if not fut.done() or fut.result() is None:
            print("⚠️  validity check timed out")
            return

        if fut.result().valid:
            print("✅ VALID (collision-free)")
        else:
            print("❌ COLLISION")

    def run_cli(self):
        while rclpy.ok():
            try:
                line = input("angles> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if line.lower() in ["q", "quit", "exit"]:
                break
            if not line:
                continue

            try:
                q = parse_angles(line)
                # publish a few times so RViz / planning scene definitely catches it
                last_js = None
                for _ in range(max(1, self.repeats)):
                    last_js = self.publish_joint_state(q)
                    rclpy.spin_once(self, timeout_sec=0.01)

                print("Published q =", q)
                self.check_collision(last_js)
            except Exception as e:
                print("Error:", e)


def main():
    rclpy.init()
    node = AngleSender()
    node.run_cli()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
