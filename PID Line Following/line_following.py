#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class PIDControllerNode:
    """A ROS node that performs PID control based on camera data."""

    def __init__(self):
        # Desired setpoint (pixel)
        self.desired_position = rospy.get_param('~desired_position', 320)

        # PID gains (tunable via ROS params)
        self.kp = rospy.get_param('~kp', 0.006)
        self.ki = rospy.get_param('~ki', 0.00005)
        self.kd = rospy.get_param('~kd', 1.3)

        # Internal PID state
        self._integral = 0.0
        self._last_error = 0.0

        # Latest camera reading
        self.current_position = None

        # Publishers & subscribers
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('color_mono', String, self._camera_callback)

        # Control loop rate
        self.rate = rospy.Rate(rospy.get_param('~rate', 10))

    def _camera_callback(self, msg):
        """Callback to receive camera measurement."""
        try:
            self.current_position = int(msg.data)
        except ValueError:
            rospy.logwarn("Received non‐integer camera data: '%s'", msg.data)

    def run(self):
        """Main control loop."""
        rospy.loginfo("PIDControllerNode started (setpoint=%d)", self.desired_position)
        while not rospy.is_shutdown():
            if self.current_position is None:
                # Haven't received any camera data yet
                self.rate.sleep()
                continue

            error = self.desired_position - self.current_position

            # Integral with anti-windup
            self._integral += error
            max_integral = rospy.get_param('~max_integral', 850)
            if abs(self._integral) > max_integral:
                self._integral = max_integral * (1 if self._integral > 0 else -1)

            # Derivative
            derivative = error - self._last_error

            # PID correction
            correction = (self.kp * error +
                          self.ki * self._integral +
                          self.kd * derivative)

            # Build and publish Twist
            cmd = Twist()
            cmd.linear.x = rospy.get_param('~linear_speed', 0.05)
            cmd.angular.z = correction
            self.cmd_pub.publish(cmd)

            rospy.logdebug("err=%d int=%.2f der=%.2f corr=%.3f",
                           error, self._integral, derivative, correction)

            # Prepare for next iteration
            self._last_error = error
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('pid_controller')
    try:
        PIDControllerNode().run()
    except rospy.ROSInterruptException:
        pass
