#!/usr/bin/env python
import os
import sys
import select

import rospy
import numpy as np
import pandas as pd

from geometry_msgs.msg import Twist
from std_msgs.msg import String

if os.name == 'nt':
    import msvcrt
else:
    import tty
    import termios


def get_key(timeout: float = 0.1) -> str:
    """
    Read a single character from stdin, nonblocking.
    Supports Windows (msvcrt) or Unix (tty/select/termios).
    """
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8', errors='ignore')

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class BayesPIDNode:
    """
    ROS node that fuses line‐following PID control with a
    Bayesian localization update based on observed colors.
    """

    def __init__(self):
        # Parameters
        self.des_pos = rospy.get_param('~desired_position', 320)
        self.kp = rospy.get_param('~kp', 0.002)
        self.ki = rospy.get_param('~ki', 0.0001)
        self.kd = rospy.get_param('~kd', 2.4)
        self.max_integral = rospy.get_param('~max_integral', 850)
        self.linear_speed = rospy.get_param('~linear_speed', 0.05)
        self.loop_rate_hz = rospy.get_param('~loop_rate', 10)
        colour_map = rospy.get_param(
            '~colour_map',
            ['green', 'orange', 'green', 'yellow', 'blue',
             'green', 'orange', 'yellow', 'blue', 'blue',
             'orange', 'yellow']
        )

        # State storage
        self.current_rgb = np.zeros(3)
        self.line_idx = None
        self.location = 'Nothing'
        self.integral = 0.0
        self.last_error = 0.0

        # Build Bayesian models
        self._init_colour_map(colour_map)
        self._build_state_model()
        self._build_measurement_model()

        # ROS interfaces
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('mean_img_rgb', String, self._camera_cb)
        rospy.Subscriber('line_idx', String, self._line_cb)

        self.rate = rospy.Rate(self.loop_rate_hz)
        rospy.loginfo("BayesPIDNode initialized")

    def _init_colour_map(self, colour_list):
        """Initialize the prior distribution over map positions."""
        n = len(colour_list)
        uniform_prob = 1.0 / n
        self.prior = pd.DataFrame({
            'position': range(n),
            'colour': colour_list,
            'belief': [uniform_prob] * n
        })

    def _build_state_model(self):
        """Define P(x_k | x_{k−1}, action) transition matrix."""
        data = [
            [0.85, 0.05, 0.05],
            [0.10, 0.90, 0.10],
            [0.05, 0.10, 0.85]
        ]
        idx = ['X-chi', 'X', 'X+chi']
        cols = [-1, 0, 1]
        self.state_model = pd.DataFrame(data, index=idx, columns=cols)

    def _build_measurement_model(self):
        """Define P(z | x) measurement matrix for each colour."""
        data = [
            [0.60, 0.20, 0.05, 0.05],
            [0.20, 0.60, 0.05, 0.05],
            [0.05, 0.05, 0.65, 0.20],
            [0.05, 0.05, 0.15, 0.60],
            [0.10, 0.10, 0.10, 0.10],
        ]
        idx = ['Blue', 'Green', 'Yellow', 'Orange', 'Nothing']
        cols = ['Blue', 'Green', 'Yellow', 'Orange']
        self.meas_model = pd.DataFrame(data, index=idx, columns=cols)

    def _camera_cb(self, msg: String):
        """Parse 'r:.. g:.. b:..' into self.current_rgb."""
        try:
            parts = re.split('[rgb: ,]+', msg.data.strip())
            r, g, b = map(float, parts[1:4])
            self.current_rgb = np.array([r, g, b])
        except Exception:
            rospy.logwarn("Malformed RGB msg: '%s'", msg.data)

    def _line_cb(self, msg: String):
        """Update the detected black‐line pixel index."""
        try:
            self.line_idx = int(msg.data)
        except ValueError:
            rospy.logwarn("Non-int line index: '%s'", msg.data)

    def _classify_colour(self) -> str:
        """Simple threshold‐based color to zone classifier."""
        r, g, b = self.current_rgb
        if b > 120 and (r + g) < 150:
            return 'Blue'
        if b < 5 and 50 < g < 100:
            return 'Orange'
        if b < 5 and g > r:
            return 'Green'
        if b < 5 and r > g:
            return 'Yellow'
        return 'Nothing'

    def _bayes_predict(self, action: int):
        """Apply the state transition model to get a new prior."""
        n = len(self.prior)
        new_beliefs = []
        for i in range(n):
            p = sum(
                self.state_model.loc[state, action] *
                self.prior.belief.iloc[(i - delta) % n]
                for state, delta in [('X-chi', +1), ('X', 0), ('X+chi', -1)]
            )
            new_beliefs.append(p)
        self.prior['belief'] = new_beliefs

    def _bayes_update(self, colour: str):
        """Apply the measurement model to correct the prior."""
        likelihoods = self.prior['colour'].map(
            lambda c: self.meas_model.at[colour, c]
        )
        unnorm = self.prior.belief * likelihoods
        norm = unnorm.sum()
        if norm > 0:
            self.prior['belief'] = unnorm / norm
        else:
            rospy.logwarn("Zero total probability on update")

    def _pid_control(self):
        """Compute and publish the Twist for line‐following when on the line."""
        twist = Twist()
        twist.linear.x = self.linear_speed

        colour = self._classify_colour()
        rospy.logdebug("Observed colour: %s", colour)

        if colour != 'Nothing' and self.line_idx is not None:
            # Predict & correct Bayesian belief
            self._bayes_predict(action=1)
            self._bayes_update(colour)

            # PID steering
            error = self.des_pos - self.line_idx
            self.integral = np.clip(self.integral + error,
                                    -self.max_integral, self.max_integral)
            derivative = error - self.last_error

            correction = (self.kp * error +
                          self.ki * self.integral +
                          self.kd * derivative)
            twist.angular.z = correction
            self.last_error = error

            rospy.logdebug("PID → err: %d int: %.1f der: %.1f corr: %.3f",
                           error, self.integral, derivative, correction)
        else:
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    def run(self):
        """Main loop: PID+Bayes control until Ctrl‐C."""
        rospy.loginfo("Starting control loop. Press Ctrl‐C to exit.")
        while not rospy.is_shutdown():
            self._pid_control()
            # Exit if user presses Ctrl‐C on keyboard
            if get_key() == '\x03':
                rospy.loginfo("User requested shutdown.")
                break
            self.rate.sleep()

        # Stop the robot
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Node terminated cleanly.")


if __name__ == '__main__':
    rospy.init_node('bayes_pid_node')
    node = BayesPIDNode
