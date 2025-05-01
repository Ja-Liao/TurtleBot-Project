#!/usr/bin/env python3
import os
import sys
import select
import math

import rospy
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String

if os.name == 'nt':
    import msvcrt
else:
    import tty
    import termios


def get_key(timeout=0.1):
    """
    Read a single keypress from stdin (non‐blocking).
    Supports Windows (msvcrt) and Unix (tty/select/termios).
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
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ''


class KalmanFilterNode:
    """
    A ROS node that implements a simple Kalman filter to estimate
    the x‐position of a robot relative to a tower, using noisy
    cmd_vel inputs and scan_angle measurements.
    """

    def __init__(self):
        # Parameters (can be overridden via ROS params)
        self.h = rospy.get_param('~tower_height', 0.6)      # meters
        self.d = rospy.get_param('~tower_distance', 1.5)    # meters
        self.Q = rospy.get_param('~process_noise', 1.0)     # process noise covariance
        self.R = rospy.get_param('~meas_noise', 1.0)        # measurement noise covariance
        self.P = rospy.get_param('~init_covariance', 1.0)   # initial estimate covariance
        self.x = rospy.get_param('~init_state', 0.0)        # initial x‐position estimate

        # State for inputs
        self.u = 0.0      # last cmd_vel.linear.x
        self.phi = float('nan')  # last measured scan angle (radians)

        # Data logging
        self.positions = []
        self.covariances = []
        self.times = []

        # Publishers & subscribers
        self.state_pub = rospy.Publisher('state', String, queue_size=1)
        rospy.Subscriber('cmd_vel_noisy', Twist, self._cmd_callback)
        rospy.Subscriber('scan_angle', String, self._scan_callback)

        # Frequency of Kalman update
        self.rate_hz = rospy.get_param('~rate', 30)
        self.rate = rospy.Rate(self.rate_hz)

        # For plotting trigger
        self.plot_threshold = rospy.get_param('~plot_threshold', 310.0)

        rospy.loginfo("KalmanFilterNode initialized")

    def _cmd_callback(self, msg: Twist):
        """Receive control input (noisy linear x‐velocity)."""
        self.u = msg.linear.x

    def _scan_callback(self, msg: String):
        """Receive scan angle (degrees) and convert to radians."""
        try:
            self.phi = float(msg.data) * math.pi / 180.0
        except ValueError:
            rospy.logwarn("Non‐float scan_angle: '%s'", msg.data)

    def _predict(self, dt):
        """Prediction step: xₖ = xₖ₋₁ + u·dt."""
        self.x += self.u * dt
        self.P += self.Q

    def _compute_expected_measurement(self):
        """
        Given current state x, compute expected bearing φₖ =
        arctan2(h, d − x).
        """
        return math.atan2(self.h, self.d - self.x)

    def _update(self):
        """Correction step, if a valid measurement is available."""
        if math.isnan(self.phi):
            return

        # linearize measurement function: ∂φ/∂x
        denom = (self.d - self.x)**2 + self.h**2
        H = self.h / denom

        # innovation covariance S = H·P·Hᵀ + R
        S = H * self.P * H + self.R

        # Kalman gain
        K = (self.P * H) / S

        # measurement residual
        z_hat = self._compute_expected_measurement()
        residual = self.phi - z_hat

        # update state & covariance
        self.x += K * residual
        self.P *= (1 - K * H)

    def run(self):
        """Main loop: perform predict+update at fixed rate."""
        rospy.loginfo("Starting Kalman filter loop at %d Hz", self.rate_hz)
        t = 0.0
        dt = 1.0 / self.rate_hz

        while not rospy.is_shutdown():
            # 1) Prediction step
            self._predict(dt)

            # 2) Update step
            self._update()

            # 3) Record & publish
            t += dt
            self.times.append(t)
            self.positions.append(self.x)
            self.covariances.append(self.P)
            self.state_pub.publish(f"{self.x:.4f}")

            rospy.logdebug("t=%.2f  x=%.4f  P=%.4f", t, self.x, self.P)

            # 4) Plot once threshold is crossed
            if self.x > self.plot_threshold:
                plt.figure()
                plt.plot(self.times, self.positions, label='x estimate')
                plt.plot(self.times, self.covariances, label='covariance')
                plt.xlabel('Time [s]')
                plt.legend()
                plt.savefig("kalman_plot.png")
                rospy.loginfo("Plot saved to kalman_plot.png")
                # avoid re‐plotting
                self.plot_threshold = float('inf')

            # 5) Keyboard interrupt
            if get_key() == '\x03':  # Ctrl‐C
                rospy.loginfo("User interruption, exiting.")
                break

            self.rate.sleep()

        rospy.loginfo("KalmanFilterNode shutting down.")


if __name__ == '__main__':
    rospy.init_node('kalman_filter_node')
    try:
        node = KalmanFilterNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
