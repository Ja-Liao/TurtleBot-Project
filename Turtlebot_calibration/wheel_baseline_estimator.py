#!/usr/bin/env python3

import rospy
import numpy as np
import threading

from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

INT32_MAX = 2**31
NUM_ROTATIONS = 3
TICKS_PER_ROTATION = 4096
WHEEL_RADIUS = 0.066 / 2  # meters


class WheelBaselineEstimator:
    def __init__(self):
        rospy.init_node('encoder_data', anonymous=True)

        # Subscribers
        rospy.Subscriber("cmd_vel", Twist, self.start_stop_callback)
        rospy.Subscriber("sensor_state", SensorState, self.sensor_callback)

        # Publisher
        self.reset_pub = rospy.Publisher('reset', Empty, queue_size=1)

        # Encoder state
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.del_left_encoder = 0
        self.del_right_encoder = 0

        # Motion flag & threading lock
        self.is_moving = False
        self.lock = threading.Lock()

        # Do an initial reset
        self.reset_pub.publish(Empty())
        rospy.loginfo("Ready to start wheel baseline calibration!")

    def safe_del_phi(self, a, b):
        """Compute delta between two 32-bit encoder readings, handling overflow."""
        diff = np.int64(b) - np.int64(a)
        if diff < -np.int64(INT32_MAX):
            # overflow
            return (INT32_MAX - a) + (b + INT32_MAX) + 1
        if diff > np.int64(INT32_MAX) - 1:
            # underflow
            return (a + INT32_MAX) + (INT32_MAX - b) + 1
        return int(diff)

    def sensor_callback(self, msg: SensorState):
        """Accumulate encoder deltas on each sensor update."""
        with self.lock:
            if self.left_encoder_prev is None:
                # first reading
                self.left_encoder_prev = msg.left_encoder
                self.right_encoder_prev = msg.right_encoder
            else:
                # compute increments
                dl = self.safe_del_phi(self.left_encoder_prev, msg.left_encoder)
                dr = self.safe_del_phi(self.right_encoder_prev, msg.right_encoder)
                self.del_left_encoder += dl
                self.del_right_encoder += dr

                # store for next time
                self.left_encoder_prev = msg.left_encoder
                self.right_encoder_prev = msg.right_encoder

    def start_stop_callback(self, msg: Twist):
        """
        Detect when an in-place rotation starts and stops.
        On stop, compute and print the wheel baseline.
        """
        # start rotation?
        if not self.is_moving and abs(msg.angular.z) > 0:
            self.is_moving = True
            rospy.loginfo("Starting calibration rotation...")

        # stop rotation?
        elif self.is_moving and abs(msg.angular.z) < 1e-6:
            self.is_moving = False

            # === Calibration computation ===
            # total yaw turned (radians)
            total_theta = NUM_ROTATIONS * 2.0 * np.pi

            # snapshot and reset deltas
            with self.lock:
                dl = self.del_left_encoder
                dr = self.del_right_encoder

            # convert ticks to distance
            left_dist = (dl / float(TICKS_PER_ROTATION)) * (2.0 * np.pi * WHEEL_RADIUS)
            right_dist = (dr / float(TICKS_PER_ROTATION)) * (2.0 * np.pi * WHEEL_RADIUS)

            # each wheel travels s = (baseline/2) * theta
            s_mean = 0.5 * (abs(left_dist) + abs(right_dist))
            baseline = 2.0 * s_mean / total_theta

            rospy.loginfo("Calibrated wheel separation (baseline): %.4f m", baseline)
            # ================================

            # reset for next iteration
            with self.lock:
                self.left_encoder_prev = None
                self.right_encoder_prev = None
                self.del_left_encoder = 0
                self.del_right_encoder = 0

            self.reset_pub.publish(Empty())
            rospy.loginfo("Robot reset; ready for another calibration.")

    def run(self):
        rospy.loginfo("WheelBaselineEstimator node up and running.")
        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = WheelBaselineEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
