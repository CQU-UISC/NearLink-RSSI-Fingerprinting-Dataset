import rospy
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class ScoutMiniRotate:
    def __init__(self):
        self.target_yaw = 1.5708
        self.angular_speed = 0.2
        self.yaw_tolerance = 0.05
        self.initial_yaw = None
        self.current_yaw = 0.0
        self.current_linear_x = 0.0
        self.current_linear_y = 0.0
        self.is_rotating = False
        rospy.init_node('scout_mini_rotate_left', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/imu/data_raw', Imu, self.imu_receiver)
        self.cmd_vel = Twist()

    def odom_callback(self, msg):
        self.current_linear_x = msg.twist.twist.linear.x
        self.current_linear_y = msg.twist.twist.linear.y

    def imu_receiver(self, msg):
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        if self.initial_yaw is None:
            self.initial_yaw = yaw
            rospy.loginfo("Initial yaw set: %f radians" % self.initial_yaw)
        self.current_yaw = yaw - self.initial_yaw
        rospy.loginfo("Relative yaw: %f radians" % self.current_yaw)

    def rotate_left(self):
        rate = rospy.Rate(10)
        self.is_rotating = True
        while not rospy.is_shutdown():
            if abs(self.current_yaw - self.target_yaw) <= self.yaw_tolerance:
                self.cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel)
                rospy.loginfo("Target rotation reached")
                break
            if abs(self.current_linear_x) < 0.01 and abs(self.current_linear_y) < 0.01:
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = self.angular_speed
            else:
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.0
                rospy.logwarn("Linear motion detected, stopping")
            self.cmd_vel_pub.publish(self.cmd_vel)
            rate.sleep()

    def shutdown(self):
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel)
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    try:
        scout = ScoutMiniRotate()
        scout.rotate_left()
    except rospy.ROSInterruptException:
        scout.shutdown()
