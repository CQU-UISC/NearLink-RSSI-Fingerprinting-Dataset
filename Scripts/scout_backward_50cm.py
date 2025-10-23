import rospy
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class ScoutMiniControl:
    def __init__(self):
        self.target_distance = 0.47
        self.linear_speed = -0.05
        self.start_position = None
        self.current_position = None
        self.distance_traveled = 0.0
        self.current_yaw = 0.0
        self.initial_yaw = None
        self.is_moving = False
        rospy.init_node('scout_mini_move', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/imu/data_raw', Imu, self.imu_receiver)
        self.cmd_vel = Twist()

    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        if self.start_position is None and self.is_moving:
            self.start_position = self.current_position
        if self.start_position is not None:
            dx = self.current_position.x - self.start_position.x
            dy = self.current_position.y - self.start_position.y
            self.distance_traveled = math.sqrt(dx * dx + dy * dy)

    def imu_receiver(self, msg):
        orientation = msg.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        if self.initial_yaw is None:
            self.initial_yaw = yaw
            rospy.loginfo("Initial yaw set: %f radians" % self.initial_yaw)
        self.current_yaw = yaw - self.initial_yaw
        rospy.loginfo("Relative yaw: %f radians" % self.current_yaw)

    def move_forward(self):
        rate = rospy.Rate(10)
        self.is_moving = True
        while not rospy.is_shutdown():
            if self.distance_traveled >= self.target_distance:
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel)
                rospy.loginfo("Target distance reached")
                break
            if abs(self.current_yaw) < 0.3:
                self.cmd_vel.linear.x = self.linear_speed
                self.cmd_vel.angular.z = 0.0
            else:
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.0
                rospy.logwarn("Yaw deviation detected, stopping")
            self.cmd_vel_pub.publish(self.cmd_vel)
            rate.sleep()

    def shutdown(self):
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel)
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    try:
        scout = ScoutMiniControl()
        scout.move_forward()
    except rospy.ROSInterruptException:
        scout.shutdown()
