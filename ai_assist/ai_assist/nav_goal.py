#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import rclpy, math, time,os,yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

class eBotNav(Node):
    def __init__(self):
        super().__init__('nav_cmd')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose') 

        file_path = os.path.join(
            get_package_share_directory('ai_assist'), 'config', 'config.yaml'
        )

        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        self.pre_dock_position = {}
        for item in config_data['pre_dock_position']:
            for key, value in item.items():
                self.pre_dock_position[key] = {'x': value[0], 'y': value[1], 'yaw': value[2]}
        

        self.pose_subscription = self.create_subscription(
            Odometry,  
            'odom',  
            self.pose_callback,
            10)

        self.current_pose = None  
        self.get_logger().info('Waiting for action server...')
        self.client.wait_for_server()


    def pose_callback(self, msg):
        """Callback to handle the robot's current pose."""
        self.current_pose = msg.pose.pose  

    def get_current_pose(self):
        """Returns the robot's current pose as (x, y, yaw in radians)"""
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            orientation_q = self.current_pose.orientation
            _, _, yaw = euler_from_quaternion(
                [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
            return x, y, yaw
        return None

    def send_goal(self, x, y, yaw):
        """Sends a navigation goal to the action server."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        q = self.quaternion_from_yaw(yaw)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        yaw_degrees = math.degrees(yaw)
        self.get_logger().info(f'Sending goal to x: {x}, y: {y}, yaw: {yaw_degrees} degrees')
        return self.client.send_goal_async(goal_msg)

    def quaternion_from_yaw(self, yaw):
        """Helper function to create a quaternion from yaw."""
        return quaternion_from_euler(0, 0, yaw)

    def check_goal_status(self, goal_handle_future):
        """Wait for the result of the goal and print the actual robot pose when reached."""
        goal_handle = goal_handle_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected :(')
            return False

        self.get_logger().info('Goal accepted, waiting for result!!!')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.status == 4:  
            self.get_logger().info('Goal succeeded!')
            current_pose = self.get_current_pose()
            if current_pose:
                x, y, yaw = current_pose
                yaw_degrees = math.degrees(yaw)
                self.get_logger().info(f'Robot reached position: x: {x}, y: {y}, yaw: {yaw_degrees} degrees')
            else:
                self.get_logger().info('Unable to retrieve current pose')
            return True
        else:
            self.get_logger().info(f'Goal failed with status code: {result.status}')
            return False


    def move_to_goal(self, name):
        """Move to a specific waypoint and perform actions at the waypoint."""
        coordinates = self.pre_dock_position[name]
        x, y, yaw = coordinates['x'], coordinates['y'], coordinates['yaw']
        self.get_logger().info(f"Navigating to {name} (x: {x}, y: {y}, yaw: {yaw})")

        future = self.send_goal(x, y, yaw)
        rclpy.spin_until_future_complete(self, future)

        if not self.check_goal_status(future):
            self.get_logger().info(f'Navigation failed at {name}, stopping.')
            return 
        else:
            self.get_logger().info(f'Reached {name} successfully.')

def main(args=None):
    rclpy.init(args=args)
    ebot_nav = eBotNav()

    ebot_nav.move_to_goal('burger_king')
    ebot_nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
  


