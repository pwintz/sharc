#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64 


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        queue_size = 10
        topic_name = 'topic'
        self.publisher_ = self.create_publisher(Int64, topic_name, queue_size)
        self.i = 0
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Int64()
        msg.data = self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
