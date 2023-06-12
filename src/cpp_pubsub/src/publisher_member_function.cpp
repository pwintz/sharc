#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
      publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/f_output", 10);
      timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

  private:
    void timer_callback()
    {
      // auto message = std_msgs::msg::Float64MultiArray();
      std::vector<double> vec1 = { 1.1, 2., 3.1};
      std_msgs::msg::Float64MultiArray msg;

      // set up dimensions
      msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
      msg.layout.dim[0].size = vec1.size();
      msg.layout.dim[0].stride = 1;
      msg.layout.dim[0].label = "x"; // or whatever name you typically use to index vec1

      // copy in the data
      msg.data.clear();
      msg.data.insert(msg.data.end(), vec1.begin(), vec1.end());
      // message.data = "Hello, world! " + std::to_string(count_++);
      // RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
      publisher_->publish(msg);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}