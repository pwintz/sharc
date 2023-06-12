#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/f_input", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) const
    {
      auto vec = msg->data;
      // string = std::string str(vec.begin(), vec.end())/*  */;
      std::string str(vec.begin(), vec.end());

      RCLCPP_INFO(this->get_logger(), "I heard: '[%f, %f, %f]'", vec[1], vec[2], vec[3]);
    }
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
