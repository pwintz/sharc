#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// What is this for?
using std::placeholders::_1; 

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MinimalReplier : public rclcpp::Node
{
  public:
    MinimalReplier()
    : Node("minimal_replier"), count_(0)
    {
      subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("/f_input", 10, std::bind(&MinimalReplier::subscriber_callback, this, _1));
      publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/f_output", 10);
      // timer_ = this->create_wall_timer(500ms, std::bind(&MinimalReplier::reply_callback, this));
    }

  private:
    void subscriber_callback(const std_msgs::msg::Float64MultiArray::SharedPtr in_msg) const
    {
      auto vec_in = in_msg->data;
      // string = std::string str(vec.begin(), vec.end())/*  */;
      std::string str(vec_in.begin(), vec_in.end());

      RCLCPP_INFO(this->get_logger(), "I heard: '[%f, %f, %f]'", vec_in[1], vec_in[2], vec_in[3]);

      // Compute feedback.
      // #SCARAB_START
      std::vector<double> vec1 = { vec_in[1], 
                                  -vec_in[2], 
                                  -vec_in[3] };
      // #SCARAB_END

      // Publish feedback
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
      RCLCPP_INFO(this->get_logger(), 
      "Publishing: '[%f, %f, %f]'", vec1[1], vec1[2], vec1[3]);

      publisher_->publish(msg);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalReplier>());
  rclcpp::shutdown();
  return 0;
}