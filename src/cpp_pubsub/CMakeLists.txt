cmake_minimum_required(VERSION 3.5)
project(cpp_pubsub)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# # Add talker
# add_executable(talker src/publisher_member_function.cpp)
# ament_target_dependencies(talker rclcpp std_msgs)
# 
# # Add listener
# add_executable(listener src/subscriber_member_function.cpp)
# ament_target_dependencies(listener rclcpp std_msgs)

# Add replier
add_executable(replier src/reply_member_function.cpp)
ament_target_dependencies(replier rclcpp std_msgs)

install(TARGETS
  # talker
  # listener
  replier
  DESTINATION lib/${PROJECT_NAME})

ament_package()
