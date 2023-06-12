# ROS+Scarab in Docker for Feedback Control Simulation
This project uses Docker to provide a consistent development environment for ROS2 and Scarab so that 

# Setup
## Install Tools
* Docker
* Visual Studio Code
  - Install recommended extensions


## Build and Run ROS2 Nodes
In this section, we follow the steps given [here](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html#build-a-package) and [here](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html#build-and-run).

1. Within the root of the ROS2 workspace, build the package using `colcon`:
```
colcon build
```
2. Automatically install any needed dependencies:
```
rosdep update
rosdep install -i --from-path src --rosdistro foxy -y
```
3. Setup colcon in the current bash environment(?)(Do we also need to run `source install/local_setup.bash`?)
```
. install/setup.bash
```
4. Start the ROS2 node using `ros2 run <package> <node>`:
```
ros2 run cpp_pubsub replier
```

# Development

## Adding new ROS nodes