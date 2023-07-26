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

# Using Scarab

After building (as described above), while in `~`, run 
```
./trace_cmd ./install/cpp_pubsub/lib/cpp_pubsub/replier
```
and 
```
./run_scarab
```
Sometimes you need to run 
```
./run_scarab
```
twice for it to work. 

The resulting statistics are placed into `scarab_out


# Development in Dev Container

The context in the dev container (i.e., the folder presented in `/workspaces/ros-docker` in the container) is the root directory of the `ros-docker` project. Changes to the files in /workspaces/ros-docker are synced to the local host (outside of the Docker container) where as 

The contents of the user directory `~` are generated from `ros-docker/docker_user_home` (due to a copying it in our `Dockerfile`), but changes are not synced to the copy on the local machine. Changes to files in `~` are, however, preserved if Visual Studio Code is closed and restarted because the same Docker image is used each time. If the dev container is rebuilt, however, (via CTRL+SHIFT+P followed by "Dev containers: Rebuild Container"), then changes to `~` are lost. 

# Tools used by this project
* Docker -- Creates easily reproducible enviornments so that we can immediately spin-up new virtual machines that run Scarab and ROS.
* ROS -- Used to comunicate between controller running in a Docker container and a plant simulator running elsewhere (in particular, we simulate the dynamics in Windows via MATLAB)
* Dev Container -- 
* MATLAB with ROS plugin
* CMake?
* ROS compile tool (what is it called??)
* Scarab
* DynamoRIO


## Adding new ROS nodes