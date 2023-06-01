FROM ros:foxy

# install ros package
RUN apt-get update && apt-get install -y \
      ros-${ROS_DISTRO}-demo-nodes-cpp \
      ros-${ROS_DISTRO}-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

# launch ros package
#CMD ["ros2", "launch", "demo_nodes_cpp", "talker_listener.launch.py"]

# Install tools for testing.
RUN apt-get update && apt-get install -y \
	iputils-ping 


ARG USER=user
RUN useradd -ms /bin/bash $USER
USER $USER
COPY --chmod=$USER:$USER docker_user_home /home/$USER
WORKDIR /home/$USER	

