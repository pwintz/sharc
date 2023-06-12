# Get the base Ubuntu image from Docker Hub
FROM ros:foxy
ARG USERNAME=default_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# install ros package
RUN apt-get update && apt-get install -y \
      ros-${ROS_DISTRO}-demo-nodes-cpp \
      ros-${ROS_DISTRO}-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

# Install tools for testing.
RUN apt-get update && apt-get install -y \
	iputils-ping 

# Install tools for development.
RUN apt-get update && apt-get install -y \
	tig 

# Update apps on the base image
RUN apt-get -y update && apt-get install -y

# Create a user and copy the contents of this
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
# COPY --chmod=$USERNAME:$USERNAME docker_user_home /home/$USERNAME
WORKDIR /home/$USERNAME	

# Copy the current folder which contains C++ source code to the Docker image under /usr/src
COPY . /home/$USERNAME