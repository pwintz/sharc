# Get the base Ubuntu image from Docker Hub. 
# The Ubuntu version for "ros:foxy" is 20.04. 
FROM ros:foxy

##### SCARAB SETUP #####
# We do Scarab setup first because it takes a long time.

COPY scarab scarab 
COPY pins pins 

RUN apt-get update 
RUN apt-get install -y \
	python3-pip \
	python2 # Required by run_portabilize_trace.sh

RUN pip install -r /scarab/bin/requirements.txt

# Install tzdata separately with variables defined to prevent prompts during the installation.
RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angelos apt-get -y install tzdata
RUN apt-get install -y \
	clang \
	g++ \
	gpp \
	git \
	cmake \
	vim \
	libsnappy-dev \
	libconfig++-dev

ENV PIN_ROOT=/pins/pinplay-drdebug-3.5-pin-3.5-97503-gac534ca30-gcc-linux
ENV SCARAB_ENABLE_MEMTRACE=1

RUN cd scarab/src && make
  
# Add the DynamoRIO binaries directory to path.
ENV PATH /scarab/src/build/opt/deps/dynamorio/bin64:$PATH
# Add the directory containing the Scarab binary to the path.
ENV PATH /scarab/src/build/opt:$PATH

# Define environment variables that point to the root 
# directories of DynamoRIO and Scarab. 
ENV DYNAMORIO_ROOT=/scarab/src/build/opt/deps/dynamorio
ENV SCARAB_ROOT=/scarab
ENV SCARAB_BUILD_DIR=/scarab/src/build/opt
ENV SCARAB_OUT_PATH=/workspaces/ros-docker/scarab_out
ENV TRACES_PATH=/workspaces/ros-docker/traces

# Copy files, giving ownership to the user (instead of root).
# COPY --chown=$USERNAME:$USERNAME docker_user_home /home/$USERNAME

##### ROS2 SETUP #####
# Install ros package
RUN apt-get update && apt-get install -y \
      ros-${ROS_DISTRO}-demo-nodes-cpp \
      ros-${ROS_DISTRO}-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

##### CONFIGURATION ###
ARG USERNAME=default_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

##### GENERAL SETUP #####

RUN apt-get -y update && apt-get install -y \
	# Install C libraries for numerical computing
	libeigen3-dev \
	libgsl-dev \
	# Install tools for testing.
	iputils-ping \
	# Install tools for development.	
	tig 

# Create a user and copy the contents of this
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
# COPY --chmod=$USERNAME:$USERNAME docker_user_home /home/$USERNAME
# COPY --chown=$USERNAME:$USERNAME docker_user_home /home/$USERNAME
# COPY --chown=$USERNAME:$USERNAME .profile /home/$USERNAME/.profile

# Copy the ".profile" file to the home directory in the container so that we have our aliases available.
COPY --chown=$USERNAME:$USERNAME .profile /home/$USERNAME/.bashrc

# COPY --chown=$USERNAME:$USERNAME src /home/$USERNAME/src
# WORKDIR /home/$USERNAME	

# Copy the current folder which contains C++ source code to the Docker image under /usr/src
# COPY --chmod=$US./docker_user_home /home/$USERNAME