# Configurations
ARG USERNAME=dcuser
ARG TIME_ZONE=America/Los_Angeles

# By default, we set WORKSPACE_ROOT to be the user's home directory, but 
# if running using dev-containers, this should be set in .devcontainer/devcontainer.json
# to be the workspaceMount target path. 
ARG WORKSPACE_ROOT=/home/$USERNAME

# The "resources" directory contains files that were developed as a part of SHARC. 
# For development, this directory should persist after a container is closed.
ARG RESOURCES_DIR=$WORKSPACE_ROOT/resources

ARG EXAMPLES_DIR=$WORKSPACE_ROOT/examples

ARG PIN_NAME=pin-3.15-98253-gb56e429b1-gcc-linux
ARG PIN_ROOT=/$PIN_NAME
ARG SCARAB_ROOT=/scarab

ARG DYNAMORIO_VERSION=DynamoRIO-Linux-9.0.19314
ARG DYNAMORIO_HOME=/${DYNAMORIO_VERSION}

ARG LIBMPC_DIR=$RESOURCES_DIR/libmpc

##################################
############## BASE ##############
##################################
FROM ubuntu:20.04 AS apt-base
ARG USERNAME
ARG TIME_ZONE
ARG RESOURCES_DIR
ARG EXAMPLES_DIR

# Environment Variables
ENV RESOURCES_DIR=$RESOURCES_DIR
ENV EXAMPLES_DIR=$EXAMPLES_DIR
ENV CONTROLLERS_DIR="${RESOURCES_DIR}/controllers"
ENV DYNAMICS_DIR="${RESOURCES_DIR}/dynamics"

# Initialize PYTHONPATH
ENV PYTHONPATH=""

# Set the timezone to avoid getting stuck on a prompt when installing packages with apt-get.
RUN ln -fs /usr/share/zoneinfo/$TIME_ZONE /etc/localtime 

# Update the apt repositories.
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt install --assume-yes --quiet=2 --no-install-recommends \ 
    build-essential \
    manpages-dev \
    software-properties-common && \
  add-apt-repository ppa:ubuntu-toolchain-r/test && \
  apt-get update --assume-yes --quiet=2

# Install basic programs that will be used in later stages.
RUN apt-get install --assume-yes --quiet=2 --no-install-recommends \
    python3 \
    python3-pip \
    python2 \
    git \
    tig \
    vim \
    sudo \
    # A tool for simplifying usage of sudo.
    gosu

# Create a new user '$USERNAME' with password '$USERNAME'
RUN useradd --create-home --home-dir /home/$USERNAME --shell /bin/bash --user-group --groups adm,sudo $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd 

# Make sure the user owns their home drive. 
# When running with Dev Conatiners this is needed for some unknown reason.
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# # # Authorize SSH Host. Is this neccessary?
# RUN mkdir -p /home/$USERNAME/.ssh && \
#     chown -R $USERNAME:root /home/$USERNAME/.ssh && \
#     chmod 700 /home/$USERNAME/.ssh

##############################
# Non-root User Setup
##############################
## Let the user have sudo control.
RUN echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL >> /etc/sudoers \
    && touch /home/$USERNAME/.sudo_as_admin_successful
RUN gosu $USERNAME mkdir -p /home/$USERNAME/.xdg_runtime_dir
ENV XDG_RUNTIME_DIR=/home/$USERNAME/.xdg_runtime_dir

############
# SIMPOINT #
############
FROM apt-base AS simpoint
ARG USERNAME

WORKDIR /home/$USERNAME/

# Build SimPoint 3.2
# Reference:
# https://github.com/intel/pinplay-tools/blob/main/pinplay-scripts/PinPointsHome/Linux/bin/Makefile
ADD http://cseweb.ucsd.edu/~calder/simpoint/releases/SimPoint.3.2.tar.gz SimPoint.3.2.tar.gz
RUN  tar --extract --gzip -f SimPoint.3.2.tar.gz

ADD https://raw.githubusercontent.com/intel/pinplay-tools/main/pinplay-scripts/PinPointsHome/Linux/bin/simpoint_modern_gcc.patch SimPoint.3.2/simpoint_modern_gcc.patch
RUN patch --directory=SimPoint.3.2 --strip=1 < SimPoint.3.2/simpoint_modern_gcc.patch && \
    make -C SimPoint.3.2 && \
    ln -s SimPoint.3.2/bin/simpoint ./simpoint

################################
############ SCARAB ############
################################
FROM apt-base	AS scarab
ARG PIN_NAME
ARG PIN_ROOT

ENV SCARAB_ENABLE_PT_MEMTRACE=1
ENV LD_LIBRARY_PATH=$PIN_ROOT/extras/xed-intel64/lib
ENV LD_LIBRARY_PATH=$PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH
ENV SCARAB_ROOT=/scarab
# The root of the Scarab repository, as used by scarab_paths.py (found in scarab/bin/scarab_globals).
ENV SIMDIR=$SCARAB_ROOT

# Install unzip so for unzipping the PIN file
RUN apt-get install --assume-yes unzip

# Download and unzip the Pin file. 
# The file is placed at /$PIN_NAME
ADD https://software.intel.com/sites/landingpage/pintool/downloads/$PIN_NAME.tar.gz $PIN_NAME.tar.gz
RUN tar --extract --gzip -f $PIN_NAME.tar.gz && rm $PIN_NAME.tar.gz

# Check that PIN was downloaded correctly and contains what we expect.
RUN test -e $PIN_ROOT/source

# Install required packages
RUN apt-get install --assume-yes \
    # cmake is used to build Scarab.
    cmake \
    binutils \
    libunwind-dev \
    libboost-dev \
    zlib1g-dev \
    libsnappy-dev \
    liblz4-dev \
    g++-9 \
    # g++-9-multilib \
    # # Debugger
    # gdb \
    # doxygen \
    libconfig++-dev \
    bc

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1

#####################################
############# Simpoint? #############
#####################################

# # Build DynamoRIO package for fingerprint client
# RUN mkdir /home/$USERNAME/dynamorio/package && \
#     cd /home/$USERNAME/dynamorio/package && \
#     ctest -V -S ../make/package.cmake,build=1\;no32
# ENV DYNAMORIO_HOME=/home/$USERNAME/dynamorio/package/build_release-64/

# # Build fingerprint client
# COPY --chown=$USERNAME fingerprint_src /home/$USERNAME/fingerprint_src/
# RUN mkdir /home/$USERNAME/fingerprint_src/build && \
#     cd /home/$USERNAME/fingerprint_src/build && \
#     cmake -DDynamoRIO_DIR=$DYNAMORIO_HOME/cmake .. && \
#     make && \
#     cp ./libfpg.so /home/$USERNAME/libfpg.so

# # Copy workflow simpoint/no_simpoint script
# COPY --chown=$USERNAME utilities.sh /home/$USERNAME/utilities.sh
# COPY --chown=$USERNAME run_clustering.sh /home/$USERNAME/run_clustering.sh
# COPY --chown=$USERNAME run_trace_post_processing.sh /home/$USERNAME/run_trace_post_processing.sh
# 
# COPY --chown=$USERNAME run_simpoint_trace.sh /home/$USERNAME/run_simpoint_trace.sh
# COPY --chown=$USERNAME run_scarab.sh /home/$USERNAME/run_scarab.sh
# COPY --chown=$USERNAME gather_fp_pieces.py /home/$USERNAME/gather_fp_pieces.py

# COPY --chown=$USERNAME run_scarab_mode_4.sh /home/$USERNAME/run_scarab_mode_4.sh
# COPY --chown=$USERNAME gather_cluster_results.py /home/$USERNAME/gather_cluster_results.py

######################
#### Setup Scarab ####
######################

# # Copy scarab from the local directory into the image. 
# You must initialize the Git Submodule before this happens!
# COPY scarab $SCARAB_ROOT
# 
# # Remove the .git file, which indicates that the scarab/ folder is a submodule. 
# # Then, reinitialize the directory as a git repo. 
# # Not sure why this is needed, but without it building Scarab fails.
# RUN rm $SCARAB_ROOT/.git && cd $SCARAB_ROOT && git init

# Download Scarab from GitHub, using the commit with hash "3541854b9e6c0a0ab400246e12b4d4485b5a6e8f" (The full hash must be used---Docker does not support truncated hashes). 
# There is nothing special about this commit, and it should be occasionally update it. 
# We use a fixed commit to ensure that any changes to Scarab don't unexpectedly break SHARC.
ADD https://github.com/Litz-Lab/scarab.git#3b38da01acf86ce35bc1785c97d638e29eab0647 $SCARAB_ROOT

# Then, initialize the directory as a git repo.
# Not sure why this is needed, but without it building Scarab fails.
RUN cd $SCARAB_ROOT && git init

# Check that all of Scarab's Git submodules are correctly initialized.
RUN test -e $SCARAB_ROOT/src/deps/mbuild && \
    test -e $SCARAB_ROOT/src/deps/xed && \
    test -e $SCARAB_ROOT/src/deps/dynamorio

# Check that the $PIN_ROOT directory has the contents we expect.
RUN test -e $PIN_ROOT/source


# Install Scarab's Python dependencies
RUN pip3 install -r $SCARAB_ROOT/bin/requirements.txt

# Build Scarab.
RUN cd $SCARAB_ROOT/src && make

# Add Scarab bin folder to Python path so we can import scarab_globals from Python.
ENV PYTHONPATH="${PYTHONPATH}:${SCARAB_ROOT}/bin"
# Add the Scarab "src" directory to path. 
ENV PATH="${PATH}:$SCARAB_ROOT:$SCARAB_ROOT/src:$SCARAB_ROOT/bin"


###############################
############ SHARC ############
###############################
FROM scarab	AS sharc

ARG USERNAME
ARG RESOURCES_DIR

ARG PIN_ROOT
ENV PIN_ROOT=$PIN_ROOT

ARG SCARAB_ROOT
ENV SCARAB_ROOT=$SCARAB_ROOT
# The root of the Scarab repository, as used by scarab_paths.py (found in scarab/bin/scarab_globals).
ENV SIMDIR=$SCARAB_ROOT
ENV SCARAB_ENABLE_PT_MEMTRACE=1
ENV LD_LIBRARY_PATH=$PIN_ROOT/extras/xed-intel64/lib
ENV LD_LIBRARY_PATH=$PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH

# Copy PIN file.
RUN test -e $PIN_ROOT/source

RUN apt-get install --assume-yes --quiet=2 --no-install-recommends \
      # Manual pages about using GNU/Linux for development
      manpages-dev \
      # apt-utils \
      # lsb-release \
      # Manage the repositories that you install software from
      software-properties-common \
      # Common CA certificates to check for the authenticity of SSL connections
      ca-certificates \
      # Tool for secure communication and data storage.
      gpg-agent \
      # wget \
      # git \
      # cmake \
      # Tool to summarise Code coverage information from GCOV
      lcov \
      gcc-11 \
      g++-11 \
      # OpenMP runtime for managing multiple threads.
      libomp-dev \ 
      man

# RUN yes | unminimize

# Copy Bash configurations
COPY --chown=$USERNAME .profile /home/$USERNAME/.bashrc

COPY resources/sharc/requirements.txt sharc-requirements.txt
RUN pip3 install -r sharc-requirements.txt && rm sharc-requirements.txt

#####################################
############# DynamoRIO #############
#####################################
# DynamoRIO build from source
# RUN git clone --recursive https://github.com/DynamoRIO/dynamorio.git && cd dynamorio && git reset --hard release_10.0.0 && mkdir build && cd build && cmake .. && make
# ADD https://github.com/DynamoRIO/dynamorio.git#release_10.0.0 /home/$USERNAME/dynamorio
# RUN cd /home/$USERNAME/dynamorio && mkdir build && cd build && cmake .. && make

# Set environment variables for the setup
ARG DYNAMORIO_HOME
ENV DYNAMORIO_HOME=$DYNAMORIO_HOME 
ARG DYNAMORIO_VERSION
ENV SCARAB_ENABLE_PT_MEMTRACE=1
ENV SCARAB_ENABLE_MEMTRACE=1 
ENV LD_LIBRARY_PATH=$PIN_ROOT/extras/xed-intel64/lib:$PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH

# Download and extract DynamoRIO
RUN mkdir -p $DYNAMORIO_HOME
ADD https://github.com/DynamoRIO/dynamorio/releases/download/cronbuild-9.0.19314/$DYNAMORIO_VERSION.tar.gz $DYNAMORIO_VERSION.tar.gz
# Extract DynamoRIO in the $DYNAMORIO_HOME directory. We use "--strip-components=1" to remove the top-level directory during extraction---otherwise 
# the resulting path is something like 
#   /DynamoRIO-Linux-9.0.19314/DynamoRIO-Linux-9.0.19314/<files>, 
# whereas what we actually want is
#   /DynamoRIO-Linux-9.0.19314/<files>
RUN tar --extract --gzip --verbose --strip-components=1 -f $DYNAMORIO_VERSION.tar.gz --directory $DYNAMORIO_HOME && rm $DYNAMORIO_VERSION.tar.gz 


# Copy the resources directory. We do this after the other installation tasks because every time the resources directory changes, Docker has to repeat all of the tasks below this line. 
COPY --chown=$USERNAME resources $RESOURCES_DIR

### Setup the PATH and PYTHONPATH ###  
ENV PATH="${PATH}:${RESOURCES_DIR}/sharc:${RESOURCES_DIR}/sharc/scripts"
ENV PYTHONPATH="${PYTHONPATH}:${RESOURCES_DIR}"

########################
##### MPC EXAMPLES #####
########################
FROM sharc AS examples
ARG USERNAME
ARG WORKSPACE_ROOT
ARG RESOURCES_DIR
ARG EXAMPLES_DIR

RUN apt-get install --assume-yes --quiet=2 --no-install-recommends \
    # CMake build toolchain
    cmake \
    # Boost C++ general-purpose library
    libboost-dev \
    # GNU Compiler
    g++-9 \
    # g++-9-multilib \
    # SFML is used(?) by the inverted-pendulum example
    libsfml-dev

COPY examples/acc_example/requirements.txt acc-requirements.txt
RUN pip3 install -r acc-requirements.txt && rm acc-requirements.txt

##############################
# libMPC++
##############################
ARG LIBMPC_DIR 
ENV LIBMPC_DIR=$LIBMPC_DIR 
ADD https://github.com/pwintz/libmpc.git $LIBMPC_DIR
RUN $LIBMPC_DIR/configure.sh
RUN mkdir $LIBMPC_DIR/build && cd $LIBMPC_DIR/build && cmake .. && cmake --install .

# Copy the example folders.
COPY --chown=$USERNAME examples $EXAMPLES_DIR

# Check that all of the expected directories exist.
ARG PIN_ROOT
ARG RESOURCES_DIR
ARG SCARAB_ROOT
ARG DYNAMORIO_HOME
RUN test -e $PIN_ROOT/source \
    && test -e $RESOURCES_DIR    \
    && test -e $SCARAB_ROOT      \
    && test -e $DYNAMORIO_HOME   \
    && test -e $LIBMPC_DIR/build \
    && test -e $EXAMPLES_DIR/acc_example

USER $USERNAME
WORKDIR $EXAMPLES_DIR

# ###################################
# ## DevContainer for mpc-examples ##
# ###################################
# FROM mpc-examples-base AS mpc-examples-dev
# ARG USERNAME
# ARG WORKSPACE_ROOT
# ARG RESOURCES_DIR
# ARG EXAMPLES_DIR
# 
# # # Create a Simlink from the 
# # RUN ln -s /dev-workspace/resources $RESOURCES_DIR
# # run rm -r 
# 
# 
# # For convenience, set the working to the ACC example. 
# WORKDIR $EXAMPLES_DIR/acc_example
# 
# #################################################
# ## Stand-alone mpc-examples (no dev container) ##
# #################################################
# FROM mpc-examples-base AS mpc-examples
# ARG RESOURCES_DIR
# ARG WORKSPACE_ROOT
# ARG EXAMPLES_DIR
# 
# # At this point, we have already created the resources directory, so we cannot copy the whole thing, but we want the local contents to be added to the existing remove directory. 
# # COPY --chown=$USERNAME resources/controllers $RESOURCES_DIR/controllers
# # COPY --chown=$USERNAME resources/dynamics $RESOURCES_DIR/dynamics
# # COPY --chown=$USERNAME resources/include $RESOURCES_DIR/include
# # COPY --chown=$USERNAME resources/sharc $RESOURCES_DIR/sharc
# # ENV CONTROLLERS_DIR=$RESOURCES_DIR/controllers
# # ENV DYNAMICS_DIR=$RESOURCES_DIR/dynamics
# # Set the working directory
# WORKDIR $EXAMPLES_DIR/acc_example