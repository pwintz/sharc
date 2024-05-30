ARG USERNAME=dcuser
ARG PIN_NAME=pin-3.15-98253-gb56e429b1-gcc-linux
ARG PIN_ROOT=/$PIN_NAME
ARG SCARAB_ROOT=/scarab
ARG RESOURCES_DIR=/home/$USERNAME/resources/
ARG TIME_ZONE=America/Los_Angeles

##################################
############## BASE ##############
##################################
FROM ubuntu:20.04 AS apt-base
ARG USERNAME
ARG TIME_ZONE

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

# # Authorize SSH Host. Is this neccessary?
RUN mkdir -p /home/$USERNAME/.ssh && \
    chown -R $USERNAME:root /home/$USERNAME/.ssh && \
    chmod 700 /home/$USERNAME/.ssh

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
FROM apt-base as simpoint
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
FROM apt-base	as scarab
ARG PIN_NAME
ARG PIN_ROOT

ENV SCARAB_ENABLE_PT_MEMTRACE 1
ENV LD_LIBRARY_PATH $PIN_ROOT/extras/xed-intel64/lib
ENV LD_LIBRARY_PATH $PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH
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
    g++-9-multilib \
    # # Debugger
    # gdb \
    # doxygen \
    libconfig++-dev \
    bc

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1

#####################################
############# DynamoRIO #############
#####################################
# FROM ubuntu:20.04	as dynamorio
# # DynamoRIO build from source
# RUN git clone --recursive https://github.com/DynamoRIO/dynamorio.git && cd dynamorio && git reset --hard release_10.0.0 && mkdir build && cd build && cmake .. && make -j 40

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

# COPY --chown=$USERNAME run_simpoint_trace.sh /home/$USERNAME/run_simpoint_trace.sh
# COPY --chown=$USERNAME run_scarab.sh /home/$USERNAME/run_scarab.sh
# COPY --chown=$USERNAME gather_fp_pieces.py /home/$USERNAME/gather_fp_pieces.py

# COPY --chown=$USERNAME run_scarab_mode_4.sh /home/$USERNAME/run_scarab_mode_4.sh
# COPY --chown=$USERNAME gather_cluster_results.py /home/$USERNAME/gather_cluster_results.py

######################
#### Setup Scarab ####
######################


# Copy scarab from the local directory into the image. 
# You must initialize the Git Submodule before this happens!
COPY scarab $SCARAB_ROOT

# Check that all of Scarab's Git submodules are correctly initialized.
RUN test -e $SCARAB_ROOT/src/deps/mbuild && \
    test -e $SCARAB_ROOT/src/deps/xed && \
    test -e $SCARAB_ROOT/src/deps/dynamorio

# # Copy PIN file from previous stage.
# COPY --from=PIN $PIN_ROOT $PIN_ROOT

RUN ls $PIN_ROOT
# Check that the $PIN_ROOT directory has the contents we expect.
RUN test -e $PIN_ROOT/source

# Remove the .git file, which indicates that the scarab/ folder is a submodule. 
# Then, reinitialize the directory as a git repo. 
# Not sure why this is needed, but without it building Scarab fails.
RUN rm $SCARAB_ROOT/.git && cd $SCARAB_ROOT && git init

# Install Scarab's Python dependencies
RUN pip3 install -r $SCARAB_ROOT/bin/requirements.txt

# Build Scarab.
RUN cd $SCARAB_ROOT/src && make


FROM scarab	as base

ARG USERNAME

ARG PIN_ROOT
ENV PIN_ROOT $PIN_ROOT

ARG SCARAB_ROOT
ENV SCARAB_ROOT=$SCARAB_ROOT
# The root of the Scarab repository, as used by scarab_paths.py (found in scarab/bin/scarab_globals).
ENV SIMDIR=$SCARAB_ROOT
ENV SCARAB_ENABLE_PT_MEMTRACE 1
ENV LD_LIBRARY_PATH $PIN_ROOT/extras/xed-intel64/lib
ENV LD_LIBRARY_PATH $PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH

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
      libomp-dev

#######################
##### ACC EXAMPLE #####
#######################
FROM base	as acc-example-base
ARG USERNAME
ARG RESOURCES_DIR

RUN apt-get install --assume-yes --quiet=2 --no-install-recommends \
    cmake \
    # binutils \
    # libunwind-dev \
    libboost-dev \
    # zlib1g-dev \
    # libsnappy-dev \
    # liblz4-dev \
    g++-9 \
    g++-9-multilib 
    # # Debugger
    # gdb \
    # doxygen \
    # libconfig++-dev \
    # bc

##############################
# libMPC++
##############################
# Downloads and unzips lipmpc into the home directory.
# USER $USERNAME
# RUN cd ~ && wget https://github.com/nicolapiccinelli/libmpc/archive/refs/tags/0.4.0.tar.gz \
#     && tar -xzvf 0.4.0.tar.gz \
#     && rm 0.4.0.tar.gz


##############################
# Eigen
##############################
RUN apt-get install --assume-yes --quiet=2 --no-install-recommends libeigen3-dev
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/*


##############################
# NL Optimization
##############################
# Get the nlopt code from the GitHub repository. 
# By default, this excludes the .git folder.
ADD https://github.com/stevengj/nlopt.git /tmp/nlopt
RUN cd /tmp/nlopt \
    && mkdir build \
    && cd build \
    && cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D NLOPT_PYTHON=OFF \
        -D NLOPT_OCTAVE=OFF \
        -D NLOPT_MATLAB=OFF \
        -D NLOPT_GUILE=OFF \
        -D NLOPT_SWIG=OFF \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/nlopt


##############################
# OSQP Solver
##############################

ARG OSQP_BRANCH=v0.6.3
ADD https://github.com/osqp/osqp.git#$OSQP_BRANCH /tmp/osqp
# RUN git clone --depth 1 --branch v0.6.3 --recursive https://github.com/osqp/osqp /tmp/osqp \
RUN cd /tmp/osqp \
    && mkdir build \
    && cd build \
    && cmake \ 
        -G "Unix Makefiles" \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/*


##############################
# Catch2 - Testing Framework.
##############################
# RUN git clone https://github.com/catchorg/Catch2.git /tmp/Catch2 \
#     && cd /tmp/Catch2 \
#     && mkdir build \
#     && cd build \
#     && cmake \ 
#         -D BUILD_TESTING=OFF \
#         .. \
#     && make -j$(($(nproc)-1)) \
#     && make install \
#     && rm -rf /tmp/*

# Update the linker to recognize recently added libraries. 
# See: https://stackoverflow.com/questions/480764/linux-error-while-loading-shared-libraries-cannot-open-shared-object-file-no-s
RUN ldconfig

# Copy Bash configurations
COPY --chown=$USERNAME .profile /home/$USERNAME/.bashrc

COPY --chown=$USERNAME scarabizor.py /home/$USERNAME/resources/scarabizor.py
ENV PYTHONPATH "${PYTHONPATH}:${RESOURCES_DIR}"

USER $USERNAME

################################
# DevContainer for acc-example #
################################
FROM acc-example-base as acc-example-dev

##############################################
# Stand-alone acc-example (no dev container) #
##############################################
FROM acc-example-base as acc-example

COPY --chown=$USERNAME acc_example /home/$USERNAME/acc_example

# Set the working directory
WORKDIR /home/$USERNAME/acc_example

RUN make acc_controller_7_4
CMD ./run_examples.py; cat controller.log; cat plant_dynamics.log
