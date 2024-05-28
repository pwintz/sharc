ARG USERNAME=dcuser
ARG PIN_FILENAME=pin-3.15-98253-gb56e429b1-gcc-linux
ARG PIN_ROOT=/home/$USERNAME/$PIN_FILENAME

##################################
############## BASE ##############
##################################

FROM ubuntu:20.04 AS apt-base

# Set the timezone to avoid getting stuck on a prompt when installing packages with apt-get.
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime 

# Update the apt repositories.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y -qq --no-install-recommends build-essential manpages-dev software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update -y -qq

#################################
############## PIN ##############
#################################

FROM apt-base AS PIN 
ARG PIN_FILENAME

# Install required packages
RUN apt-get install -y \
    # Used for downloading and unzipping PIN file
    wget \
    unzip

# Download and unzip the Pin file. 
# The file is placed at /$PIN_FILENAME
RUN wget https://software.intel.com/sites/landingpage/pintool/downloads/$PIN_FILENAME.tar.gz && tar -xzvf $PIN_FILENAME.tar.gz


####################################
############## SCARAB ##############
####################################
FROM apt-base	as scarab

# Load arguments into this stage.
ARG PIN_ROOT
ENV PIN_ROOT $PIN_ROOT

# Install required packages
RUN apt-get install -y \
    python3 \
    python3-pip \
    python2 \
    git \
    # tig \
    # sudo \
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
    vim \
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
ARG PIN_ROOT
ARG PIN_FILENAME
ENV PIN_ROOT /root/$PIN_FILENAME
ENV SCARAB_ENABLE_PT_MEMTRACE 1
ENV LD_LIBRARY_PATH $PIN_ROOT/extras/xed-intel64/lib
ENV LD_LIBRARY_PATH $PIN_ROOT/intel64/runtime/pincrt:$LD_LIBRARY_PATH
ENV SCARAB_ROOT=/scarab

# The root of the Scarab repository, as used by scarab_paths.py (found in scarab/bin/scarab_globals).
ENV SIMDIR=$SCARAB_ROOT

# Copy scarab from the local directory into the image. 
# You must initialize the Git Submodule before this happens!
COPY scarab $SCARAB_ROOT

# Copy PIN file.
COPY --from=PIN /$PIN_FILENAME $PIN_ROOT

# Check that all of the Git submodules are correctly initialized.
RUN test -e $SCARAB_ROOT/src/deps/mbuild && \
    test -e $SCARAB_ROOT/src/deps/xed && \
    test -e $SCARAB_ROOT/src/deps/dynamorio

# Remove the .git file, which indicates that the scarab folder is a submodule. 
# Then, reinitialize the directory as a git repo. 
# Not sure why this is needed, but without it building Scarab fails.
RUN rm $SCARAB_ROOT/.git && cd $SCARAB_ROOT && git init

# Install Scarab Python dependencies
RUN pip3 install -r $SCARAB_ROOT/bin/requirements.txt

RUN cd $SCARAB_ROOT/src && make

# # Build SimPoint 3.2
# # Reference:
# # https://github.com/intel/pinplay-tools/blob/main/pinplay-scripts/PinPointsHome/Linux/bin/Makefile
# RUN cd /home/$USERNAME/ && \
#     wget -O - http://cseweb.ucsd.edu/~calder/simpoint/releases/SimPoint.3.2.tar.gz | tar -x -f - -z && \
#     wget https://raw.githubusercontent.com/intel/pinplay-tools/main/pinplay-scripts/PinPointsHome/Linux/bin/simpoint_modern_gcc.patch -P SimPoint.3.2/ && \
#     patch --directory=SimPoint.3.2 --strip=1 < SimPoint.3.2/simpoint_modern_gcc.patch && \
#     make -C SimPoint.3.2 && \
#     ln -s SimPoint.3.2/bin/simpoint ./simpoint

ENV DOCKER_BUILDKIT 1
ENV COMPOSE_DOCKER_CLI_BUILD 1


FROM apt-base	as base

ARG USERNAME
ENV SCARAB_ROOT=/home/$USERNAME/scarab

RUN apt-get install -y -qq --no-install-recommends \
        build-essential manpages-dev software-properties-common \
        apt-utils \
        lsb-release \
        build-essential \
        software-properties-common \
        ca-certificates \
        gpg-agent \
        # wget \
        # git \
        # cmake \
        lcov \
        gcc-11 \
        g++-11 \
        libomp-dev \
        sudo \
        gosu


# Create a new user '$USERNAME' with password '$USERNAME'
RUN useradd --create-home --home-dir /home/$USERNAME --shell /bin/bash --user-group --groups adm,sudo $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd 

# Authorize SSH Host
RUN mkdir -p /home/$USERNAME/.ssh && \
    chown -R $USERNAME:root /home/$USERNAME/.ssh && \
    chmod 700 /home/$USERNAME/.ssh

##############################
# Non-root user Setup
##############################
RUN echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL >> /etc/sudoers \
    && touch /home/$USERNAME/.sudo_as_admin_successful \
    && gosu $USERNAME mkdir -p /home/$USERNAME/.xdg_runtime_dir
ENV XDG_RUNTIME_DIR=/home/$USERNAME/.xdg_runtime_dir

# Set the working directory
WORKDIR /home/$USERNAME

# Switch to the $USERNAME user
USER $USERNAME

# Copy scarab from the local directory into the image. 
COPY --from=scarab --chown=$USERNAME /scarab $SCARAB_ROOT


#######################
##### ACC EXAMPLE #####
#######################
FROM base	as acc-example-base
ARG USERNAME

USER root

RUN apt-get install -y -qq --no-install-recommends \
    python3 \
    python3-pip \
    # python2 \
    git \
    # tig \
    # sudo \
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
    # vim \
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
RUN apt-get install -y -qq --no-install-recommends \
        libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################
# NL Optimization
##############################
RUN git clone https://github.com/stevengj/nlopt /tmp/nlopt \
    && cd /tmp/nlopt \
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
    && rm -rf /tmp/*


##############################
# OSQP Solver
##############################
RUN git clone --depth 1 --branch v0.6.3 --recursive https://github.com/osqp/osqp /tmp/osqp \
    && cd /tmp/osqp \
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
RUN git clone https://github.com/catchorg/Catch2.git /tmp/Catch2 \
    && cd /tmp/Catch2 \
    && mkdir build \
    && cd build \
    && cmake \ 
        -D BUILD_TESTING=OFF \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/*

# Update the linker to recognize recently added libraries. 
# See: https://stackoverflow.com/questions/480764/linux-error-while-loading-shared-libraries-cannot-open-shared-object-file-no-s
RUN ldconfig

USER $USERNAME

# Copy Bash configurations
COPY --chown=$USERNAME .profile /home/$USERNAME/.bashrc

# COPY requirements.txt /tmp/
# RUN pip install --requirement /tmp/requirements.txt

# Install acc-example python dependencies. 
# TODO: Make this use the acc_example/requirements.txt 
RUN pip3 install gdown numpy scipy pandas
# RUN pip3 install -r acc_example/requirements.txt

COPY --chown=$USERNAME scarabizor.py /home/$USERNAME/resources/scarabizor.py
ENV PYTHONPATH "${PYTHONPATH}:/home/$USERNAME/resources/"

FROM acc-example-base as acc-example

COPY --chown=$USERNAME acc_example /home/$USERNAME/acc_example
# CMD cd ~/acc_example && ls && ./run_examples.py; cat 
RUN cd /home/$USERNAME/acc_example && make acc_controller_7_4
# CMD cd ~/acc_example && make 
CMD cd ~/acc_example && ./run_examples.py; cat controller.log; cat plant_dynamics.log

FROM acc-example-base as acc-example-dev

