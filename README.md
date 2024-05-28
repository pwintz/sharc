# Scarab in Docker for Feedback Control Simulation
This project uses Docker to provide a consistent development environment for 
Scarab so that we can simulate the execution of a controller on a given processor in order to determine the time to compute. 
Using the simulated computation time, we simulate the evolution of the control system incorporating the computation delays.

# Setup (in Linux or the Windows Linux Subsystem)
1. Install Docker
2. Install Visual Studio Code
    1.  Install recommended extensions, most notably the Dev Containers extension and (if running on a Windows machine) the WSL extensions.
3. Clone this repository.
  3.1 Run `git submodule update --init --recursive` to initialize Git submodules (`scarab` and `libmpc`)
  Use `git submodule init` and `git submodule update` to setup the libmpc submodule.
4. Currently, it is necessary to manually download the pin file into the `pins/` directory.'
5. Open the repository folder in VS Code and use Dev Containers to build and run the Docker file (via CTRL+SHIFT+P followed by "Dev containers: Build and Open Container"). This will change your VS Code enviroment to running within the Docker container, where Scarab and LibMPC are configured.

# Tools used by this project
* Docker -- Creates easily reproducible enviornments so that we can immediately spin-up new virtual machines that run Scarab and ROS.
* Scarab
* DynamoRIO (Optional - Only needed if doing trace-based simulation with Scarab.)
* [libmpc](https://github.com/nicolapiccinelli/libmpc)

# Development in Dev Container

The context in the dev container (i.e., the folder presented in `/workspaces/ros-docker` in the container) is the root directory of the `ros-docker` project. Changes to the files in /workspaces/ros-docker are synced to the local host (outside of the Docker container) where as changes made in the user's home directory are not.

The contents of the user directory `~` are generated from `ros-docker/docker_user_home` (due to a `COPY` statement in our `Dockerfile`), but changes are not synced to the copy on the local machine. Changes to files in `~` are, however, preserved if Visual Studio Code is closed and restarted because the same Docker image is used each time. If the dev container is rebuilt, however, (via CTRL+SHIFT+P followed by "Dev containers: Rebuild Container"), then changes to `~` are lost. 


## Running MPC with Executable-driven mode
After opening the development container, navigate to `~/code/ros-docker/libmpc_example` and followg the README contained therein.

## Tracing and example with Scarab

After opening the development container, navigate to `~` and run 
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
