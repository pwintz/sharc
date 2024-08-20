# Scarab Simulation for Feedback Control 
The [Scarab Simulator](https://github.com/hpsresearchgroup/scarab) is a tool for simulating the execution of a computer program on different processor and memory hardware.   
This project uses Scarab to simulate the execution of control feedback in various computational configurations (processor speed, cache size, etc.).
We use the simulation at each time-step to determine the computation time of the controller, which is used in-the-loop to simulate the trajectory of a closed-loop control system that includes the effects of computational delays.

The setup of Scarab is somewhat difficult, so this project uses Docker to provide a consistent development environment for Scarab so that it can be quickly installed and run by developers and users. 

# Getting Started 

## Installation
1. Install Docker
1. Clone this repository.
    1. Navigate to the folder where you want this project located.
    1. Run `git clone <repository>`, where `<repository>` is a HTTP or SSH reference to this repository.
    1. Within the root of the cloned repository, run `git submodule update --init --recursive` to initialize Git submodules (`scarab` and `libmpc`). 
      ~~Then(?), use `git submodule init` and `git submodule update` to setup the libmpc submodule.~~ (Is this redundant or still necessary?)
1. ~~Currently, it is necessary to manually download the pin file into the `pins/` directory.'~~ (Is this still necessary?)

## Running an Example (Adaptive Cruise Control)
As an introductory example, the `acc_example` folder contains for simulating a vehicle controlled by an adaptive cruise control (ACC) system. 

There are two options for how to run the example:

1. Use Docker directly to build and image and run an interactive containerBuild and run a docker un the example in a Docker container or you can 
1. Open a Dev Container environment that uses a Docker container for executing code but persists changes to the code allowing for easy development.

To build the ACC example as a temporary Docker image and open it in the terminal:

- `docker run --rm -it $(docker build -q . --target=acc-example-base)` 
- or (to show the build steps), run `docker build . --tag acc-image --target=acc-example && docker run -it --rm acc-image`. If you want the container to persist, delete `--rm`. When making a persistant container, you may also wish to name it using `--name acc-container`. You may later delete the container by running `docker rm acc-container`. To delete the image, run `docker rmi acc-image`.

1. Build the Docker image for the example by running `docker buildx build
1. Navigate into `acc_example` (`cd acc_example`).


## Development Setup Using VS Code and Dev Containers development (in Linux or the Windows Linux Subsystem)
1. Install Docker
2. Install Visual Studio Code
    1.  Install recommended extensions, most notably the Dev Containers extension and (if running on a Windows machine) the WSL extensions.
3. Clone this repository.
  3.1 Run `git submodule update --init --recursive` to initialize Git submodules (`scarab` and `libmpc`)
  Use `git submodule init` and `git submodule update` to setup the libmpc submodule.
4. Currently, it is necessary to manually download the pin file into the `pins/` directory.'
5. Open the repository folder in VS Code and use Dev Containers to build and run the Docker file (via CTRL+SHIFT+P followed by "Dev containers: Build and Open Container"). This will change your VS Code enviroment to running within the Docker container, where Scarab and LibMPC are configured.

# Software Tools used by this project
* Docker -- Creates easily reproducible environments so that we can immediately spin-up new virtual machines that run Scarab.
* Scarab -- A microarchitectural simulator of computational hardware (e.g., CPU and memory).
* DynamoRIO (Optional - Only needed if doing trace-based simulation with Scarab.)
* [libmpc](https://github.com/nicolapiccinelli/libmpc) (Optional - Used for examples running MPC controllers)

# Development in Dev Container

The context in the dev container (i.e., the folder presented in `/dev-workspace` in the container) is the root directory of the `ros-docker` project. Changes to the files in `/dev-workspace` are synced to the local host (outside of the Docker container) where as changes made in the user's home directory are not.

~~The contents of the user directory `~` are generated from `ros-docker/docker_user_home` (due to a `COPY` statement in our `Dockerfile`), but changes are not synced to the copy on the local machine. Changes to files in `~` are, however, preserved if Visual Studio Code is closed and restarted because the same Docker image is used each time. If the dev container is rebuilt, however, (via CTRL+SHIFT+P followed by "Dev containers: Rebuild Container"), then changes to `~` are lost.~~

## Running MPC with Executable-driven mode
After opening the development container, navigate to `/dev-workspace/libmpc_example` and follow the README contained therein.

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
