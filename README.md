# Simulator for Hardware Architecture and Real-time (SHARC) 
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

- `docker run --rm -it $(docker build -q . --target=mpc-examples-base)` 
- or (to show the build steps), run `docker build . --tag mpc-image --target=mpc-examples && docker run -it --rm mpc-image`. If you want the container to persist, delete `--rm`. When making a persistant container, you may also wish to name it using `--name mpc-container`. You may later delete the container by running `docker rm mpc-container`. To delete the image, run `docker rmi mpc-image`.

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


# SHARC Directory Structure

- `sharc/`: Directory containg scripts and libraries used to execute SHARC.  The structure of `sharc/` is described below "SHARC Directory Structure".
- `examples/`: Directory containing several example projects. The structure of `examples/` is described below "Example Project Directory Structure".
- `.devcontainer/`: Directory containing Dev Container configuration.
- `Dockerfile/`: File that defines the steps for building a Docker image. The Dockerfile is divided into several stages, or "targets", which can be built specifically by running `docker build . --target <target-name>`. Targets that you might want to use are listed here:
  - `scarab`: Configures Scarab (and DynamoRIO) without setting up SHARC or examples. 
  - `mpc-examples-dev`: Sets up several SHARC examples with MPC controllers for development. This is designed to be the target used for to create a Dev Container, but it could be used directly via an interactive Docker container. When using Dev Containers for developement, the project source code is persisted on the host machine and accesible in the `/dev-workspace` folder within the container. This prevents changes to code from being lost each time a new container is created. 
  - `mpc-examples`: Docker target for running a SHARC example without setting up a development environment. Changes to code within a `mpc-examples` container will be lost when the container is deleted. 


## SHARC Directory Structure

The structure of `scaraintheloop/` is still in flux, but as it currently stands, it contains the following:

- `run_sharc.py`: Python script that executes one or more SHARC experiments. 
- `scarabizor.py`: Python module that provides tools for reading Scarab statistics.
- `plant_runner.py`: Python module for executing the simulation of the plant for a given system. Handles reading and writing to files for inter-process communication. It should not be called directly. 
- `utils.py`: Python modulue 
- `scripts/run_portabilize_trace.sh` and `scripts/portabilize_trace.py`: Scripts for pre-processing DynamoRIO traces.
- `scripts/make_sharc_pipes.sh`: Creates pipe files in the working directory that are used to pass information between the controller and plant processes.

## Example Project Directory Structure

The `examples/` folder contains some example projects that are configured to be simulated using SHARC.
In particular, the `examples/acc_example` folder contains an example of MPC used for adaptive cruise control (ACC) of a road vehicle.
Each project must have the following structure:

- `base_config.json`: A JSON file that defines 
- `simulation_configs/`: A directory containing `default.json` and (optionally) other simulation configuration files. The JSON files in `simulation_configs/` cannot contain any keys (including nested keys) that are not present in `base_config.json`. When `run_sharc.py` is run in the project root (e.g., `acc_example/`), the optional argument `--config_filename` can be used to specify one of the JSON files in `simulation_configs/`, such as `run_sharc.py --config_filename example_configs.json`. Values from `example_configs.json` will be patched onto the values from `base_config.json`, using the values in `base_config.json` as "defaults" and the values in `example_configs.json` when present. Some keys are required in config JSON files, but a given project may add additional keys to define model parameters or other options. [TODO: Write a section describing the requirements for the config json files.]
- `chip_configs/`: A directory containing PARAMS files used to specify hardward parameters, such as clock speed. In the configs json, the key-value pair `"PARAMS_base_file":  "PARAMS.base"` would specify that `chip_configs/PARAMS.base` will be used as the base for the chip parameters (modifications can be made based on other key-value pairs).
- `scripts/controller_delegator.py`: A Python module for building the controller executable based on the needs of the particular system. In the ACC example, CMake is used with various modifications applied at compile time based on the config JSON input. Other build systems can also be used.
- `scripts/plant_dynamics.py`: A Python module that defines the dynamics of the given plant. The dynamics are defined by implementing and returning a function `evolve_state(t0, x0, u, tf)` that takes the initial time and state `t0` and `x0`, a constant control value `u`, and a final time `tf`, and returns the state of the system at `tf`. For the ACC example, the evolution of the system is defined as a differential equation which is numerically evaluated using `scipy.integrate.ode`.

The results of experiments are placed into the `experiments/` folder (creating it if it does not exist).