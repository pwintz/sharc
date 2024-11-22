# Simulator for Hardware Architecture and Real-time Control (SHARC) 

In cyber-physical systems (CPSs), there is a tight coupling between computation, communication, and control. 
Due to the complexity of these systems, advanced design procedures that account for these tight interconnections are paramount to ensure the safe and reliable operation of control algorithms under computational constraints. 
The Simulator for Hardware Architecture and Real-time (SHARC) is a tool to assist in the co-design of control algorithms and the computational hardware on which they are run. 
SHARC simulates a user-specified control algorithm on a user-specified microarchitecture, evaluating how computational constraints affect the performance of the control algorithm and the safety of the physical system.

The [Scarab Simulator](https://github.com/hpsresearchgroup/scarab) is used to simulate the computational hardware. 
Scarab is a microarchitectural simulator, which can simulate the execution of a computer program on different processor and memory hardware than the one the simulation runs on.   
This project uses Scarab to simulate the execution of control feedback in various computational configurations (processor speed, cache size, etc.).
We use the simulation at each time-step to determine the computation time of the controller, which is used in-the-loop to simulate the trajectory of a closed-loop control system that includes the effects of computational delays.

<!-- The setup of Scarab is somewhat difficult, so this project uses Docker to provide a consistent development environment for Scarab so that it can be quickly installed and run by developers and users.  -->

# Getting Started 

SHARC is fully Dockerized, so installation only requires installing Docker, getting a Docker image, and creating a container from that image.
The Docker container can be run three ways:

- Non-interactive Docker 
- Interactive Docker
- Dev-containers

<!-- 
A document (either a webpage, a pdf, or a plain text file) explaining at a minimum:
* What elements of the paper are included in the REP (e.g.: specific figures, tables, etc.).
* Instructions for installing and running the software and extracting the corresponding results. Try to keep this as simple as possible through easy-to-use scripts.
* The system requirements for running the REP (e.g.: OS, compilers, environments, etc.). The document should also include a description of the host platform used to prepare and test the docker image or virtual machine.
* The software and any accompanying data. This should be made available with a link that should remain accessible throughout the review process. Please prepare either a:
  * Docker Image (preferred). 
-->

## Getting SHARC Docker Image

### Pre-built Image

For a limited number of OS and host architectures, a Docker image is provided on Docker Hub. 
To access these images, run 
```
docker pull [TODO]
``` 

### Building an Image


## Running Pre-build SHARC Docker Image 

1. Install Git and ensure SSH is setup. 
1. Install Docker by following the installations instructions for 
[Linux](https://docs.docker.com/engine/install/),
[MacOS](https://docs.docker.com/desktop/setup/install/mac-install/) (M1 chip is not supported), or
[Windows](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Run `docker pull pwintz/sharc:latest` to download the SHARC Docker image from Docker Hub.
4. Set an experiment 
3. Run ```docker docker run --rm -it \
  -v "$(pwd)/resources:/home/dcuser/resources" \
  -v "$(pwd)/examples:/examples" \
  sharc:latest \
  bash -c "cd /examples/cartpole && sharc --config_filename ${CONFIG_FILE}" ```

## Building Docker Image and Running
For platforms where a Docker image is not availble on Docker Hub, it is necessary to build a Docker image. 


## Development in VS Code with Dev-Containers
1. Install Docker, VS Code, and the Dev-Containers VS Code extension.
1. Clone this repository.
    1. Navigate to the folder where you want this project located.
    1. Run `git clone <repository>`, where `<repository>` is a HTTP or SSH reference to this repository.
    1. Within the root of the cloned repository, run `git submodule update --init --recursive` to initialize Git submodules (`scarab` and `libmpc`). 
      ~~Then(?), use `git submodule init` and `git submodule update` to setup the libmpc submodule.~~ (Is this redundant or still necessary?)
1. ~~Currently, it is necessary to manually download the pin file into the `pins/` directory.'~~ (Is this still necessary?)

## Running an Example (Adaptive Cruise Control)
As an introductory example, the `acc_example` folder contains a simulation a vehicle controlled by an adaptive cruise control (ACC) system. 

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

<!-- - `run_sharc.py`: Python script that executes one or more SHARC experiments.  -->
- `scarabizor.py`: Python module that provides tools for reading Scarab statistics.
- `plant_runner.py`: Python module for executing the simulation of the plant for a given system. Handles reading and writing to files for inter-process communication. It should not be called directly.
- `utils.py`: Python module.
<!-- - `scripts/run_portabilize_trace.sh` and `scripts/portabilize_trace.py`: Scripts for pre-processing DynamoRIO traces. -->
<!-- - `scripts/make_sharc_pipes.sh`: Creates pipe files in the working directory that are used to pass information between the controller and plant processes. -->

## Projects Directory Structure

The `examples/` folder contains some example projects that are configured to be simulated using SHARC.
In particular, the `examples/acc_example` folder contains an example of MPC used for adaptive cruise control (ACC) of a road vehicle.
Each project must have the following structure:

- `base_config.json`: A JSON file that defines the settings. Some settings are required by Scarb, but users can add additional configurations in the `base_config.json` that are used by their particular project.
- `simulation_configs/`: A directory containing `default.json` and (optionally) other simulation configuration files. The JSON files in `simulation_configs/` cannot contain any keys (including nested keys) that are not present in `base_config.json`. When `run_sharc.py` is run in the project root (e.g., `acc_example/`), the optional argument `--config_filename` can be used to specify one of the JSON files in `simulation_configs/`, such as `run_sharc.py --config_filename example_configs.json`. Values from `example_configs.json` will be patched onto the values from `base_config.json`, using the values in `base_config.json` as "defaults" and the values in `example_configs.json` when present. Some keys are required in config JSON files, but a given project may add additional keys to define model parameters or other options. [TODO: Write a section describing the requirements for the config json files.]
- `chip_configs/`: A directory containing PARAMS files used to specify hardward parameters, such as clock speed. In the configs json, the key-value pair `"PARAMS_base_file":  "PARAMS.base"` would specify that `chip_configs/PARAMS.base` will be used as the base for the chip parameters (modifications can be made based on other key-value pairs).
- `scripts/controller_delegator.py`: A Python module for building the controller executable based on the needs of the particular system. In the ACC example, CMake is used with various modifications applied at compile time based on the config JSON input. Other build systems can also be used.
- `scripts/plant_dynamics.py`: A Python module that defines the dynamics of the given plant. The dynamics are defined by implementing and returning a function `evolve_state(t0, x0, u, tf)` that takes the initial time and state `t0` and `x0`, a constant control value `u`, and a final time `tf`, and returns the state of the system at `tf`. For the ACC example, the evolution of the system is defined as a differential equation which is numerically evaluated using `scipy.integrate.ode`.

The results of experiments are placed into the `experiments/` folder (creating it if it does not exist).

# Configuration Files

The following example configuration file contains all of the values required by SHARC. Additional settings can be included to set values such as system parameters.
In the following example, C++-style comments are included (`//....`), but these are not permitted in JSON files and should be removed.   
```jsonc
{
  // A human-readable discription of the experiment.
  "label": "base",
  // Setting skip to true causes an experiment in an experiment list to be skipped.
  "skip": false,
  "Simulation Options": {
    // The "in-the-loop_delay_provider" value determines what is used to determine delays. 
    // This should be left as "onestep".
    "in-the-loop_delay_provider": "onestep",
    // Enable parallel simulation.
    "parallel_scarab_simulation": false, 
    // use_fake_delays is deprecated. 
    "use_fake_delays": false,
    // Define the maximum number of batches when running in parallel.
    // Useful for ensuring that a simulation runs within a desired amount n 
    "max_batches": 9999999,
    // The maximum batch size when running in parallel. 
    // The actual batch size is chosen to be the smaller of the following:
    // * max_batch_size
    // * number of CPUs
    // * Number of steps remaining
    "max_batch_size": 9999999
  },
  // Select the module where the dynamics are loaded from.
  "dynamics_module_name": "dynamics.dynamics",
  // Select the name of the dynamics class, which must be located in the dynamics module.
  "dynamics_class_name": "ACCDynamics",
  // Set the maximum number of time steps to run the simulation.
  "n_time_steps": 6,
  // Initial State
  "x0": [0, 20.0, 20.0],
  // Initial control
  "u0": [0.0, 100.0],
  "only_update_control_at_sample_times": true,
  // If fake delays are enabled, then Scarab is not used, instead all computations are assumed to be fast enough except at the 
  // time steps listed. The length of the delay is set to the value in "sample_time_multipliers" multiplied by the sample time.
  "fake_delays": {
    "enable": false,
    "time_steps":              [ 12,  15],
    "sample_time_multipliers": [1.2, 2.2]
  },
  // 
  "computation_delay_model": {
    "computation_delay_slope": 0.001801,
    "computation_delay_y-intercept": 0.147
  },
  "==== Debgugging Levels ====": {
    "debug_program_flow_level": 1,
    "debug_interfile_communication_level": 2,
    "debug_optimizer_stats_level": 0,
    "debug_dynamics_level":        2,
    "debug_configuration_level":   0,
    "debug_build_level":           0, 
    "debug_shell_calls_level":     0, 
    "debug_batching_level":        0, 
    "debug_scarab_level":          0
  },
  "system_parameters": {
    // Select the controller to run. 
    "controller_type": "ACC_Controller",
    "state_dimension": 3, 
    "input_dimension": 2,
    "exogenous_input_dimension": 2,
    "output_dimension": 1,
    "sample_time": 0.2,
    // Human-readable names for variables. Useful for generating plots.
    "x_names": ["p", "h", "v"],
    "u_names": ["F^a", "F^b"], 
    "y_names": ["v"]
  },  
  // Choose the chip configuration file. 
  "PARAMS_base_file": "PARAMS.base",
  // Select values to patch in the chip configuration file. 
  // Each key included here must occur as a parameter in the PARAMS_base_file.
  // A null value indicates that the value in PARAMS_base_file should be used.
  "PARAMS_patch_values": {
    "chip_cycle_time": null,
    "l1_size":         null,
    "icache_size":     null,
    "dcache_size":     null
  }
}
```


# Building Docker Images

To build new Docker images, run 
```
docker build .
```
in the root directory of `sharc` repository. 