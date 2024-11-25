# Simulator for Hardware Architecture and Real-time Control (SHARC) 


![SHARC](https://github.com/user-attachments/assets/93d6d0fc-bfcf-4780-91d3-e4b2941cb94f)


In cyber-physical systems (CPSs), computation, communication, and control are tightly coupled. 
Due to the complexity of these systems, advanced design procedures that account for these tight interconnections are vital to ensure safe and reliable operation of control algorithms under computational constraints. 
The Simulator for Hardware Architecture and Real-time (SHARC) is a tool to assist in the co-design of control algorithms and the computational hardware on which they are run. 
SHARC simulates a user-specified control algorithm on a user-specified microarchitecture, evaluating how computational constraints affect the performance of the control algorithm and the safety of the physical system.

The [Scarab Simulator](https://github.com/hpsresearchgroup/scarab) is used to simulate the computational hardware. 
Scarab is a microarchitecture simulator, which can simulate the execution of a computer program on different processor and memory hardware than the one the simulation runs on.   
This project uses Scarab to simulate the execution of control feedback in various computational configurations (processor speed, cache size, etc.).
We use the simulation at each time-step to determine the computation time of the controller, which is used in-the-loop to simulate the trajectory of a closed-loop control system that includes the effects of computational delays.

<!-- The setup of Scarab is somewhat difficult, so this project uses Docker to provide a consistent development environment for Scarab so that it can be quickly installed and run by developers and users.  -->

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Repeatability Evaluation Package](#rep)
3. [🚀 Quick Start](#quick-start)
4. [Getting Started](#getting-started)
    - [Obtaining the SHARC Docker Image](#obtaining-the-sharc-docker-image)
    - [Creating a SHARC Docker Container from an Image](#creating-a-sharc-docker-container-from-an-image)
5. [Example: Adaptive Cruise Control](#example-adaptive-cruise-control)
6. [Configuration Files](#configuration-files)
7. [Testing](#testing)
8. [Directory Structure](#directory-structure)
9. [Troubleshooting](#troubleshooting)

---

# Overview

SHARC simulates control feedback loops with computational delays introduced by hardware constraints. It uses the [Scarab Simulator](https://github.com/hpsresearchgroup/scarab) to model hardware performance, incorporating parameters such as processor speed, cache size, and memory latency. The tool supports both serial and parallel simulations for efficient modeling.

Key Features:
- 🖥️ Simulate control feedback loops with computational delays
- 🏗️ Model hardware performance using Scarab Simulator
- 🚀 Support for both serial and parallel simulations
- 🐳 Fully Dockerized for consistent and reproducible environments

---
# <a id="requirements"></a> Requirements

Before you begin, ensure your system meets the following requirements:

- **Supported Architecture**  
  The SCARAB simulator is currently **incompatible with ARM architectures**.
  
- **Docker**  
  SHARC operates within a Docker container. Install Docker by following the appropriate instructions for your platform:  
  - [Linux](https://docs.docker.com/engine/install/)  
  - [MacOS](https://docs.docker.com/desktop/setup/install/mac-install/)  
  - [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)  

- **Git**  
  Install [Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git) and ensure SSH is set up if you plan to build the Docker image yourself. Follow these steps to [generate and configure an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

---


# <a id="rep"></a> Repeatability Evaluation Package

To install the SHARC simulator and repeat the experiments in the submitted manuscript, do the following steps:

1. **Check Requirements**  
  Check that your system satisfies SHARC's [requirements](#requirements), listed above. 

1. **Clone the Repository**  
   Clone the SHARC repository:  
   ```bash
   git clone git@github.com:pwintz/sharc.git && cd sharc
   ```

2. **Run Setup Script**  
  Execute the `setup_sharc.sh` script in the root of the `sharc` directory: <!-- [`setup_sharc.sh`](https://github.com/pwintz/sharc/blob/quick-start/sharc_setup_and_run.sh)    script.  -->
   ```
   ./setup_sharc.sh
   ```
   This script offers you a choice to either pull a SHARC Docker image from Docker Hub or build a Docker image locally.
   After an image is available, the script starts a container and runs a suite of (quick) automated tests.
   Once the tests finish, you will have an option to enter a temporary interactive SHARC container where you can explore the file system. 
   (Any changes made inside this container are lost when you exit.)  
   
3. **Run Adaptive Cruise Control (ACC) Example**  
   Once a SHARC Docker image is available, examples can be run using a collection of scripts in the `repeatability_evaluation/` folder. 
   For a quick initial example, run the following command on the host machine (not in a Docker container): <pre>
   cd repeatability_evaluation/ 
   ./run_acc_example_with_fake_delays.sh
   </pre>
   The results of the simulation will appear `repeatability_evaluation/acc_example_experiments/` on the host machine. 
   The `fake_data` configuration is useful for testing, but does not use the Scarab microarchitecture simulator to determine computation times, resulting in a much faster test. 
   To run the ACC example using the Scarab simulator, execute the following command (This can take several hours, depending on your system): <pre>
   cd repeatability_evaluation/  # Unless already in folder.
   ./run_acc_example_parallel_vs_serial.sh
   </pre>
   After the simulation finishes, the `repeatability_evaluation/acc_example_experiments/` folder will contain an image matching Figure 5 in the submitted manuscript.

4. **Run Cart Pole Example**  
  The Cart Pole example uses nonlinear MPC, which results in long computation times, requiring over 24 hours to complete. 
  To start the simulation, run
  ```
  ./run_example_in_container.sh cartpole default.json
  ```
  while in the `repeatability_evaluation` folder. 

# <a id="quick-start"></a>🚀 Quick Start

Get SHARC up and running in two simple steps:

1. **Clone the Repository**  
   Clone the SHARC repository:  
   ```bash
   git clone git@github.com:pwintz/sharc.git && cd sharc
   ```

2. **Run the Setup Script**  
   Execute the `setup_sharc.sh` script. <!-- [`setup_sharc.sh`](https://github.com/pwintz/sharc/blob/quick-start/sharc_setup_and_run.sh)    script.  -->This script offers you a choice to either pull a SHARC Docker image from Docker Hub or build a Docker image locally.
   After an image is available, the script starts a container and runs a suite of (quick) automated tests.
   Once the tests finish, you will have an option to enter a temporary interactive SHARC container where you can explore the file system. 
   (Any changes made inside this container are lost when you exit.)  
   

# Getting Started 

SHARC is fully Dockerized, so installation only requires installing Docker, getting a Docker image, and starting a container from that image.
The Docker container can be run in three ways:

- Dev-containers — Useful for the development of SHARC projects or SHARC itself. Dev containers allow developers to connect to a Docker container using VS Code and automatically persist changes to source code.
- Interactive Docker — Useful for manually interacting with the Docker container in environments where dev containers are not supported or configured.
- Non-interactive Docker — Useful for automated running of simulations.

<!-- 
A document (either a webpage, a PDF, or a plain text file) explaining at a minimum:
* What elements of the paper are included in the REP (e.g.: specific figures, tables, etc.).
* Instructions for installing and running the software and extracting the corresponding results. Try to keep this as simple as possible through easy-to-use scripts.
* The system requirements for running the REP (e.g.: OS, compilers, environments, etc.). The document should also include a description of the host platform used to prepare and test the docker image or virtual machine.
* The software and any accompanying data. This should be made available with a link that should remain accessible throughout the review process. Please prepare either a:
  * Docker Image (preferred). 
-->

## Obtaining the SHARC Docker Image

To get a SHARC Docker image, you can either pull a pre-build image from Docker Hub or build your own image using the provided Dockerfile.
While the pre-build image is generally easier, only images for the Linux operating system are available.
If your OS or architecture is not available, then follow the instructions for building an image.

### Pre-built Image

For a limited number of OS and host architectures, a Docker image is provided on Docker Hub. 
To download the latest SHARC Docker image, run 
```bash
docker pull pwintz/sharc:latest
``` 

Troubleshooting: If you get an error that says, `Error response from daemon: manifest for pwintz/sharc:latest not found: manifest unknown: manifest unknown`, then

* check for typos 
* check your selected tag exists [here](https://hub.docker.com/repository/docker/pwintz/sharc/general)
* try logging in to Docker Hub via the command line:
```bash
docker login
```

### Building an Image

For platforms where a Docker image is not available on Docker Hub, it is necessary to build a Docker image. 
To build an image, you must install [Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git) and ensure [SSH is setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for authenticating with GitHub. 
1. Clone this repository.
    1. Navigate to the folder where you want this project located.
    1. Run `git clone git@github.com:pwintz/sharc.git`.
    1. Run `git submodule update --init --recursive`.
2. Change your working directory to the newly created `sharc/` folder. Inside you should see a file named `Dockerfile`.
3. Inside `'sharc/`, run `docker build --tag sharc:my_tag .`, where `my_tag` can changed to an identifier of your choosing. Note the "`.`" at the end of the `docker build` command, which tells Docker to build the `Dockerfile` in the current directory. 

Warning: Each time you run `docker build`, it adds another Docker image to your system. For SHARC, each image is 5 GB, so you can quickly fill up your hard drive! To build without saving the image, use `docker build --rm [rest of the command]`. If you have previously built several images, you can cleanup unused ones by running `docker image prune`.

## Creating a SHARC Docker Container from an Image
You should now have a SHARC Docker image on your system that is either named `pwintz/sharc` (if you pulled from Docker Hub) or `sharc`, if you built locally. It will also have a tag such as `latest` or `my_tag`. 
For simplicity, we will assume the image name and tag are `sharc:my_tag` from here on out.
To check that your image is in fact available run 
```bash
docker images
```
Now that you have an image, you need to create a Docker container—that is, a virtual environment initialized using the system state contained in the Docker image.

As mentioned above, you can create interactive or non-interactive containers, or open a container using a dev-container. 

### Interactive Docker Container
When you open an interactive container, your current terminal changes to be inside the container where you can run commands and explore the container's file system. 
Changes to the container's files will persist throughout the life of the container, but are not automatically saved on the host file system and will be lost if the container is deleted.  

To create and start a container in interactive mode from the image `sharc:my_tag`, run 
```
docker run -it sharc:my_tag
```
To leave the container, run `exit`. 

Each time you run `docker run`, it creates a new container.
Just like building Docker images, creating many containers will quickly fill up your hard drive.
To avoid saving the container after you exit, add "`--rm`" before the image name in the `docker run` command.
You can also delete all stopped containers by running `docker containers prune`.

Files on your host machine can be made accessible within a container as a "volume" by adding 
```
-v "<path_on_host>:<path_in_container>"
```
to the `docker run` command. 
Changes made to a volume inside a container are persisted on the host machine after the container is closed and deleted.

### Non-interactive Docker Container

To create and start a container in non-interactive mode from the image `sharc:my_tag`, run 
```
docker run --rm sharc:my_tag <command_to_run>
```
(without `-it`). 
The `--rm` argument ensures that the container is deleted after execution. 
The initial working directory of the sharc images is the `examples/` folder. 
To run the ACC example, `<command_to_run>` can be replaced by "cd acc_example && sharc --config_filename fake_delays.json", resulting in 
```
docker run --rm sharc:my_tag bash -c "cd acc_example && sharc --config_filename fake_delays.json"
```

To access the results of the simulation, create a volume for the container and copy the results of the simulation into the volume path.

### Development in VS Code with Dev-Containers
1. Install Docker and VS Code.
2. Open the `sharc` repository directory in VS Code. VS Code should prompt you to install recommended extensions, including `dev-containers`. Accept this suggestion. 
3. Use Dev Containers to build and run the Docker file (via CTRL+SHIFT+P followed by "Dev containers: Build and Open Container"). This will change your VS Code environment to running within the Docker container, where Scarab and libmpc are configured.

<!-- ## Development Setup Using VS Code and Dev Containers development (in Linux or the Windows Linux Subsystem) -->
<!-- 1. Install Docker -->
<!-- 2. Install Visual Studio Code -->
<!-- 1.  Install recommended extensions, most notably the Dev Containers extension and (if running on a Windows machine) the WSL extensions. -->
<!-- 3. Clone this repository. -->
  <!-- 3.1 Run `git submodule update --init --recursive` to initialize Git submodules (`scarab` and `libmpc`) -->
  <!-- Use `git submodule init` and `git submodule update` to setup the libmpc submodule. -->
<!-- 5. Open the repository folder in VS Code and  -->

## Example: Adaptive Cruise Control
As an introductory example, the `acc_example` folder contains a simulation of a vehicle controlled by an adaptive cruise control (ACC) system. 

<!-- There are two options for how to run the example:

1. Use Docker directly to build an image and run an interactive container to run the example in a Docker container or you can 
1. Open a Dev Container environment that uses a Docker container for executing code but persists changes to the code allowing for easy development. -->

To run the ACC example, use one of the methods above for [creating a SHARC Docker container](#creating-a-sharc-docker-container-from-an-image). 
<!-- To build the ACC example as a temporary Docker image and open it in the terminal: -->
Within the container, navigate to `examples/acc_example` and run 
```bash
sharc --config_file fake_delays.json
```
The `fake_delays.json` file is located in `examples/acc_example/simulation_configs/`, and defines configurations for quickly running the example in the serial and parallel mode without actually executing the microarchitectural simulations with Scarab. 
Select other configuration files in `examples/acc_example/simulation_configs/` to explore the settings available. 
For configurations that use Scarab, the time to execute the simulation can range from minutes to hours depending on the number of sample times, the number of iterations used by the MPC optimization algorithm, the prediction and control horizons, and whether parallel or serial simulation is used. 

The results of the simulation are saved into `examples/acc_example/experiments/`.
Within the appropriate experiment directory, a file named `experiment_list_data_incremental.json` will be populated incrementally during the simulation, so that you can monitor the progress of the simulation. 
At the end of the simulation, `experiment_list_data_incremental.json` is copied to `experiment_list_data.json`. 

A Jupyter notebook `make_plots.ipynb` is located within `acc_example/` for generating plots based on the last experiment.

<!-- - `docker run --rm -it $(docker build -q . --target=examples)` 
- or (to show the build steps), run `docker build . --tag mpc-image --target=mpc-examples && docker run -it --rm mpc-image`.
If you want the container to persist, delete `--rm`. When making a persistent container, you may also wish to name it using `--name mpc-container`. You may later delete the container by running `docker rm mpc-container`. To delete the image, run `docker rmi mpc-image`. -->

# Development in Dev Container

The context in the dev container (i.e., the folder presented in `/dev-workspace` in the container) is the root directory of the `ros-docker` project. Changes to the files in `/dev-workspace` are synced to the local host (outside of the Docker container) whereas changes made in the user's home directory are not.


# Directory Structure

<!-- - `.devcontainer/`: Directory containing Dev Container configuration. -->
- `docs/`: Documentation files.
- `examples/`: Directory containing several example projects. The structure of `examples/` is described in ["Project Directory Structure"](#projects-directory-structure).
- `resources/`: Directory containing files and folders that are included in Docker images, include 
  - `controllers/`: C++ Source code for controllers
  - `dynamics/`: Python code for dynamics
  - `include/`: C++ header files
  - `sharc/`: Python `sharc` package and subpackages.
<!-- - `sharc/`: Directory containing scripts and libraries used to execute SHARC.  The structure of `sharc/` is described below "SHARC Directory Structure". -->
- `Dockerfile/`: File that defines the steps for building a Docker image. The Dockerfile is divided into several stages, or "targets", which can be built specifically by running `docker build. --target <target-name>`. Targets that you might want to use are listed here:
  - `scarab`: Configures Scarab (and DynamoRIO) without setting up SHARC or examples. 
  - `sharc`: Sets up SHARC and its dependencies.
  - `examples`: Sets up several SHARC examples. 
  <!-- This is designed to be the target used to create a Dev Container, but it could be used directly via an interactive Docker container. When using Dev Containers for development, the project source code is persisted on the host machine and accessible in the `/dev-workspace` folder within the container. This prevents changes to code from being lost each time a new container is created.  -->
  <!-- - `mpc-examples`: Docker target for running a SHARC example without setting up a development environment. Changes to code within a `mpc-examples` container will be lost when the container is deleted.  -->
  By default, running `docker build .` will build the last target in the Dockerfile, which is `examples`.


<!-- ## SHARC Python Package

The structure of the `sharc` package is as follows:

` `sharc`: The entry point for the Sharc simulator. Handles setting up and running simulations.
- `scarabizor.py`: Python module that provides tools for reading Scarab statistics.
- `plant_runner.py`: Python module for executing the simulation of the plant for a given system. Handles reading and writing to files for inter-process communication. It should not be called directly.
- `utils.py`: Python module. -->
<!-- - `scripts/run_portabilize_trace.sh` and `scripts/portabilize_trace.py`: Scripts for pre-processing DynamoRIO traces. -->
<!-- - `scripts/make_sharc_pipes.sh`: Creates pipe files in the working directory that are used to pass information between the controller and plant processes. -->

## Projects Directory Structure

The `examples/` folder contains some example SHARC projects that are configured to be simulated using SHARC.
<!-- In particular, the `examples/acc_example` folder contains an example of MPC used for adaptive cruise control (ACC) of a road vehicle. -->
Each SHARC project must have the following structure:

- The controller and dynamics are defined as described in [Tutorial: Implementing Custom Dynamics and Controllers.md](docs/Tutorial:%20Implementing%20Custom%20Dynamics%20and%20Controllers.md).
- `base_config.json`: A JSON file that defines the settings. Some settings are required by Scarab, but users can add additional configurations in the `base_config.json` that are used by their particular project.
- `simulation_configs/`: A directory containing `default.json` and (optionally) other simulation configuration files. The JSON files in `simulation_configs/` cannot contain any keys (including nested keys) that are not present in `base_config.json`. When `sharc` is run in the project root (e.g., `examples/acc_example/`), the optional argument `--config_filename` can be used to specify one of the JSON files in `simulation_configs/`, such as `run_sharc.py --config_filename example_configs.json`. Values from `example_configs.json` will be patched onto the values from `base_config.json`, using the values in `base_config.json` as "defaults" and the values in `example_configs.json` when present. Some keys are required in config JSON files, but a given project may add additional keys to define model parameters or other options. [TODO: Write a section describing the requirements for the config json files.]
- `chip_configs/`: A directory containing PARAMS files used to specify hardware parameters, such as clock speed. In the configuration JSON files, the key-value pair `"PARAMS_base_file":  "PARAMS.base"` would specify to use `chip_configs/PARAMS.base` as the base for the chip parameters (modifications can be made based on other key-value pairs).
- `controller_delegator.py`: A Python module for building the controller executable based on the needs of the particular system. In the ACC example, CMake is used with various modifications applied at compile time based on the config JSON input. Other build systems can also be used.
<!-- - `scripts/plant_dynamics.py`: A Python module that defines the dynamics of the given plant. The dynamics are defined by implementing and returning a function `evolve_state(t0, x0, u, tf)` that takes the initial time and state `t0` and `x0`, a constant control value `u`, and a final time `tf`, and returns the state of the system at `tf`. For the ACC example, the evolution of the system is defined as a differential equation which is numerically evaluated using `scipy.integrate.ode`. -->

The results of experiments are placed into the `experiments/` folder (the folder will be created if it does not exist).

# Configuration Files
SHARC uses JSON configuration files to customize simulations. Key configurations include:
- Simulation mode (serial/parallel)
- Dynamics and controller selection
- Hardware parameters
- Debugging levels

The following example configuration file contains all of the values required by SHARC. Additional settings can be included to set values such as system parameters.
In the following example, C++-style comments are used (`//....`), but these are not permitted in JSON files and should be removed.   
```jsonc
{
  // A human-readable description of the experiment.
  "label": "base",
  "skip": false,
  "Simulation Options": {
    // Select serial or parallel simulation.
    "parallel_scarab_simulation": false, 
    // The "in-the-loop_delay_provider" value determines what is 
    // used to determine delays. 
    // This should be set to "onestep" when using the parallel mode
    // and "execution-driven scarab" when using serial mode.
    "in-the-loop_delay_provider": "onestep",
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
  // Select the name of the dynamics class, which must be located 
  // in the dynamics module.
  "dynamics_class_name": "ACCDynamics",
  // Set the maximum number of time steps to run the simulation.
  "n_time_steps": 6,
  // Initial State
  "x0": [0, 20.0, 20.0],
  // Initial control
  "u0": [0.0, 100.0],
  "only_update_control_at_sample_times": true,
  // If fake delays are enabled, then Scarab is not used to determine 
  // computational delays, instead all computations are assumed to be 
  // fast enough except for those at the time steps listed. The length of 
  // the delay is set to the value in "sample_time_multipliers" multiplied 
  // by the sample time.
  "fake_delays": {
    "enable": false,
    "time_steps":              [ 12,  15],
    "sample_time_multipliers": [1.2, 2.2]
  },
  "==== Debugging Levels ====": {
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

# Testing
SHARC comes with a suite of automated testing to check its correctness. 
Unit tests (fast tests of small pieces of the software) are located in 
```
<sharc_root>/tests
```
To run all unit tests, change to the `tests/` directory and run `./run_all.sh`.

# Update Docker Hub Images
To update a [pwintz/sharc](https://hub.docker.com/repository/docker/pwintz/sharc/general) Docker image on Docker Hub use the following commands:
```bash
docker login
docker build --tag sharc:latest .
docker tag sharc:latest pwintz/sharc:latest
docker push pwintz/sharc:latest
```

# Troubleshooting

## Problem: Docker Build Fails
* Make sure you are connected to the internet. 
* Make sure that SSH is setup in your environment.
* Check that you are running `docker build .` in the directory that contains the `Dockerfile`.
* If all else fails, try building from scratch, discarding the Docker cache, by running `docker build --no-cache .` 

## Problem Docker Push Fails
* Log-in using `docker login` before pushing. 
* Tag the image with the user name as a prefix in the form `username/tag` so that Docker knows where to direct the pushed image.

## Runnning a Serial SHARC Simulation Fails
When running serial simulations in Docker, the following error occurs in certain circmstances:
```
setarch: failed to set personality to x86_64: Operation not permitted
```
The reason this error occurs is because the Docker container does not allow this operations, by default. 
To fix the problem, use the `--privileged` flag when starting the Docker container. 


# Software Tools used by this project
* [Docker](https://www.docker.com/) -- Creates easily reproducible environments so that we can immediately spin-up new virtual machines that run Scarab.
* [Scarab](https://github.com/Litz-Lab/scarab) -- A microarchitectural simulator of computational hardware (e.g., CPU and memory).
* [DynamoRIO](https://dynamorio.org/) (Optional - Only needed if doing trace-based simulation with Scarab.)
* [libmpc](https://github.com/nicolapiccinelli/libmpc) (Optional - Used for examples running MPC controllers)
