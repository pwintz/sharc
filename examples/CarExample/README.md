# This file is to help interperate the CarExample here in Sharc

## File list

- Build - All of the build configurations for the c++ controller files

- Chip_configs - Different configurations that can be inserted into base_config.json to use scarab on a different chip

- Simulation_configs - currently only contains the default configuration

- base_config.json - This is the file that you will edit to make the simulation run

- CMakeLists - Cmake, no more explination needed

- controller_delegator - Returns the executable by running the CMake

- generate_example_figures - run locally with user input for the file path of the experiment

- sim_of_car - 2D simulation of the car with user input for the file path of the experiment

## MPC formulation

The MPC formulation is in /sharc/resources/controllers/src/MPCCar.cpp

This controller uses Nonlinear Model Predictive Control (NMPC).  
At each time step, it solves a finite-horizon optimal control problem and applies only the first control input (receding horizon strategy).

### System Model

The vehicle is modeled using a nonlinear kinematic bicycle model:

x_dot   = v * cos(psi + beta)  
y_dot   = v * sin(psi + beta)  
v_dot   = a  
psi_dot = (v / lr) * sin(beta)

State vector:
x = [x, y, v, psi]

Control input:
u = [a, beta]

---

### Optimization Problem

At each control step, the controller minimizes:

J = sum (k = 0 to N-1):
      (x_k - x_ref)^T Q (x_k - x_ref)
    + u_k^T R u_k
  + (x_N - x_ref)^T Qf (x_N - x_ref)

Subject to:

- Nonlinear vehicle dynamics
- State constraints:  xmin <= x_k <= xmax
- Input constraints:  umin <= u_k <= umax

All weighting matrices, horizons, constraints, and solver parameters are configurable via the JSON file.

## To run this example

First you need to look at the json file to meet your needs
Then you are going to run:
./run_example_apptainer.sh CarExample default.json
^
|
This will run the mpc example witht the current configuration

If you want to run the configuration in parallel mode you must have:
"Simulation Options": {
    "in-the-loop_delay_provider": "onestep",
    "parallel_scarab_simulation": true, 
    ...
}

And serial mode will look like this:
"Simulation Options": {
    "in-the-loop_delay_provider": "execution-driven scarab",
    "parallel_scarab_simulation": false, 
    ...
}

The main things that you will need to modify:
    "n_time_steps": 100,
    "x0": [0, 0, 0, 0],
    "u0": [0.25, 0],
    "lr": 1.738,   <- Distance to the read from center of mass
    "lf": 1.738,   <- Same but to the front
    "constraints": {
    "xmin": [-200, -100, -5, -6.14],
    "xmax": [200, 100, 10, 6.14],
    "umin": [-30, -2],
    "umax": [30, 2]
    },
    "controller_parameters": {
        "terminal_state": [5, 5, 0, 1.6],
        "Q":   [5, 5, 0.5, 100],
        "Qf":  [0 , 0, 0 , 0],
        "Rdiag": [0.01, 0.01],
        "Rd": [0,0]
    },