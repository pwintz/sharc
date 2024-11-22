## Tutorial: Implementing Custom Dynamics and Controllers

SHARC separates the dynamics and controllers, allowing them to interact independently. The **dynamics module** requires you to define a system derivative function. An ODE solver operates in the background to solve the initial value problem, integrating the dynamics to provide the state evolution over a specified time period. The **controller interface** takes the state and time as inputs, returning the desired control input. For implementation, we use C++ for the controller (simulated with SCARAB) and Python for the dynamics (for ease of use). This tutorial will guide you through creating custom dynamics and controllers.

### Custom Dynamics

To implement custom dynamics:
1. Navigate to the `resources/dynamics/dynamics.py` file.
2. Modify the `CustomDynamics` class.

Some predefined dynamics, like LTI dynamics, are already implemented in `dynamics.py`. You can either use these directly or inherit them. For custom dynamics, you need to modify two specific functions in the `CustomDynamics` class:
- **`setup_system()`**
- **`system_derivative(t, x, u, w)`**

#### `setup_system()`
This function retrieves system parameters that will be used in other functions from the configuration file. For example, for LTI systems:
```python
def setup_system(self):
    self.A      = self.config["system_parameters"]["A"]
    self.B      = self.config["system_parameters"]["B"]
    self.B_dist = self.config["system_parameters"]["B_dist"]
    self.n = self.config["system_parameters"]["state_dimension"]
```
#### `system_derivative(t, x, u, w)`
In the `system_derivative(t, x, u, w)` function you are expected to implement the time derivative of your states. Ensure the returned derivative has the same shape as the state vector:
```python
def system_derivative(self, t, x, u, w):
    dxdt = self.A @ x + self.B @ u
    if w is not None: # If a disturbance is given, then add its effect.
        dxdt += self.B_dist @ w
    assert dxdt.shape == (self.n, 1)
    return dxdt
```

After implementing custom dynamics, you can use it in your example by modifying the following element from the `config.json` file:
```json
  "dynamics_class_name": "CustomDynamics",
```
After this custom dynamics will be automatically used in the calculations.
### Custom Controller
To implement custom controller:
1. Navigate to `resources/controller/CustomController.h` and `resources/controller/CustomController.cpp` files.
2. Modify the `CustomController` class.

Predefined controllers, such as linear and nonlinear MPC controllers, are in the `controllers` directory. You can modify these or create new ones by implementing:
- **`CustomController::setup(const nlohmann::json &json_data)`**
- **`CustomController::calculateControl(int k, double t, const xVec &x, const wVec &w)`**

#### `CustomController::setup(const nlohmann::json &json_data)`
This function retrieves system parameters that will be used in other functions from the configuration file and setups the controller. You can get the controller parameters from your config file here, feel free to define any variables you will use here in the header file `resources/controller/CustomController.h`. For example:
```C++
void CustomController::setup(const nlohmann::json &json_data){
    // Load system parameters
    M = json_data.at("system_parameters").at("M");
    m = json_data.at("system_parameters").at("m");

    sample_time = json_data.at("system_parameters").at("sample_time");
    input_cost_weight = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");

    nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setDiscretizationSamplingTime(sample_time);
    ...
```

#### `CustomController::calculateControl(int k, double t, const xVec &x, const wVec &w)`
This function is the main body of the control calculations at each timestep. Since only this function will be traced for simulation, please keep nothing but the complete control calculations here. You can still reach to other variables of the `CustomController` here and use them. It is a void function that sets the `control` value. For example:
```C++
void NLMPCController::calculateControl(int k, double t, const xVec &x, const wVec &w){
    state = x;
    // Call NLMPC control calculation here
    nlmpc_step_result = nlmpc.optimize(state, control);
    control = nlmpc_step_result.cmd;
    ...
```

Lastly, register the name of your controller at the end of the `CustomController.cpp` file:
```c++
// Register the controller using a name of your choice that will be used in json to call
REGISTER_CONTROLLER("CustomController", CustomController)
```

You can use the registered controller by calling it in `config.json` file by modifying:
```json
  "system_parameters": {
    "controller_type": "CustomController",
    }
```