// controller/Controller.h
#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <unordered_map>


// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_WITH_FILE_LOCATION(x) std::cout << __FILE__  << ":" << __LINE__ << ": " << x << std::endl;

class Controller {
public:
    Eigen::VectorXd state;   // Internal state
    Eigen::VectorXd control; // Control input

    // Constructor that initializes state dimensions and calls setup
    Controller(const nlohmann::json &json_data) {
        initializeDimensions(json_data);
    }

    // Default constructor
    Controller() : state(0), control(0) {}

    using CreatorFunc = std::function<Controller*(const nlohmann::json&)>;
    static void registerController(const std::string& name, CreatorFunc creator);
    static Controller* createController(const std::string& name, const nlohmann::json &json_data);

    Eigen::VectorXd getLastControl() {
      return control;
    }

    virtual ~Controller() = default;

    // Controller specific functions
    virtual void setup(const nlohmann::json &json_data) = 0;
    virtual void update_internal_state(const Eigen::VectorXd &x) = 0;
    virtual void calculateControl() = 0;

protected:
    // Method to initialize state and control based on dimensions from JSON
    void initializeDimensions(const nlohmann::json &json_data) {

      int state_dim = json_data.at("system_parameters").at("state_dimension");
      int control_dim = json_data.at("system_parameters").at("input_dimension");

      if (state_dim <= 0 || control_dim <= 0) {
          throw std::invalid_argument("Dimensions must be positive.");
      }

      state.resize(state_dim);
      control.resize(control_dim);
    }

private:
    static std::unordered_map<std::string, CreatorFunc>& getRegistry();
};

// Registration macro
#define REGISTER_CONTROLLER(NAME, TYPE) \
    namespace { \
        const bool registered_##TYPE = []() { \
            Controller::registerController(NAME, [](const nlohmann::json& json_data) -> Controller* { \
                return new TYPE(json_data); \
            }); \
            return true; \
        }(); \
    }
    