// controller/Controller.h
#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <unordered_map>
#include "nlohmann/json.hpp"

using xVec = Eigen::Matrix<double, TNX, 1>;
using uVec = Eigen::Matrix<double, TNU, 1>;
using wVec = Eigen::Matrix<double, TNDU, 1>;
using yVec = Eigen::Matrix<double, TNY, 1>;

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_WITH_FILE_LOCATION(x) std::cout << __FILE__  << ":" << __LINE__ << ": " << x << std::endl;

class Controller {
protected:
  constexpr static int Tnx  = TNX;
  constexpr static int Tnu  = TNU;
  constexpr static int Tndu = TNDU;
  constexpr static int Tny  = TNY;
  xVec state;   // Internal state
  uVec control; // Control input
  wVec exogeneous_input; // Control input    

  nlohmann::json latest_metadata;

public:
    // Constructor that initializes state dimensions and calls setup
    Controller(const nlohmann::json &json_data) {
        initializeDimensions(json_data);
    }

    // Default constructor
    Controller()  {
      state            = xVec::Zero();  
      control          = uVec::Zero(); 
      exogeneous_input = wVec::Zero();
    }

    using CreatorFunc = std::function<Controller*(const nlohmann::json&)>;
    static void registerController(const std::string& name, CreatorFunc creator);
    static Controller* createController(const std::string& name, const nlohmann::json &json_data);

    uVec getLastestControl() {
      return control;
    }

    nlohmann::json getLastestMetadata() {
      return latest_metadata;
    }

    virtual ~Controller() = default;

    // Controller-specific functions
    virtual void calculateControl(int k, double t, const xVec &x, const wVec &w) = 0;

protected:
    // Controller-specific functions
    virtual void setup(const nlohmann::json &json_data) = 0;

    // Method to initialize state and control based on dimensions from JSON
    void initializeDimensions(const nlohmann::json &json_data) {

      int state_dim = json_data.at("system_parameters").at("state_dimension");
      int control_dim = json_data.at("system_parameters").at("input_dimension");

      if (state_dim != Tnx) {
          throw std::invalid_argument("State dimension from JSON does not match dimension from compilation parameter Tnx.");
      }
      if (control_dim != Tnu) {
          throw std::invalid_argument("Control dimension from JSON does not match dimension from compilation parameter Tnx.");
      }
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
    