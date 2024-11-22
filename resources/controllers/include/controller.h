// controller/Controller.h
#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <unordered_map>

// These MUST be overwritten
#ifndef TNX
  #define TNX -1
#endif
#ifndef TNU
  #define TNU -1
#endif
#ifndef TNDU
  #define TNDU -1
#endif
#ifndef TNY
  #define TNY -1
#endif

using xVec = Eigen::Matrix<double, TNX,  1>;
using uVec = Eigen::Matrix<double, TNU,  1>;
using wVec = Eigen::Matrix<double, TNDU, 1>;
using yVec = Eigen::Matrix<double, TNY,  1>;

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
  wVec exogenous_input; // Control input    

  nlohmann::json latest_metadata;

public:
    // Constructor that initializes state dimensions and calls setup
    Controller(const nlohmann::json &json_data) {
      initializeDimensions(json_data);
      state           = xVec::Zero();  
      control         = uVec::Zero(); 
      exogenous_input = wVec::Zero();
      assert(Tnx > 0);
      assert(Tnu > 0);
      assert(Tndu > 0);
      assert(Tny > 0);
      initializeDimensions(json_data);
    }

    // Default constructor
    Controller()  {
      state           = xVec::Zero();  
      control         = uVec::Zero(); 
      exogenous_input = wVec::Zero();
    }

    using CreatorFunc = std::function<Controller*(const nlohmann::json&)>;
    static void registerController(const std::string& name, CreatorFunc creator);
    static Controller* createController(const std::string& name, const nlohmann::json &json_data);

    virtual ~Controller() = default;
   
    // Accessors
    uVec getLatestControl() {return control;}
    nlohmann::json getLatestMetadata() {return latest_metadata;}

    // Controller-specific functions
    virtual void calculateControl(int k, double t, const xVec &x, const wVec &w) = 0;

protected:
    // Controller-specific functions
    virtual void setup(const nlohmann::json &json_data) = 0;

    // Method to initialize state and control based on dimensions from JSON
    void initializeDimensions(const nlohmann::json &json_data);
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
    