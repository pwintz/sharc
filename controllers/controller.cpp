// controller/controller.cpp
#include "controller.h"

// Static registry to hold the controller types
std::unordered_map<std::string, Controller::CreatorFunc>& Controller::getRegistry() {
    static std::unordered_map<std::string, CreatorFunc> registry;
    return registry;
}

// Register a new controller type with a creator function
void Controller::registerController(const std::string& name, CreatorFunc creator) {
    getRegistry()[name] = creator;
}

// Create a controller of the specified type using the creator function
Controller* Controller::createController(const std::string& name, const nlohmann::json &json_data) {
    auto& registry = getRegistry();
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second(json_data);  // Call the creator function with json_data
    }
    return nullptr;  // Return nullptr if the controller type is not found
}