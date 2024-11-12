#pragma once

#include <mpc/Utils.hpp>
#include <Eigen/Core>
#include "nlohmann/json.hpp"

using namespace mpc;

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_WITH_FILE_LOCATION(x) std::cout << __FILE__  << ":" << __LINE__ << ": " << x << std::endl;

// Vector/Matrix Printing Format
const Eigen::IOFormat fmt(3,    // precision
                          0,    // flags
                          ", ", // coefficient separator
                          "\n", // row prefix
                          "[",  // matrix prefix
                          "]"); // matrix suffix.

template<int n>
void printVector(std::string description, cvec<n>& vec_to_print){
  std::cout << description << ":" << std::endl << vec_to_print.transpose().format(fmt) << std::endl;
}

template<int rows, int cols>
void printMat(std::string description, mat<rows, cols>& mat_to_print){
  std::cout << description << ": " << std::endl << mat_to_print.format(fmt) << std::endl;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    // From https://stackoverflow.com/a/26221725/6651650
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ 
      throw std::runtime_error( "string_format(): Error during formatting." ); 
    }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

// template<int rows, int cols>
// void loadMatrixValuesFromJson(mpc::mat<rows, cols>& mat_out, const nlohmann::json& json_data, const std::string& key) {
//     // assertHasKeys(json_data, key);
//     if (json_data.at(key).size() != rows*cols)
//     {
//       throw std::invalid_argument(string_format(
//             "loadColumnValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in mat_out (%dx%d).", 
//             json_data.at(key).size(),  key.c_str(), rows, cols));
//     }
// 
//     for (int row = 0; row < rows; row++) {
//       for (int col = 0; col < cols; col++) {
//         mat_out(row, col) = json_data.at(key).at(row).at(col);
//       }
//     }
// }

template<int rows>
void loadColumnValuesFromJson(mpc::cvec<rows>& vector_out, const nlohmann::json& json_data, const std::string& key) {
    if (json_data.at(key).size() != rows)
    {
      throw std::invalid_argument(string_format(
            "loadColumnValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in vector_out (%dx1).", 
            json_data.at(key).size(),  key.c_str(), rows));
    }

    int i = 0;
    for (auto& element : json_data.at(key)) {
      vector_out[i] = element;
      i++;
    }
}

template<int rows>
void assertVectorLessThan(const std::string&  left_name, const mpc::cvec<rows> left, 
                          const std::string& right_name, const mpc::cvec<rows> right) {
    for(int row = 0; row < rows; row++) {
      PRINT(       left_name << "[" << row << "]: " <<  left[row] << " must be less than " 
              <<  right_name << "[" << row << "]: " << right[row])
      assert(left[row] <= right[row] - 0.1);
    }
}

template<int rows>
void assertVectorAlmostLessThan(const std::string&  left_name, const mpc::cvec<rows> left, 
                          const std::string& right_name, const mpc::cvec<rows> right, 
                          double abs_tol) {
    for(int row = 0; row < rows; row++) {
      PRINT(       left_name << "[" << row << "]: " <<  left[row] << " must be less than " 
              <<  right_name << "[" << row << "]: " << right[row] << " + " << abs_tol << " = " << right[row] + abs_tol)
      assert(left[row] <= right[row] + abs_tol);
    }
}

// template<int rows, int cols>
// void loadColumnValuesFromJson(mat<rows, cols>& mat_out, nlohmann::json json_data, std::string key) {
//     
//       // Check debug level and print the appropriate message
//       if (debug_interfile_communication_level >= 2) {
//           PRINT_WITH_FILE_LOCATION("Building a matrix from key='" << key << "'...");
//           PRINT("Value: " << json_data.at(key))
//       }
// 
//       if (json_data.at(key).size() != rows*cols)
//       {
//         throw std::invalid_argument(string_format(
//               "loadColumnValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in mat_out (%dx%d).", 
//               json_data.at(key).size(),  key.c_str(), rows, cols));
//       }
// 
//       int i = 0;
//       for (auto& element : json_data.at(key)) {
//         mat_out[i] = element;
//         i++;
//       }
//       
//       // Check debug level and print the appropriate message
//       if (debug_interfile_communication_level >= 2) {
//           PRINT_WITH_FILE_LOCATION("Built a matrix from key='" << key << "': " << mat_out);
//       }
// }


inline void assertFileExists(const std::string& name) {
  // Check that a file exists.
  if (std::filesystem::exists(name)){
    return;
  } else {
    throw std::runtime_error("The file \"" + name + "\" does not exist. The current working directory is \"" + std::filesystem::current_path().string() + "\"."); 
  }
}

