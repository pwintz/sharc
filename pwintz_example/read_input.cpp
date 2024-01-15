#include "scarab_markers.h"

int main () {
        int power = 4;
        int base = 2;
        int result = 1;

        scarab_begin(); // this is needed for '--pintool_args -fast_forward_to_start_inst 1'
        for (int i = 0; i < power; i++) {
                scarab_roi_dump_begin();
                result = result * base;
                scarab_roi_dump_end();
        }

        printf("result: %d\n", result);
        return 0;
}


// #include <iostream>
// #include <string>
// #include "scarab_markers.h"

// #include <fstream>

// int main() {
//     // std::ofstream fout;
//     // std::ifstream fin;

//     // fout.open("c_to_python_pipe"); // This blocks until something is read from c_to_python_pipe.
//     // std::cout << "Opened 'c_to_python_pipe'." << std::endl;

//     // fout << "Here is text from C++!";
//     // // fout.flush();

//     // std::cout << "Sent text 'Here is text from C++!'." << std::endl;
//     // fout.close();
//     // std::cout << "Closed 'c_to_python_pipe'." << std::endl;

//     // fin.open("python_to_c_pipe");
//     // std::cout << "Opened 'python_to_c_pipe'." << std::endl;

//     int value = 0;
//     for(int i = 1; i < 10; i++)
//     {
//     // scarab_roi_dump_begin();
//         value += i;
//     // scarab_roi_dump_end();
//     }
// //     fout << "Here is your entered text:" << std::endl;
// //     fout << line << std::endl;

//     // std::string line; 
//     // std::getline(fin, line);
//     // std::cout << "C++ recived this text from Python: " << line << std::endl;
//     // fin.close();

//     // if ( ! fout )
//     // {
//     //         std::cout << "An error occurred opening the file" << std::endl;
//     // }

    

//     // for(int i = 0; i < 10; i++)
//     // {
//     //     // scarab_roi_dump_begin();
//     //     fout << "Here is your entered text:" << std::endl;
//     //     fout << line << std::endl;
//     //     // scarab_roi_dump_end();
//     // }

//     // fout << "Finished read_input." << std::endl;
//     return 0;
// }