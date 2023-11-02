#include <stdio.h>
#include <stdlib.h>

// Constants
// #define n 3
// const double matA[n][n] = {
//     { 0.0,   1.0,  2.0},
//     { 3.0,  -1.2,  3.4},
//     { 2.3,   1.0, -3.1}
// };
#define n 2
#define m 1
const double matA[m][n] = {
    -1, -2
};

// Variables
double x[n] = {0.0, 0.0}; // State (n-dimensional)
double u[m] = {0.0};  // Control (n-dimensional)

int main(int argc, char *argv[]) {
    // Check the arguments.
    if (argc != n + 1) {
      fprintf(stderr, "Wrong number of argmunents given. Expected %d. Recieved %d.\n", n, argc-1);
      switch (n) {
        case 1:
          fprintf(stderr, "Usage: %s <current_state1>\n", argv[0]);
          break;
        case 2:
          fprintf(stderr, "Usage: %s <current_state1> <current_state2>\n", argv[0]);
          break;
        case 3:
          fprintf(stderr, "Usage: %s <current_state1> <current_state2> <current_state4>\n", argv[0]);
            // statements_2;
            break;
        default:
          fprintf(stderr, "Usage: %s <current_state1> <current_state2> ... <current_state%d>\n", argv[0], n);
          // default_statements;
          break;
      }
        return 1;
    }

    // Get the current states from the command-line arguments and print it.
    // printf("x:\t");
    for (int i = 0; i < n; i++) {
        x[i] = atof(argv[i + 1]);
        // printf("% 9f\t", x[i]);
    }
    // printf("\n");
   
    // Loop through the rows of A and x to compute u.
    for (int i = 0; i < m; i++) {
        // Loop through the columns of A.
        for (int j = 0; j < n; j++) {
            // Print each element with formatting
            u[i] += matA[i][j]*x[j];
            // printf("matA[%d][%d]: %f", i, j, matA[i][j]);
            // printf("x[j]: %f", x[j]);
        }
        // Print a newline to move to the next row
        // printf("\n");
    }

    // Loop through the rows of u to print them.
    // printf("u:\t");
    for (int i = 0; i < m; i++) {
        // Print each element with formatting
        printf("%f\n", u[i]);
    }
    // printf("\n");

    return 0;
}
