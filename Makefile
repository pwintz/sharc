3x3_proportional_controller.o: 3x3_proportional_controller.c
	gcc -c 3x3_proportional_controller.c -o 3x3_proportional_controller.o

3x3_proportional_controller: 3x3_proportional_controller.o
	gcc 3x3_proportional_controller.o -o 3x3_proportional_controller

# Define a default target
.DEFAULT_GOAL := default
.PHONY: default

default: 3x3_proportional_controller
	./3x3_proportional_controller 1.43 -0.234

# Clean rule to remove object and executable files
clean:
	rm -f 3x3_proportional_controller.o 3x3_proportional_controller