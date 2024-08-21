
# Set prediction_horizon and control_horizon 
# in case it is not set as a command line argument.
prediction_horizon = 7
control_horizon = 4

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
INCDIR      := include
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cpp
OBJEXT      := o

CXX = g++-11
CC = gcc-11
# C++ flags
CXXFLAGS = -std=c++20
# C Preprocessor flags.
CPPFLAGS = \
	-I/usr/local/include \
	-I/usr/include/eigen3 \
	-I/usr/local/include/osqp \
	-I${INCDIR}\
	-I${LIBMPC_INCLUDE} \
	-I${SCARAB_ROOT}/utils \
	-Wno-deprecated-enum-enum-conversion \
	-DEIGEN_STACK_ALLOCATION_LIMIT=0 \
	-D PREDICTION_HORIZON=${prediction_horizon} \
	-D CONTROL_HORIZON=${control_horizon} 
# -g <- Enable debugging.

CFLAGS = -I/usr/local/include/osqp 

# Linker flags
LDFLAGS = # -L/usr/local/mysql-connector-c++/lib64

# The command "-lm" is from the NLopt documentation.
LDLIBS = -lnlopt -lm -losqp

#The Target Binary Program
TARGET := acc_controller_${prediction_horizon}_${control_horizon}

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
SOURCES     := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

# Defauilt Make
all: $(TARGETDIR)/$(TARGET)

print_executable:
	@echo $(TARGET)

# Remake
remake: cleaner all

# Make the Directories
directories: $(TARGETDIR) $(BUILDDIR)

$(TARGETDIR):
	mkdir -p $(TARGETDIR)
	
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Clean only Objects
clean:
	@$(RM) -rf $(BUILDDIR)

#Full Clean, Objects, and Binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

clean_output:
	rm -rf out/*

# Link
$(TARGETDIR)/$(TARGET): $(OBJECTS) $(TARGETDIR)
	$(CXX) -o $(TARGETDIR)/$(TARGET) $(OBJECTS) $(LDLIBS)

# Compile the object file using built-in implicit Make rules: 
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
$(OBJECTS): $(SOURCES) $(shell find ${LIBMPC_INCLUDE} -type f) $(BUILDDIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SOURCES) -o $(OBJECTS)
	
#Non-File Targets
.PHONY: all remake clean cleaner resources