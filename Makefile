# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -MMD -MP
INCLUDES := -I/usr/include/eigen3 -I./include -I.

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -O0
else
	CXXFLAGS += -O3
endif

# Directories
SRC_DIR := src
EXAMPLES_DIR := examples/control
BUILD_DIR := build

# Find all source files
SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp')
EXAMPLE_FILES := $(wildcard $(EXAMPLES_DIR)/*.cpp)

# Object files
OBJ_FILES := $(SRC_FILES:%.cpp=$(BUILD_DIR)/%.o)
EXAMPLE_OBJ := $(EXAMPLE_FILES:%.cpp=$(BUILD_DIR)/%.o)

# Create a list of example targets (one binary per example .cpp)
EXAMPLES := $(notdir $(EXAMPLE_FILES:.cpp=))
EXE_FILES := $(EXAMPLES:%=$(BUILD_DIR)/%)

# Default target: build all examples
all: $(EXE_FILES)

# Rule for example executables
$(BUILD_DIR)/%: $(BUILD_DIR)/examples/control/%.o $(OBJ_FILES)
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $^ -o $@ $(CXXFLAGS)
	@echo "Built $@"

# Generic rule for building object files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Include dependency files for incremental rebuilds
-include $(OBJ_FILES:.o=.d)
-include $(EXAMPLE_OBJ:.o=.d)

# Clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
