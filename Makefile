# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -MMD -MP
INCLUDES := -I/usr/include/eigen3 -I./include -I.
LDFLAGS :=

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -O0
else
	CXXFLAGS += -O3
endif

# Optional compilation of Pinocchio
WITH_PINOCCHIO ?= 0
ifeq ($(WITH_PINOCCHIO),1)
	INCLUDES += -I/opt/openrobots/include
	LDFLAGS += -L/opt/openrobots/lib -lpinocchio_parsers
endif

# Directories
SRC_DIR := src
EXAMPLES_DIR := examples/control
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
EX_OBJ_DIR := $(BUILD_DIR)/examples

# Library and example sources
LIB_SRC := $(shell find $(SRC_DIR) -name '*.cpp')
LIB_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))

EXAMPLE_SRC := $(wildcard $(EXAMPLES_DIR)/*.cpp)
EXAMPLE_OBJ := $(patsubst $(EXAMPLES_DIR)/%.cpp,$(EX_OBJ_DIR)/%.o,$(EXAMPLE_SRC))

# Example binaries
EXAMPLES := $(notdir $(EXAMPLE_SRC:.cpp=))
EXE_FILES := $(EXAMPLES:%=$(BUILD_DIR)/%)

# Build configuration tracking (to avoid full rebuilds)
CONFIG_FILE := $(BUILD_DIR)/.config

# Default target
all: $(CONFIG_FILE) $(EXE_FILES)

# Detect changes in WITH_PINOCCHIO between builds
$(CONFIG_FILE):
	@mkdir -p $(BUILD_DIR)
	@echo "WITH_PINOCCHIO=$(WITH_PINOCCHIO)" > $(CONFIG_FILE)

# Rebuild only if the config changes
$(LIB_OBJ) $(EXAMPLE_OBJ): $(CONFIG_FILE)

# Rule for example executables
$(BUILD_DIR)/%: $(EX_OBJ_DIR)/%.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)
	@echo "Built $@"

# Rule for library objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Rule for example objects
$(EX_OBJ_DIR)/%.o: $(EXAMPLES_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Include dependency files
-include $(LIB_OBJ:.o=.d)
-include $(EXAMPLE_OBJ:.o=.d)

# Clean
clean:
	rm -rf $(BUILD_DIR)
