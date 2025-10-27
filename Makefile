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

INFO ?= 1
ifeq ($(INFO), 1)
	CXXFLAGS += -DINFO
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

# Library sources
LIB_SRC := $(shell find $(SRC_DIR) -name '*.cpp')
LIB_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))

# Recursively find all example source files
EXAMPLE_SRC := $(shell find $(EXAMPLES_DIR) -name '*.cpp')

# Convert to corresponding object files and executables
EXAMPLE_OBJ := $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/examples/%.o,$(EXAMPLE_SRC))
EXE_FILES := $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/bin/%,$(EXAMPLE_SRC))

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
$(BUILD_DIR)/bin/%: $(BUILD_DIR)/examples/%.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)
	@echo "Built $@"

# Rule for library objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Rule for example objects
$(BUILD_DIR)/examples/%.o: $(EXAMPLES_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Include dependency files
-include $(LIB_OBJ:.o=.d)
-include $(EXAMPLE_OBJ:.o=.d)

# Clean
clean:
	rm -rf $(BUILD_DIR)

clean-examples:
	rm -rf $(BUILD_DIR)/examples $(BUILD_DIR)/bin
