# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -MMD -MP
INCLUDES := -I/usr/include/eigen3 -I./include -I.
LDFLAGS :=
TEST_INCLUDES := -Itests/include

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

# Compilation of Pinocchio
INCLUDES += -I/opt/openrobots/include
LDFLAGS += -L/opt/openrobots/lib -lpinocchio_parsers

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

TESTS_DIR := tests
TESTS_SRC := $(shell find $(TESTS_DIR)/src -name '*.cpp')
TESTS_OBJ := $(patsubst $(TESTS_DIR)/src/%.cpp,$(BUILD_DIR)/tests/obj/%.o,$(TESTS_SRC))

TEST_RUNNER := $(BUILD_DIR)/tests/run_tests
TEST_RUNNER_SRC := $(TESTS_DIR)/run_tests.cpp
TEST_RUNNER_OBJ := $(BUILD_DIR)/tests/run_tests.o

# Default target
all: $(CONFIG_FILE) $(EXE_FILES)

tests: $(TEST_RUNNER)
	@echo "Built tests"

$(TEST_RUNNER): $(TEST_RUNNER_OBJ) $(TESTS_OBJ) $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/tests/obj/%.o: $(TESTS_DIR)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $(TEST_INCLUDES) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/tests/run_tests.o: $(TEST_RUNNER_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $(TEST_INCLUDES) -c $< -o $@ $(CXXFLAGS)

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
-include $(TESTS_OBJ:.o=.d)
-include $(TEST_RUNNER_OBJ:.o=.d)

# Clean
clean:
	rm -rf $(BUILD_DIR)

clean-nopin:
	find $(BUILD_DIR) -type f \! '(' -name PinocchioWrapper.d ')' \! '(' -name PinocchioWrapper.o ')' -delete

clean-examples:
	rm -rf $(BUILD_DIR)/examples $(BUILD_DIR)/bin

clean-tests:
	rm -rf $(BUILD_DIR)/tests
