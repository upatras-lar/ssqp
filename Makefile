# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -fPIC -march=native -DEIGEN_RUNTIME_NO_MALLOC
INCLUDES := -I/usr/include/eigen3 -I./include -I.
TEST_INCLUDES := -Itests/include
LDFLAGS :=

PINOCCHIO_STATIC := 0

ifeq ($(PINOCCHIO_STATIC), 1)
	PINOCCHIO_STATIC_DIR := ~/pinocchio-static
	INCLUDES += -I$(PINOCCHIO_STATIC_DIR)/include
	LDFLAGS += $(PINOCCHIO_STATIC_DIR)/lib/libpinocchio_default.a $(PINOCCHIO_STATIC_DIR)/lib/libpinocchio_parsers.a
else
	INCLUDES += -I/opt/openrobots/include
	LDFLAGS += -L/opt/openrobots/lib -lpinocchio_parsers
endif

PYTHON_INCLUDE := -I$(shell python -m nanobind --include) -I/usr/include/python3.12
LIBNANOBIND := $(shell python -m nanobind --include)/../libnanobind.o
PYTHON_EXT := $(shell python3-config --extension-suffix)
PYTHON_DIR := python

INCLUDES += -I./external/tracy/public -I./external/json/single_include

ENABLE_PROFILING ?= 0
ifeq ($(ENABLE_PROFILING), 1)
	CXXFLAGS += -g -O2 -pthread -fno-omit-frame-pointer -DTRACY_ENABLE -DTRACY_MEMORY
	LDFLAGS += -pthread
endif

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG
	ifeq ($(ENABLE_PROFILING), 0)
		CXXFLAGS += -O0
	endif
else ifeq ($(ENABLE_PROFILING), 0)
	CXXFLAGS += -O3
endif

INFO ?= 1
ifeq ($(INFO), 1)
	CXXFLAGS += -DINFO
endif

# Directories
SRC_DIR := src
EXAMPLES_DIR := examples/trajgen/c++
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BINDINGS_OBJ_DIR := $(BUILD_DIR)/bindings

LIB_NAME := ssqp
STATIC_LIB := $(BUILD_DIR)/lib/lib$(LIB_NAME).a

# Library sources
LIB_SRC := $(shell find $(SRC_DIR) -name '*.cpp')
LIB_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))

LIB_SRC += external/tracy/public/TracyClient.cpp
LIB_OBJ += $(BUILD_DIR)/obj/TracyClient.o

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

BINDINGS_SRC := $(shell find $(PYTHON_DIR) -name '*.cpp')
BINDINGS_OBJ := $(patsubst $(PYTHON_DIR)/%.cpp,$(BINDINGS_OBJ_DIR)/%.o,$(BINDINGS_SRC))

# Default target
all: $(EXE_FILES)

tests: $(TEST_RUNNER)
	@echo "Built tests"

python: $(BUILD_DIR)/bindings/ssqpy$(PYTHON_EXT)

$(TEST_RUNNER): $(TEST_RUNNER_OBJ) $(TESTS_OBJ) $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/tests/obj/%.o: $(TESTS_DIR)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $(TEST_INCLUDES) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/tests/run_tests.o: $(TEST_RUNNER_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $(TEST_INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Rule for example executables
$(BUILD_DIR)/bin/%: $(BUILD_DIR)/examples/%.o $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)
	@echo "Built $@"

# Rule for library objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

$(OBJ_DIR)/TracyClient.o: external/tracy/public/TracyClient.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

# Rule for example objects
$(BUILD_DIR)/examples/%.o: $(EXAMPLES_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/bindings/ssqpy$(PYTHON_EXT): $(LIB_OBJ) $(BINDINGS_OBJ)
	$(CXX) -shared $^ $(LIBNANOBIND) $(LDFLAGS) -o $@

# Rule for python bindings
$(BUILD_DIR)/bindings/%.o: $(PYTHON_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(INCLUDES) $(PYTHON_INCLUDE) -c $< -o $@ $(CXXFLAGS)

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

clean-python:
	rm -rf $(BUILD_DIR)/bindings