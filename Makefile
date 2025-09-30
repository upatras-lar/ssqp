# Compiler
CXX := g++
CXXFLAGS := -std=c++17
INCLUDES := -I/usr/include/eigen3 -I./src/ -I.

DEBUG ?= 0

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -O0
else
	CXXFLAGS += -O3
endif

build/main: main.cpp
	$(CXX) $(INCLUDES) main.cpp -o build/main $(CXXFLAGS)

example1: examples/functions/function1.cpp
	$(CXX) $(INCLUDES) examples/functions/function1.cpp -o build/function1 $(CXXFLAGS)

control1: examples/control/point_mass.cpp
	$(CXX) $(INCLUDES) examples/control/point_mass.cpp -o build/point_mass $(CXXFLAGS)

