# Compiler
CXX = g++
CXXFLAGS = -O3
INCLUDES = -I/usr/include/eigen3 -I./src/

build/main: main.cpp
	$(CXX) $(INCLUDES) main.cpp -o build/main $(CXXFLAGS)

example1: examples/functions/function1.cpp
	$(CXX) $(INCLUDES) examples/functions/function1.cpp -o build/function1 $(CXXFLAGS)

control1: examples/control/point_mass.cpp
	$(CXX) $(INCLUDES) examples/control/point_mass.cpp -o build/point_mass $(CXXFLAGS)

