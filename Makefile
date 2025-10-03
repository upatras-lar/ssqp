# Compiler
CXX := g++
CXXFLAGS := -std=c++17
INCLUDES := -I/usr/include/eigen3 -I./include/ -I.

DEBUG ?= 0

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -O0
else
	CXXFLAGS += -O3
endif

point_mass: build/point_mass.o build/types.o build/lqr.o build/admm_qp.o build/csqp.o build/sqp_step_types.o build/filter.o
	$(CXX) $(INCLUDES) -o build/point_mass build/point_mass.o build/types.o build/lqr.o build/admm_qp.o build/csqp.o build/sqp_step_types.o build/filter.o $(CXXFLAGS)

function1: build/function1.o build/types.o build/lqr.o build/admm_qp.o build/csqp.o build/sqp_step_types.o build/filter.o
	$(CXX) $(INCLUDES) -o build/function1 build/function1.o build/types.o build/lqr.o build/admm_qp.o build/csqp.o build/sqp_step_types.o build/filter.o $(CXXFLAGS)

clean:
	rm -rf build/*

build/point_mass.o: examples/control/point_mass.cpp
	$(CXX) $(INCLUDES) -c examples/control/point_mass.cpp -o build/point_mass.o $(CXXFLAGS)

build/function1.o: examples/functions/function1.cpp
	$(CXX) $(INCLUDES) -c examples/functions/function1.cpp -o build/function1.o $(CXXFLAGS)

build/types.o: src/types.cpp
	$(CXX) $(INCLUDES) -c src/types.cpp -o build/types.o $(CXXFLAGS)

build/lqr.o: src/solvers/lqr.cpp
	$(CXX) $(INCLUDES) -c src/solvers/lqr.cpp -o build/lqr.o $(CXXFLAGS)

build/admm_qp.o: src/solvers/admm_qp.cpp
	$(CXX) $(INCLUDES) -c src/solvers/admm_qp.cpp -o build/admm_qp.o $(CXXFLAGS)

build/csqp.o: src/solvers/csqp.cpp
	$(CXX) $(INCLUDES) -c src/solvers/csqp.cpp -o build/csqp.o $(CXXFLAGS)

build/sqp_step_types.o: src/sqp_step/types.cpp
	$(CXX) $(INCLUDES) -c src/sqp_step/types.cpp -o build/sqp_step_types.o $(CXXFLAGS)

build/filter.o: src/sqp_step/filter.cpp
	$(CXX) $(INCLUDES) -c src/sqp_step/filter.cpp -o build/filter.o $(CXXFLAGS)

build/gss.o: src/sqp_step/gss.cpp
	$(CXX) $(INCLUDES) -c src/sqp_tep/gss.cpp -o build/gss.o $(CXXFLAGS)

