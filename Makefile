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

point_mass: build/point_mass.o build/types.o build/Model.o build/StageConstraints.o build/lqr.o build/admm_qp.o build/ssqp.o build/sqp_step_types.o build/filter.o
	$(CXX) $(INCLUDES) -o build/point_mass build/point_mass.o build/types.o build/Model.o build/StageConstraints.o build/lqr.o build/admm_qp.o build/ssqp.o build/sqp_step_types.o build/filter.o $(CXXFLAGS)

pendulum: build/pendulum.o build/types.o build/Model.o build/StageConstraints.o build/lqr.o build/admm_qp.o build/ssqp.o build/sqp_step_types.o build/filter.o
	$(CXX) $(INCLUDES) -o build/pendulum build/pendulum.o build/types.o build/Model.o build/StageConstraints.o build/lqr.o build/admm_qp.o build/ssqp.o build/sqp_step_types.o build/filter.o $(CXXFLAGS)

clean:
	rm -rf build/*

build/point_mass.o: examples/control/point_mass.cpp
	$(CXX) $(INCLUDES) -c examples/control/point_mass.cpp -o build/point_mass.o $(CXXFLAGS)

build/pendulum.o: examples/control/pendulum.cpp
	$(CXX) $(INCLUDES) -c examples/control/pendulum.cpp -o build/pendulum.o $(CXXFLAGS)

build/function1.o: examples/functions/function1.cpp
	$(CXX) $(INCLUDES) -c examples/functions/function1.cpp -o build/function1.o $(CXXFLAGS)

build/types.o: src/types.cpp
	$(CXX) $(INCLUDES) -c src/types.cpp -o build/types.o $(CXXFLAGS)

build/lqr.o: src/solvers/lqr.cpp
	$(CXX) $(INCLUDES) -c src/solvers/lqr.cpp -o build/lqr.o $(CXXFLAGS)

build/admm_qp.o: src/solvers/admm_qp.cpp
	$(CXX) $(INCLUDES) -c src/solvers/admm_qp.cpp -o build/admm_qp.o $(CXXFLAGS)

build/ssqp.o: src/solvers/ssqp.cpp
	$(CXX) $(INCLUDES) -c src/solvers/ssqp.cpp -o build/ssqp.o $(CXXFLAGS)

build/sqp_step_types.o: src/sqp_step/types.cpp
	$(CXX) $(INCLUDES) -c src/sqp_step/types.cpp -o build/sqp_step_types.o $(CXXFLAGS)

build/filter.o: src/sqp_step/filter.cpp
	$(CXX) $(INCLUDES) -c src/sqp_step/filter.cpp -o build/filter.o $(CXXFLAGS)

build/gss.o: src/sqp_step/gss.cpp
	$(CXX) $(INCLUDES) -c src/sqp_step/gss.cpp -o build/gss.o $(CXXFLAGS)

build/Model.o: src/model/Model.cpp
	$(CXX) $(INCLUDES) -c src/model/Model.cpp -o build/Model.o $(CXXFLAGS)

build/StageConstraints.o: src/model/stagewise/StageConstraints.cpp
	$(CXX) $(INCLUDES) -c src/model/stagewise/StageConstraints.cpp -o build/StageConstraints.o $(CXXFLAGS)
