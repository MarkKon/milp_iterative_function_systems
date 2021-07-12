import pyomo.environ as pyo
import pyomo
from Treppenfunktionen.step_models import *
from Treppenfunktionen.step_benchmarking import *
import numpy as np
import pandas


settings = [
        {
        "n_its": 10,
        "n_intervals": 64,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },{
        "n_its": 20,
        "n_intervals": 128,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        }
    ]

#Nonmonotone Benchmark
B = Benchmark(**settings[0])
#Monotone Benchmark
B_m = MonotoneBenchmark(**settings[0])
#Concave Benchmark
B_c = ConcaveBenchmark(**settings[0])
#add solvers
gurobi = B_Solver("gurobi")
cplex = B_Solver("cplex")
#add models

models = [
    Indicator_Weak(),
    Indicator_Strong(),
    Indicator_Strong_SOS1(),
    Indicator_Weak_Strong(),
    Indicator_Layered(),
    Incremental(),
    Binary_Model(),
    Binary_Model_2()
    ]

B.add_solver(gurobi)
B.add_solver(cplex)
for model in models:
    B.add_model(model)

B_m.add_solver(gurobi)
B_m.add_solver(cplex)

m_models = [
    Indicator_Weak_Strong(),
    Indicator_Layered(),
    Incremental(),
    Binary_Model(),
    Binary_Model_2()]

for model in m_models:
    B_m.add_model(model)


B_c.add_solver(gurobi)
B_c.add_solver(cplex)

for model in m_models: 
    B_c.add_model(model)

Benchmarks = [
    B, 
    B_m,
    B_c
    ]

for setting in settings:
    for Benchmark in Benchmarks:
        print("Starting Benchmark")
        Benchmark.set_benchmark_parameters(**setting)
        Benchmark.run(5)
        Benchmark.save_output("data\\step_to_opt\\")
        Benchmark.reset_output()
