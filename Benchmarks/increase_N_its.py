import pyomo.environ as pyo
import pyomo
from Stkw_Linear.plin_models import *
from Stkw_Linear.plin_benchmarking import *
import numpy as np
import pandas


settings = [
        {
        "n_its": 4,
        "n_intervals": 32,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },{
        "n_its": 8,
        "n_intervals": 32,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },
        {
        "n_its": 16,
        "n_intervals": 32,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },
        {
        "n_its": 32,
        "n_intervals": 32,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        }
    ]

#Concave Benchmark
B_c = ConcaveBenchmark(**settings[0])
#add solvers
gurobi = B_Solver("gurobi")
cplex = B_Solver("cplex")
#add models


m_models = [
    Indicator_Strong_Concrete(),
    Incremental_Concrete(),
    DLog_Concrete(),
    ]


B_c.add_solver(gurobi)
B_c.add_solver(cplex)

for model in m_models: 
    B_c.add_model(model)

Benchmarks = [
    B_c
    ]

for setting in settings:
    for Benchmark in Benchmarks:
        print("Starting Benchmark")
        Benchmark.set_benchmark_parameters(**setting)
        Benchmark.run(5)
        Benchmark.save_output("data\\increase_N_its\\")
        Benchmark.reset_output()
