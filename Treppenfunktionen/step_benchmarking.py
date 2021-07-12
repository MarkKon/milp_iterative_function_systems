import pyomo.environ as pyo
import pyomo
from Treppenfunktionen.step_models import Indicator_Weak, Indicator_Strong, Indicator_Strong_SOS1, Incremental, Indicator_Weak_Strong, Indicator_Layered, Binary_Model, Binary_Model_2, Indicator_Weak_Concrete
import numpy as np
import pandas
import random
import string


class B_Solver(object):
    def __init__(self, solver_name):
        self.name = solver_name
        self.solver = pyomo.opt.SolverFactory(solver_name)
            
class Benchmark(object):
    def __init__(self, n_intervals, n_its, rng, time_limit):
        self.n_intervals = n_intervals
        self.n_its = n_its
        self.time_limit = time_limit
        self.step_function_models = []
        self.b_solvers = []
        self.output_df = pandas.DataFrame(columns = [
            "benchmark_iteration",
            "model",
            "solver",
            "lower_bound",
            "upper_bound",
            "time",
            #"wall_time",
            "status",
            #"return_code",
            "error_rc"
            ])
        #Festsetzten des RNGs pro benchmark
        self.rng = rng
    
    def add_model(self,model):
        self.step_function_models.append(model)
    
    def add_solver(self, b_solver):
        #Add Time Limit
        if b_solver.name == 'cplex':
            b_solver.solver.options['timelimit'] = self.time_limit
        elif b_solver.name == 'glpk':         
            b_solver.solver.options['tmlim'] = self.time_limit
        elif b_solver.name == 'gurobi':           
            b_solver.solver.options['TimeLimit'] = self.time_limit
        self.b_solvers.append(b_solver)
    
    def set_benchmark_parameters(self, n_intervals, n_its, rng, time_limit):
        self.n_intervals = n_intervals
        self.n_its = n_its
        self.time_limit = time_limit
        self.rng = rng

    def reset_output(self):
        self.output_df = pandas.DataFrame(columns = [
            "benchmark_iteration",
            "model",
            "solver",
            "lower_bound",
            "upper_bound",
            "time",
            #"wall_time",
            "status",
            #"return_code",
            "error_rc"
            ])

    def run(self, benchmark_its):
        for i in range(benchmark_its):
            a, xi = self.gen_data()
            for model in self.step_function_models:
                for b_solver in self.b_solvers:
                    instance = model.instantiate(self.n_its, self.n_intervals, a, xi)
                    try:
                        results = self.solve(b_solver, instance, model)
                        output_dict = {
                            "benchmark_iteration": i,
                            "model": model.name,
                            "solver": b_solver.name,
                            "lower_bound": results.problem.lower_bound,
                            "upper_bound": results.problem.upper_bound,
                            "time": results.solver.time,
                            #"wall_time": results.solver.wall_time,
                            "status": results.solver.status,
                            #"return_code": results.solver.return_code,
                            "error_rc": results.solver.error_rc,
                        }
                    except Exception as e:
                        print(e)
                        output_dict = {
                            "benchmark_iteration": i,
                            "model": model.name,
                            "solver": b_solver.name,
                            "lower_bound": 0,
                            "upper_bound": np.inf,
                            "time": self.time_limit,
                            #"wall_time": results.solver.wall_time,
                            "status": "aborted",
                            #"return_code": results.solver.return_code,
                            "error_rc": -1,
                        }
                    self.output_df = self.output_df.append(output_dict, ignore_index = True)
    
    def solve(self,b_solver, instance, model):
        return b_solver.solver.solve(instance)

    def gen_data(self):
        a = np.around(self.rng.random((self.n_its,self.n_intervals)),4)
        xi = np.array([np.linspace(0,1,self.n_intervals+1).tolist() for i in range(self.n_its)])
        return (a,xi)

    def save_output(self, path):
        full_path = path + "it" + str(self.n_its) + "_ints" + str(self.n_intervals) + ".csv"
        self.output_df.to_csv(full_path)
        print("Saved to " + full_path)

    def n_vars(self):
        _, xi = self.gen_data()
        a = np.full((self.n_its, self.n_intervals), 0)
        for model in self.step_function_models:
            s = self.b_solvers[0]
            inst  = model.instantiate(self.n_its, self.n_intervals, a, xi)
            result = s.solver.solve(inst)
            print(model.name, "Constraints:", result.problem.number_of_constraints, "Variables:", result.problem.number_of_variables, "Binary:", result.problem.number_of_binary_variables)

class MonotoneBenchmark(Benchmark):
    def gen_data(self):
        xi = np.array([np.linspace(0,1,self.n_intervals+1).tolist() for i in range(self.n_its)])
        delta = np.around(self.rng.random((self.n_its,self.n_intervals)),4)
        cum = np.cumsum(delta, axis = 1)
        a = 1- np.array([row/row[-1] for row in cum])
        return (a, xi)
    
    def save_output(self, path):
        super().save_output(path + "Mon_")

class ConcaveBenchmark(Benchmark):
    def gen_data(self):
        xi = np.array([np.linspace(0,1,self.n_intervals+1).tolist() for i in range(self.n_its)])
        delta = np.around(self.rng.random((self.n_its,self.n_intervals)),4)
        delta_sorted = np.sort(delta, axis = 1)
        cum = np.cumsum(delta_sorted, axis = 1)
        a = 1-np.array([row/row[-1] for row in cum])
        return (a,xi)

    def save_output(self, path):
        super().save_output(path + "Con_")


if __name__ == "__main__": 
    settings = [
        {
        "n_its": 5,
        "n_intervals": 8,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },{
        "n_its": 10,
        "n_intervals": 64,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },{
        "n_its": 20,
        "n_intervals": 128,
        "rng": np.random.default_rng(2021),
        "time_limit": 10
        },{
        "n_its": 25,
        "n_intervals": 512,
        "rng": np.random.default_rng(2021),
        "time_limit": 10}
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
        #B, 
        B_m,
        #B_c
        ]

    for setting in settings:
        for Benchmark in Benchmarks:
            Benchmark.set_benchmark_parameters(**setting)
            Benchmark.run(5)
            Benchmark.save_output("data\\")
            Benchmark.reset_output()
