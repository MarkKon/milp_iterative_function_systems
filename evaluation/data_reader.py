import pandas as pd
import numpy as np

class FileSearcher(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path, index_col=0)
        

    def get_model_strings(self):
        model_strings = list(set(self.df.model))
        return model_strings

    def search_in_file(self, model_string, solver_string):
        solver_df = self.df[self.df.solver == solver_string]
        return solver_df[solver_df["model"] == model_string]["time"].to_numpy()