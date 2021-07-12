import pyomo.environ as pyo
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.piecewise import Piecewise
import numpy as np
from time import perf_counter

'''
Klasse mit Pyomo-Modellen für iterierte Funktionensysteme mit
Treppenfunktionen. Die Modelle sind Subklassen der Klasse
Step_Function_Model, welche eine grundlegende Modellstruktur 
vorgibt: das pyomo abstract model und der Modellname als Attribute
sowie Funktionen welche einzeln die Parameter, Variablen, 
Zielfunktion und Constraints vorgeben. Weiters gibt es eine
Funktion, welche eine Menge an Parametern (Dimensionszahlen sowie
Funktionswerte und Stützpunkte) in ein für das Modell passendes
Dictionary konvertiert.

Die konkreten Modelle sollen nur diese Funktionen ändern, nicht
die __init__ Funktion sowie jene Funktion, die eine Modellinstanz
mit gesetzten Parametern erstellt (instantiate). Damit kann im 
Benchmarking jedes Modell gleichbehandelt werden und es ist immer
klar, welche Modellteile wo dabei sind und wo nicht.

Gewisse Modelle, für welche das Modellbauen bereits die Parameter
benötigt, sind die Funktionen in die instantiate-Funktion verschoben
und es wird ein pyomo-Concrete-Model erstellt. Wiederum sollen die 
darin vorkommenden Funktionen zwischen den Modellen verändert werden,
nicht aber der Bau der instantiate-Funktion.
'''

class NotPowerOfTwo(Exception):
    pass

class _Step_Function_Model(object):
    def __init__(self, warm = False):
        # Modell erstellen
        self.model = pyo.AbstractModel()
        # Modellteile zusammenfügen
        self._declare_parameters()
        self._declare_variables()
        self._declare_objective_function()
        self._declare_constraints()
        # Namen festsetzen
        self._declare_name()
        # Warm oder nicht:
        self.warm = warm

    # Erstellt eine konkrete Instanz nach immer demselben Muster
    def instantiate(self, n_its, n_intervals, f_values, grid_values):
            inst = self.model.create_instance(data = self._make_data_dict(n_its, n_intervals, f_values, grid_values))
            return inst

    # Diese Funktionen sollen von den Subklassen gesetzt werden
    def _declare_parameters(self): raise NotImplementedError
    def _declare_variables(self):  raise NotImplementedError
    def _declare_objective_function(self):  raise NotImplementedError
    def _declare_constraints(self):  raise NotImplementedError
    
    def _declare_name(self):
        self.name = "Generic Model"
    
    def _make_data_dict(self, n_its, n_intervals, f_values, grid_values):
        raise NotImplementedError

class _Concrete_Step_Function_Model(_Step_Function_Model):
    def instantiate(self, n_its, n_intervals, f_values, grid_values):
        self.model = pyo.ConcreteModel()
        self.n_intervals = n_intervals
        self.n_its = n_its
        self.f_values = f_values
        self.grid_values = grid_values
        self._declare_parameters()
        self._declare_variables()
        self._declare_constraints()
        self._declare_objective_function()
        return self.model
    
    def __init__(self, warm=False):
        self.warm = warm
        self._declare_name()
        if self.warm:
            self.name = self.name + " Warm"

class Indicator_Weak_Concrete(_Concrete_Step_Function_Model):
    def _declare_parameters(self):
        # Index als Parameter einspeichern (pyomo doc sagt, das sei best practice)
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)

    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
    
    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        # Bedingung sum(B)=1
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        # Ungleichungskette
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_right_ineq)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_name(self):
        self.name = "Concrete Indicator Weak"

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_step_function(self,m,i):
        return m.y[i] == sum(self.f_values[i-1,j-1]* m.B[i,j] for j in m.I_intervals)

    def _exp_left_ineq(self,m,i,j):
        return self.grid_values[i-1,j-1]*m.B[i,j]+self.grid_values[i-1,0]*(1-m.B[i,j]) <= m.x[i]

    def _exp_right_ineq(self,m,i,j):
        return m.x[i] <= self.grid_values[i-1,j]*m.B[i,j]+self.grid_values[i-1,-1]*(1-m.B[i,j])
    
    def _exp_sos1(self,m,i):
        return sum(m.B[i,j] for j in m.I_intervals) == 1

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]
    
class Indicator_Weak(_Step_Function_Model):
    def _declare_parameters(self):
        # Dimensionsparameter
        self.model.n_intervals = pyo.Param(within = pyo.NonNegativeIntegers)
        self.model.n_its = pyo.Param(within = pyo.NonNegativeIntegers)
        # Index als Parameter einspeichern (pyomo doc sagt, das sei best practice)
        self.model.I_intervals = pyo.RangeSet(1, self.model.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.model.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.model.n_its)
        self.model.I_y = pyo.RangeSet(0,self.model.n_its)
        # Funktionswerte und Stützstellen als Parameter
        self.model.f_values = pyo.Param(self.model.I_its, self.model.I_intervals)
        self.model.grid_values = pyo.Param(self.model.I_its, self.model.I_steps)
    
    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)

    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        # Bedingung sum(B)=1
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        # Ungleichungskette
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_right_ineq)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
    
    def _declare_name(self):
        self.name = "Indicator Weak"

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_step_function(self,m,i):
        return m.y[i] == sum(m.f_values[i,j]* m.B[i,j] for j in m.I_intervals)

    def _exp_left_ineq(self,m,i,j):
        return m.grid_values[i,j]*m.B[i,j]+m.grid_values[i,1]*(1-m.B[i,j]) <= m.y[i-1]

    def _exp_right_ineq(self,m,i,j):
        return m.y[i-1] <= m.grid_values[i,j+1]*m.B[i,j]+m.grid_values[i,m.n_intervals+1]*(1-m.B[i,j])
    
    def _exp_sos1(self,m,i):
        return sum(m.B[i,j] for j in m.I_intervals) == 1

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.y[i-1]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.y[i-1]-m.y[i]

    def _make_data_dict(self, n_its, n_intervals , f_values, grid_values):
        data = {None: {
            'n_intervals':{ None: n_intervals},
            'n_its': {None: n_its},
            'f_values': {(i+1,j+1): f_values[i,j] for i in range(n_its) for j in range(n_intervals)},
            'grid_values': {(i+1,j+1): grid_values[i,j] for i in range(n_its) for j in range(n_intervals+1)},
        }}
        return data

class Indicator_Strong_Concrete(Indicator_Weak_Concrete):
    def _declare_name(self):
        self.name = "Indicator Strong Concrete"

    def _exp_left_ineq(self,m,i):
        return sum(self.grid_values[i-1,j-1] * m.B[i,j] for j in m.I_intervals) <= m.x[i]

    def _exp_right_ineq(self,m,i):
        return m.x[i] <= sum(self.grid_values[i-1,j] * m.B[i,j] for j in m.I_intervals)
    
    def _declare_constraints(self):
        # Dasselbe wie in Indicator Weak mit veränderten Funktionen
        # exp_left/right_eq und dementsprechend unterschiedlichen
        # Indexmengen
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

class Indicator_Strong(Indicator_Weak):
    def _declare_name(self):
        self.name = "Indicator Strong"

    def _exp_left_ineq(self,m,i):
        return sum(m.grid_values[i,j] * m.B[i,j] for j in m.I_intervals) <= m.y[i-1]

    def _exp_right_ineq(self,m,i):
        return m.y[i-1] <= sum(m.grid_values[i,j+1] * m.B[i,j] for j in m.I_intervals)
    
    def _declare_constraints(self):
        # Dasselbe wie in Indicator Weak mit veränderten Funktionen
        # exp_left/right_eq und dementsprechend unterschiedlichen
        # Indexmengen
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)

class Indicator_Strong_SOS1(Indicator_Strong):
    def _declare_name(self):
        self.name = "Indicator Strong SOS1"

    def _declare_constraints(self):
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)

        # SOS1
        def multi(m,i):
            return ((i,j) for j in m.I_intervals)

        self.model.slice_set = pyo.Set(self.model.I_its, initialize = multi)
        self.model.sos1 = SOSConstraint(self.model.I_its, var = self.model.B, index = self.model.slice_set, sos = 1)

        # Summe = 1 weiterhin spezifizieren, da SOS nur angibt, dass höchstens eines den Wert 1 hat.
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)

        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)

class Indicator_Weak_Strong(Indicator_Weak):
    def _declare_name(self):
        self.name = "Indicator Strong Weak"

    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        # Bedingung sum(B)=1
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        # Ungleichungskette, Hinzufügen der starken constraints
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its,self.model.I_intervals, rule = self._exp_right_ineq)
        self.model.const_strong_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_strong_left_ineq)
        self.model.const_storng_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_strong_right_ineq)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)

    def _exp_strong_left_ineq(self,m,i):
        return sum(m.grid_values[i,j] * m.B[i,j] for j in m.I_intervals) <= m.y[i-1]

    def _exp_strong_right_ineq(self,m,i):
        return m.y[i-1] <= sum(m.grid_values[i,j+1] * m.B[i,j] for j in m.I_intervals)

class Indicator_Layered(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "Indicator Layered" 

    def _declare_parameters(self):
        if np.log2(self.n_intervals) - int(np.log2(self.n_intervals)) > 1e-8:
            raise NotPowerOfTwo("Not a power of Two")
        self.m = int(np.log2(self.n_intervals))
        self.model.I_k = pyo.RangeSet(0, self.m)
        def kj_init(model):
            return [(k, j) for k in range(self.m+1) for j in range(1,int(2**(self.m-k)+1))]
        def kj_init_short(model):
            return [(k, j) for k in range(self.m) for j in range(1,int(2**(self.m-k)+1))]
        self.model.I_kj = pyo.Set(initialize = kj_init)
        self.model.I_kj_short =  pyo.Set(initialize = kj_init_short)

        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
    
    def _declare_variables(self):
        self.model.D = pyo.Var(self.model.I_its, self.model.I_kj, within = pyo.Boolean)
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_constraints(self):
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_right_ineq)
        self.model.const_sum1 = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_sum1)
        self.model.const_inheritance = pyo.Constraint(self.model.I_its, self.model.I_kj_short, rule = self._exp_inheritance)
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)

    def n_1(self, k,j):
        return (j-1)*2**k
        
    def n_2(self, k,j):
        return j* 2**k
        
    def _exp_left_ineq(self, model, i, k):
        return model.x[i] >= sum([self.grid_values[i-1, self.n_1(k,j)] *  model.D[i, k, j] for j in range(1,2**(self.m-k)+1)])
    
    def _exp_right_ineq(self, model, i, k):
        return model.x[i] <= sum([self.grid_values[i-1, self.n_2(k,j)] *  model.D[i, k, j] for j in range(1,2**(self.m-k)+1)])
    
    def _exp_sum1(self,model, i, k):
        return 1 == sum([model.D[i, k, j] for j in range(1,2**(self.m-k)+1)])
    
    def _exp_inheritance(self, model, i,k, j):
        if j%2 ==0:
            return model.D[i,k,j] <= model.D[i, k+1, j/2]
        else:
            return model.D[i,k,j] <= model.D[i, k+1, (j+1)/2]
    
    def _exp_step_function(self,model,i):
        return model.y[i] == sum(self.f_values[i-1,j-1]* model.D[i,0,j] for j in model.I_intervals)
    
    
    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]

class Binary_Model(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "Binary" 
    
    def _declare_parameters(self):
        if np.log2(self.n_intervals) - int(np.log2(self.n_intervals)) > 1e-8:
            raise NotPowerOfTwo("Not a power of Two")
        self.m = int(np.log2(self.n_intervals))
        self.model.I_k = pyo.RangeSet(1, self.m)
        def I_i_I_i_init(model):
            for i in range(1, self.n_intervals+1):
                I_i = self._I_i(i, self.n_intervals)[0]
                for j in I_i:
                    yield (i, j)
        def I_i_I_i_C_init(model):
            for i in range(1, self.n_intervals+1):
                I_i_C = self._I_i(i, self.n_intervals)[1]
                for j in I_i_C:
                    yield (i, j)
        
        self.model.I_i_I_i = pyo.Set(initialize = I_i_I_i_init)
        self.model.I_i_I_i_C = pyo.Set(initialize = I_i_I_i_C_init)
        #self.model.I_i_j = pyo.

        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
    
    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, bounds = (0,1))
        self.model.D = pyo.Var(self.model.I_its, self.model.I_k, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
        
    
    def _declare_constraints(self):
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_sum1 = pyo.Constraint(self.model.I_its, rule = self._exp_sum1)
        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_i_I_i, rule = self._exp_prod_1)
        self.model.const_prod_0 = pyo.Constraint(self.model.I_its, self.model.I_i_I_i_C, rule = self._exp_prod_0)
        self.model.const_prod_sum = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_sum)

        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)


        

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _exp_left_ineq(self,model, i):
        return model.x[i] >= sum(self.grid_values[i-1, j]*model.B[i,j+1] for j in range(self.n_intervals))
    
    def _exp_right_ineq(self,model, i):
        return model.x[i] <= sum(self.grid_values[i-1, j+1]*model.B[i,j+1] for j in range(self.n_intervals))
    
    def _exp_sum1(self,model, i):
        return 1 == sum([model.B[i,j] for j in range(1,self.n_intervals+1)])

    def _exp_prod_1(self, model, i, j, k):
        return model.B[i,j] <= model.D[i,k]
    
    def _exp_prod_0(self, model, i, j, k):
        return model.B[i,j] <= 1-model.D[i,k]

    def _exp_prod_sum(self, model, i, j):
        return model.B[i,j] >= 1-self.m + sum(model.D[i,k] for k in self._I_i(j,self.n_intervals)[0]) + sum(1 - model.D[i,k] for k in self._I_i(j,self.n_intervals)[1])

    def _exp_step_function(self,model,i):
        return model.y[i] == sum(self.f_values[i-1,j-1]* model.B[i,j] for j in model.I_intervals)

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)
    

    def _I_i(self, i, n):
        m = int(np.log2(n))
        C_temp = [int(x) for x in bin(i-1)[2:]]
        C = [0 for i in range(m-len(C_temp))]+C_temp
        I_i = m-np.argwhere(np.array(C)==1).flatten()
        I_i_C = m- np.argwhere(np.array(C)==0).flatten()
        return (I_i, I_i_C)
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]
    
class Binary_Model_2(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "Binary 2"

    def _declare_parameters(self):
        if np.log2(self.n_intervals) - int(np.log2(self.n_intervals)) > 1e-8:
            raise NotPowerOfTwo("Not a power of Two")
        self.m = int(np.log2(self.n_intervals))
        def I_i_init(model):
            for i in range(1,self.n_intervals+1):
                I_i,_ = self._bisection(i, range(1,self.n_intervals+1), 0)
                for tupel in I_i:
                    yield (i, tupel[0], tupel[1])
        
        def I_i_C_init(model):
            for i in range(1,self.n_intervals+1):
                _, I_i_C = self._bisection(i, range(1,self.n_intervals+1), 0)
                for tupel in I_i_C:
                    yield (i, tupel[0], tupel[1])
        
        self.model.I_i = pyo.Set(initialize = I_i_init)
        self.model.I_i_C = pyo.Set(initialize = I_i_C_init)

        def kj_init(model):
            return [(k, j) for k in range(self.m+1) for j in range(1,int(2**(self.m-k)+1))]
        def kj_init_short(model):
            return [(k, j) for k in range(1,self.m+1) for j in range(1,int(2**(self.m-k)+1))]
        self.model.I_kj = pyo.Set(initialize = kj_init)
        self.model.I_kj_short =  pyo.Set(initialize = kj_init_short)

        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
    
    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, bounds = (0,1))
        self.model.D = pyo.Var(self.model.I_its, self.model.I_kj_short, within = pyo.Boolean)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
        
    
    def _declare_constraints(self):
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_sum1 = pyo.Constraint(self.model.I_its, rule = self._exp_sum1)
        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_i, rule = self._exp_prod_1)
        self.model.const_prod_0 = pyo.Constraint(self.model.I_its, self.model.I_i_C, rule = self._exp_prod_0)
        self.model.const_prod_sum = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_sum)

        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)


        

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _exp_left_ineq(self,model, i):
        return model.x[i] >= sum(self.grid_values[i-1, j]*model.B[i,j+1] for j in range(self.n_intervals))
    
    def _exp_right_ineq(self,model, i):
        return model.x[i] <= sum(self.grid_values[i-1, j+1]*model.B[i,j+1] for j in range(self.n_intervals))
    
    def _exp_sum1(self,model, i):
        return 1 == sum([model.B[i,j] for j in range(1,self.n_intervals+1)])

    def _exp_prod_1(self, model, i, j, k, l):
        return model.B[i,j] <= model.D[i,k,l]
    
    def _exp_prod_0(self, model, i, j, k, l):
        return model.B[i,j] <= 1-model.D[i,k,l]

    def _exp_prod_sum(self, model, i, j):
        return model.B[i,j] >= 1-self.m + sum(model.D[i,tuple[0], tuple[1]] for tuple in self._bisection(j,range(1,self.n_intervals+1),0)[0]) + sum(1 - model.D[i,tuple[0], tuple[1]] for tuple in self._bisection(j,range(1,self.n_intervals+1),0)[1])
        

    def _exp_step_function(self,model,i):
        return model.y[i] == sum(self.f_values[i-1,j-1]* model.B[i,j] for j in model.I_intervals)

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]
    

    def _bisection(self, i, array, k):
        n = len(array)
        m = int(np.log2(self.n_intervals))
        add_index = int((array[0]-1)/n) +1
        if n == 1:
            return ([], [])
        n2 = int(n/2)
        if i<= array[n2-1]:
            I, Ic = self._bisection(i, array[:n2], k+1)
            Ic.append((m-k, add_index))
            return (I, Ic)
        else:
            I, Ic = self._bisection(i, array[n2:], k+1)
            I.append((m-k, add_index))
            return (I, Ic)

class Incremental(_Step_Function_Model):
    def _declare_name(self):
        self.name = "Incremental"
    
    def _declare_parameters(self):
        self.model.n_intervals = pyo.Param(within = pyo.NonNegativeIntegers)
        self.model.n_its = pyo.Param(within = pyo.NonNegativeIntegers)
        self.model.I_intervals = pyo.RangeSet(1, self.model.n_intervals)
        # Zusätzliche Indexmengen. An dieser Stelle wäre zu
        # überlegen ob man gegen das pyomo "best practice" gehen 
        # sollte.
        self.model.I_f_delta = pyo.RangeSet(1,self.model.n_intervals-1)
        self.model.I_B_delta = pyo.RangeSet(1,self.model.n_intervals-2)
        self.model.I_steps = pyo.RangeSet(1,self.model.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.model.n_its)
        self.model.I_y = pyo.RangeSet(0,self.model.n_its)

        # Funktionswerte und Stützstellen
        self.model.f_values = pyo.Param(self.model.I_its, self.model.I_intervals)
        self.model.grid_values = pyo.Param(self.model.I_its, self.model.I_steps)
        # Differenzen davon
        self.model.f_delta = pyo.Param(self.model.I_its, self.model.I_f_delta)
        self.model.grid_delta = pyo.Param(self.model.I_its, self.model.I_intervals)
    
    def _declare_variables(self):
        self.model.y = pyo.Var(self.model.I_y)
        self.model.B = pyo.Var(self.model.I_its, self.model.I_f_delta, within = pyo.Binary)
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)

    def _declare_constraints(self):
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_order = pyo.Constraint(self.model.I_its,self.model.I_B_delta, rule = self._exp_order)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_step_function(self,m,i):
        return m.y[i] == m.f_values[i,1] + sum(m.f_delta[i,j]* m.B[i,j] for j in m.I_f_delta)

    def _exp_left_ineq(self,m,i):
        return m.grid_values[i,1] + sum(m.grid_delta[i,j]*m.B[i,j] for j in m.I_f_delta) <= m.y[i-1]

    def _exp_right_ineq(self,m,i):
        return m.grid_values[i,2] + sum(m.grid_delta[i,j+1]*m.B[i,j] for j in m.I_f_delta) >= m.y[i-1]
    
    def _exp_order(self,m,i,j):
        return m.B[i,j+1]<= m.B[i,j]

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.y[i-1]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.y[i-1]-m.y[i]

    def _make_data_dict(self, n_its, n_intervals , f_values, grid_values):
        data = {None: {
            'n_intervals':{ None: n_intervals},
            'n_its': {None: n_its},
            'f_values': {(i+1,j+1): f_values[i,j] for i in range(n_its) for j in range(n_intervals)},
            'grid_values': {(i+1,j+1): grid_values[i,j] for i in range(n_its) for j in range(n_intervals+1)},
            'f_delta': {(i+1,j+1): f_values[i,j+1]- f_values[i,j] for i in range(n_its) for j in range(n_intervals-1)},
            'grid_delta': {(i+1,j+1): grid_values[i,j+1]- grid_values[i,j] for i in range(n_its) for j in range(n_intervals)},
        }}
        return data


class Incremental_Concrete(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "Incremental Concrete"
    
    def _declare_parameters(self):
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        # Zusätzliche Indexmengen. An dieser Stelle wäre zu
        # überlegen ob man gegen das pyomo "best practice" gehen 
        # sollte.
        self.model.I_f_delta = pyo.RangeSet(1,self.n_intervals-1)
        self.model.I_B_delta = pyo.RangeSet(1,self.n_intervals-2)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
        self.f_delta = np.array([[self.f_values[i,j+1]- self.f_values[i,j] for j in range(self.n_intervals-1)] for i in range(self.n_its)])
        self.grid_delta = np.array([[self.grid_values[i,j+1]- self.grid_values[i,j] for j in range(self.n_intervals)] for i in range(self.n_its)])
    
    def _declare_variables(self):
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.B = pyo.Var(self.model.I_its, self.model.I_f_delta, within = pyo.Binary)
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)

    def _declare_constraints(self):
        self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self._exp_step_function)
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, rule = self._exp_right_ineq)
        self.model.const_order = pyo.Constraint(self.model.I_its,self.model.I_B_delta, rule = self._exp_order)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_step_function(self,m,i):
        return m.y[i] == self.f_values[i-1,0] + sum(self.f_delta[i-1,j-1]* m.B[i,j] for j in m.I_f_delta)

    def _exp_left_ineq(self,m,i):
        return self.grid_values[i-1,0] + sum(self.grid_delta[i-1,j-1]*m.B[i,j] for j in m.I_f_delta) <= m.x[i]

    def _exp_right_ineq(self,m,i):
        return self.grid_values[i-1,1] + sum(self.grid_delta[i-1,j]*m.B[i,j] for j in m.I_f_delta) >= m.x[i]
    
    def _exp_order(self,m,i,j):
        return m.B[i,j+1]<= m.B[i,j]

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]





# class Incremental2(Incremental):
    # def declare_name(self):
    #     self.name = "Incremental 2"

    # def declare_variables(self):
    #     self.model.y = pyo.Var(self.model.I_y)
    #     self.model.B = pyo.Var(self.model.I_its, self.model.I_f_delta, within = pyo.Binary)
    #     self.model.v = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.NonNegativeReals)
    #     self.model.z = pyo.Var(self.model.I_its)

    # def declare_constraints(self):
    #     self.model.const_step_function = pyo.Constraint(self.model.I_its, rule = self.exp_step_function)
    #     self.model.const_left_ineq = pyo.Constraint(self.model.I_its, self.model.I_f_delta, rule = self.exp_left_ineq)
    #     self.model.const_right_ineq = pyo.Constraint(self.model.I_its, self.model.I_B_delta, rule = self.exp_right_ineq)
    #     self.model.const_x = pyo.Constraint(self.model.I_its, rule = self.exp_x)
    #     self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self.exp_abs1)
    #     self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self.exp_abs2)


    # def exp_left_ineq(self,m,i,j):
    #     return m.grid_delta[i,j]*m.B[i,j] <= m.v[i,j]

    # def exp_right_ineq(self,m,i,j):
    #     return m.v[i,j]<=m.grid_delta[i,j]*m.B[i,j+1]
    
    # def exp_x(self,m,i):
    #     return sum(m.v[i,j] for j in m.I_f_delta) == m.y[i-1]


class Pyomo_Step_Function(_Concrete_Step_Function_Model):
    # Representation für pyomo contstraints als Argument des
    # Konstruktors mitgeben
    def __init__(self, repn = 'SOS2'):
        self.repn = repn
        super().__init__()

    # Wir können Parameter und Objective function erben. Damit
    # auch die _make_data_dict Funktion
    def _declare_name(self):
        self.name = "Incremental Pyomo"    
    def _declare_parameters(self):
        # Dimensionsparameter
        # Index als Parameter einspeichern (pyomo doc sagt, das sei best practice)
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
    
    # keine expliziten Binärvariablen nötig im Vergleich zu 
    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x = y[0]
        lb = np.min([np.min(self.grid_values), np.min(self.f_values)])
        ub = np.max([np.max(self.grid_values), np.max(self.f_values)])
        self.model.y = pyo.Var(self.model.I_y, bounds = (lb,ub))
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
    
    def _declare_constraints(self):
        # Nachdem mir keine Art bekannt ist, Piecewise-Constraints
        # als pyomo-Constraint List zu erstellen, diese "hässliche"
        # Art mit setattr. 
        for i in range(self.n_its):
            gridarray = np.repeat([self.grid_values[i,j] for j in range(self.n_intervals+1)], 2)[1:-1].tolist()
            farray = np.repeat([self.f_values[i,j] for j in range(self.n_intervals)],2).tolist()
            setattr(self.model, "step_"+str(i), Piecewise(
                self.model.y[i+1], 
                self.model.y[i],
                pw_pts = gridarray,
                f_rule = farray,
                pw_repn = self.repn,
                pw_constr_type = 'EQ',
                warn_domain_coverage = True,
                ))
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)

    
    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.y[i-1]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.y[i-1]-m.y[i]
    
    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)
        

if __name__ == "__main__":
    np.random.seed(4)
    def I_i(i, n):
        m = int(np.log2(n))
        C_temp = [int(x) for x in bin(i-1)[2:]]
        C = [0 for i in range(m-len(C_temp))]+C_temp
        I_i = m-np.argwhere(np.array(C)==1).flatten()
        I_i_C = m- np.argwhere(np.array(C)==0).flatten()
        return (I_i, I_i_C)


    


    def random_variables(N,n):
        a = np.around(np.random.rand(N,n),4)
        xi = np.array([np.linspace(0,1,n+1).tolist() for i in range(N)])
        d = [[a[i][j+1]-a[i][j] for j in range(n-1)] for i in range(N)]
        delta = [[xi[i][j+1]-xi[i][j] for j in range(n)] for i in range(N)]
        return (a,xi,d,delta)
    
    n = 4
    m = int(np.log2(n))
    def bisection(i, array, n_init, k):
        n = len(array)
        m = int(np.log2(n_init))
        add_index = int((array[0]-1)/n) +1
        if n == 1:
            return ([], [])
        n2 = int(n/2)
        if i<= array[n2-1]:
            I, Ic = bisection(i, array[:n2], n_init, k+1)
            Ic.append((m-k, add_index))
            return (I, Ic)
        else:
            I, Ic = bisection(i, array[n2:], n_init, k+1)
            I.append((m-k, add_index))
            return (I, Ic)
    n = 16
    # print(bisection(2,[i for i in range(1, n+1)], n, 0))

    N =10
    n = 64
    a, xi, _,__ = random_variables(N,n)
    Model = Binary_Model()
    inst = Model.instantiate(N, n, a, xi)
    #inst.pprint()
    #inst.pprint()
    inst2 = Indicator_Weak().instantiate(N,n,a,xi)
    # print([str(i) for i in instance.I_intervals])
    #instance_step.pprint()
    opt = pyo.SolverFactory("gurobi")
    opt.options['TimeLimit'] =20
    tic = perf_counter()
    #try:
    results = opt.solve(inst, tee = True)
    print(results)
    # except:
    #     print(perf_counter()-tic)

    results2 = opt.solve(inst2)

    print(results2.problem.number_of_binary_variables)
    # print([inst.y[i].value for i in range(N+1)])



    #print([instance_step.y[i].value for i in instance_weak.I_y])
    #print([instance_weak.y[i].value for i in instance_weak.I_y])
    # print([instance.B[i,j].value for i in range(1,N+1) for j in range(1,n+1)] )
    #print(results)
    #print(results.problem.lower_bound)
    #print([pyo.value(instance.y[i]) for i in range(0,N)])