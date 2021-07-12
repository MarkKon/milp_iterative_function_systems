import pyomo.environ as pyo
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.piecewise import Piecewise
import numpy as np
from time import perf_counter

'''
Klasse mit Pyomo-Modellen für iterierte Funktionensysteme mit
stückweise linearen Funktionen. Die Modelle sind Subklassen der Klasse
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

class _Concrete_Step_Function_Model:
    def instantiate(self, n_its, n_intervals,f_values,  f_intercepts, f_slopes, grid_values):
        self.model = pyo.ConcreteModel()
        self.n_intervals = n_intervals
        self.n_its = n_its
        self.f_values = f_values
        self.f_intercepts = f_intercepts
        self.f_slopes = f_slopes
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
        self.model.w = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.NonNegativeReals)
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

        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_1)
        self.model.const_prod_2 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_2)
        self.model.const_prod_3 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_3)

        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_name(self):
        self.name = "Indicator Weak"

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_step_function(self,m,i):
        return m.y[i] == sum(self.f_intercepts[i-1,j-1]* m.B[i,j] + self.f_slopes[i-1,j-1]*(m.w[i,j]+self.grid_values[i-1,0]) for j in m.I_intervals)


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

    def _exp_prod_1(self,m, i, j):
        return m.w[i,j] <= (self.grid_values[i-1,j]-self.grid_values[i-1,0])*m.B[i,j]
    
    def _exp_prod_2(self,m,i,j):
        return m.w[i,j] <= m.x[i] -self.grid_values[i-1,0]

    def _exp_prod_3(self, m, i,j):
        return m.w[i,j] >= m.x[i]-self.grid_values[i-1,0] -(self.grid_values[i-1,-1]-self.grid_values[i-1,0])*(1-m.B[i,j])
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]

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
        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_1)
        self.model.const_prod_2 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_2)
        self.model.const_prod_3 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_3)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_name(self):
        self.name = "Indicator Strong"

    def _exp_left_ineq(self,m,i):
        return sum(self.grid_values[i-1,j-1] * m.B[i,j] for j in m.I_intervals) <= m.y[i-1]

    def _exp_right_ineq(self,m,i):
        return m.y[i-1] <= sum(self.grid_values[i-1,j] * m.B[i,j] for j in m.I_intervals)
    
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
        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_1)
        self.model.const_prod_2 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_2)
        self.model.const_prod_3 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_3)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

class DCC_Concrete(_Concrete_Step_Function_Model):
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
        self.model.lambd = pyo.Var(self.model.I_its,self.model.I_intervals, within = pyo.NonNegativeReals)
        self.model.mu = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.NonNegativeReals)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
    
    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_plin_function = pyo.Constraint(self.model.I_its, rule = self._exp_plin_function)
        # x als Konvexkombination
        self.model.const_x_convex = pyo.Constraint(self.model.I_its, rule = self._exp_x_convex)
        # Bedingung sum(B)=1
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        self.model.const_convexity = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_convexity)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_name(self):
        self.name = "DCC"

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_plin_function(self,m,i):
        return m.y[i] == sum(m.lambd[i,j]*self.f_values[i-1,j-1] + m.mu[i,j]*self.f_values[i-1,j] for j in m.I_intervals)

    def _exp_x_convex(self, m, i):
        return m.x[i] == sum(m.lambd[i,j]*self.grid_values[i-1,j-1] + m.mu[i,j]*self.grid_values[i-1,j] for j in m.I_intervals)
    
    def _exp_convexity(self, m, i, j):
        return m.lambd[i,j] + m.mu[i,j] == m.B[i,j]

    def _exp_sos1(self,m,i):
        return sum(m.B[i,j] for j in m.I_intervals) == 1

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]

    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i] 

class CC_Concrete(_Concrete_Step_Function_Model):
    def _declare_parameters(self):
        # Index als Parameter einspeichern (pyomo doc sagt, das sei best practice)
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_inside = pyo.RangeSet(2,self.n_intervals)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)

    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.lambd = pyo.Var(self.model.I_its,self.model.I_steps, within = pyo.NonNegativeReals)
        # Binärvariablen
        self.model.B = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
    
    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_plin_function = pyo.Constraint(self.model.I_its, rule = self._exp_plin_function)
        # x als Konvexkombination
        self.model.const_x_convex = pyo.Constraint(self.model.I_its, rule = self._exp_x_convex)
        self.model.const_convexity = pyo.Constraint(self.model.I_its, rule = self._exp_convexity)
        #SOS2:
        self.model.const_sos_first = pyo.Constraint(self.model.I_its, rule = self._exp_sos_first)
        self.model.const_sos_mid = pyo.Constraint(self.model.I_its, self.model.I_inside, rule = self._exp_sos_mid)
        self.model.const_sos_last = pyo.Constraint(self.model.I_its, rule = self._exp_sos_last)
        # Bedingung sum(B)=1
        self.model.const_sos1 = pyo.Constraint(self.model.I_its, rule = self._exp_sos1)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _declare_name(self):
        self.name = "CC"

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_plin_function(self,m,i):
        return m.y[i] == sum(m.lambd[i,j]*self.f_values[i-1,j-1] for j in m.I_intervals) + m.lambd[i,self.n_intervals+1]*self.f_values[i-1,-1]

    def _exp_x_convex(self, m, i):
        return m.x[i] == sum(m.lambd[i,j]*self.grid_values[i-1,j-1]  for j in m.I_steps)

    def _exp_convexity(self,m,i):
        return sum(m.lambd[i,j] for j in m.I_steps) ==1
    
    def _exp_sos_first(self, m, i):
        return m.lambd[i,1]  <= m.B[i,1]
    
    def _exp_sos_mid(self, m, i,j):
        return m.lambd[i,j] <= m.B[i,j-1] + m.B[i,j]

    def _exp_sos_last(self,m, i):
        return m.lambd[i,self.n_intervals+1] <= m.B[i,self.n_intervals]

    def _exp_sos1(self,m,i):
        return sum(m.B[i,j] for j in m.I_intervals) == 1

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]

    def _exp_prod_1(self,m, i, j):
        return m.y[i,j] <= self.grid_values[i-1,j]*m.B[i,j]
    
    def _exp_prod_2(self,m,i,j):
        return m.y[i,j] <= m.x[i]

    def _exp_prod_3(self, m, i,j):
        return m.y[i,j] >= m.x[i]- self.grid_values[i-1,-1]*(1-m.B[i,j])
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i] 

class Incremental_Concrete(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "Incremental Concrete"
    
    def _declare_parameters(self):
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_short = pyo.RangeSet(1,self.n_intervals-1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
        self.f_delta = self.f_values[:,1:]-self.f_values[:,:-1]
        self.grid_delta = self.grid_values[:,1:]-self.grid_values[:,:-1]
    
    def _declare_variables(self):
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.theta = pyo.Var(self.model.I_its, self.model.I_intervals, bounds = (0,1))
        self.model.B = pyo.Var(self.model.I_its, self.model.I_short, within = pyo.Binary)
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)

    def _declare_constraints(self):
        self.model.const_plin_function = pyo.Constraint(self.model.I_its, rule = self._exp_plin_function)
        self.model.const_x_lin_com = pyo.Constraint(self.model.I_its, rule = self._exp_x_lin_com)
        self.model.const_order_first = pyo.Constraint(self.model.I_its, rule = self._exp_order_first)
        self.model.const_order_mid_left = pyo.Constraint(self.model.I_its,self.model.I_short, rule = self._exp_order_mid_left)
        self.model.const_order_mid_right = pyo.Constraint(self.model.I_its,self.model.I_short, rule = self._exp_order_mid_right)
        self.model.const_order_last = pyo.Constraint(self.model.I_its, rule = self._exp_order_last)
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_plin_function(self,m,i):
        return m.y[i] == self.f_values[i-1,0] + sum(self.f_delta[i-1,j-1]* m.theta[i,j] for j in m.I_intervals)
    
    def _exp_x_lin_com(self,m,i):
        return m.x[i] == self.grid_values[i-1,0] + sum(self.grid_delta[i-1,j-1]*m.theta[i,j] for j in m.I_intervals)
    
    def _exp_order_first(self, m, i):
        return m.theta[i, 1] <= 1
    
    def _exp_order_mid_left(self,m, i, j):
        return m.theta[i,j+1] <= m.B[i,j]
    
    def _exp_order_mid_right(self,m, i, j):
        return m.B[i,j] <= m.theta[i,j]
    
    def _exp_order_last(self, m, i):
        return m.theta[i, self.n_intervals] >= 0

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]

class DLog_Concrete(_Concrete_Step_Function_Model):
    def _declare_name(self):
        self.name = "DLog"
    
    def Pl1(self, l):
        return [i+1 for i in range(self.n_intervals) if i%(2**l)<2**(l-1)]

    def Pl0(self,l):
        return [i+1 for i in range(self.n_intervals) if i%(2**l)>=2**(l-1)]
    
    def _declare_parameters(self):
        if np.log2(self.n_intervals) - int(np.log2(self.n_intervals)) > 1e-8:
            raise NotPowerOfTwo("Not a power of Two")
        self.n_log = int(np.log2(self.n_intervals))
        self.model.I_intervals = pyo.RangeSet(1, self.n_intervals)
        self.model.I_steps = pyo.RangeSet(1,self.n_intervals + 1)
        self.model.I_its = pyo.RangeSet(1,self.n_its)
        self.model.I_y = pyo.RangeSet(0,self.n_its)
        self.model.I_log = pyo.RangeSet(1,self.n_log)


    def _declare_variables(self):
        # Funktionswert y= f(x) sowie x
        self.model.y = pyo.Var(self.model.I_y)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.lambd = pyo.Var(self.model.I_its,self.model.I_intervals, within = pyo.NonNegativeReals)
        self.model.mu = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.NonNegativeReals)
        # Binärvariablen
        self.model.D = pyo.Var(self.model.I_its, self.model.I_log, within = pyo.Binary)
        # Betragswerte
        self.model.z = pyo.Var(self.model.I_its)
    
    def _declare_objective_function(self):
        self.model.OBJ = pyo.Objective(rule = self._exp_obj)
    
    def _declare_constraints(self):
        # Modellierung der Treppenfunktion
        self.model.const_plin_function = pyo.Constraint(self.model.I_its, rule = self._exp_plin_function)
        # x als Konvexkombination
        self.model.const_sum1 = pyo.Constraint(self.model.I_its, rule= self._exp_sum1)
        self.model.const_x_convex = pyo.Constraint(self.model.I_its, rule = self._exp_x_convex)
        self.model.const_convexity1 = pyo.Constraint(self.model.I_its, self.model.I_log, rule = self._exp_convexity1)
        self.model.const_convexity0 = pyo.Constraint(self.model.I_its, self.model.I_log, rule = self._exp_convexity0)
        # Modellierung von |f(x)-x|
        self.model.const_abs1 = pyo.Constraint(self.model.I_its, rule = self._exp_abs1)
        self.model.const_abs2 = pyo.Constraint(self.model.I_its, rule = self._exp_abs2)
        self.model.const_link_xy = pyo.Constraint(self.model.I_its, rule = self._exp_link_xy)

    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)

    def _exp_plin_function(self,m,i):
        return m.y[i] == sum(m.lambd[i,j]*self.f_values[i-1,j-1] + m.mu[i,j]*self.f_values[i-1,j] for j in m.I_intervals)

    def _exp_x_convex(self, m, i):
        return m.x[i] == sum(m.lambd[i,j]*self.grid_values[i-1,j-1] + m.mu[i,j]*self.grid_values[i-1,j] for j in m.I_intervals)
    
    def _exp_sum1(self, m, i):
        return sum(m.lambd[i,j] + m.mu[i,j] for j in m.I_intervals) == 1
    
    def _exp_convexity1(self, m, i, l):
        return sum(m.lambd[i,j] + m.mu[i,j] for j in self.Pl1(l)) <= m.D[i,l]

    def _exp_convexity0(self, m, i, l):
        return sum(m.lambd[i,j] + m.mu[i,j] for j in self.Pl0(l)) <= 1-m.D[i,l]

    def _exp_sos1(self,m,i):
        return sum(m.B[i,j] for j in m.I_intervals) == 1

    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]

    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i] 

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
        self.model.w = pyo.Var(self.model.I_its, self.model.I_intervals, within = pyo.NonNegativeReals)
        self.model.x = pyo.Var(self.model.I_its)
        self.model.z = pyo.Var(self.model.I_its)

    def _declare_constraints(self):
        self.model.const_left_ineq = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_left_ineq)
        self.model.const_right_ineq = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_right_ineq)
        self.model.const_sum1 = pyo.Constraint(self.model.I_its, self.model.I_k, rule = self._exp_sum1)
        self.model.const_inheritance = pyo.Constraint(self.model.I_its, self.model.I_kj_short, rule = self._exp_inheritance)

        self.model.const_prod_1 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_1)
        self.model.const_prod_2 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_2)
        self.model.const_prod_3 = pyo.Constraint(self.model.I_its, self.model.I_intervals, rule = self._exp_prod_3)

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

    def _exp_prod_1(self,m, i, j):
        return m.w[i,j] <= (self.grid_values[i-1,j]-self.grid_values[i-1,0])*m.D[i,0,j]
    
    def _exp_prod_2(self,m,i,j):
        return m.w[i,j] <= m.x[i] -self.grid_values[i-1,0]

    def _exp_prod_3(self, m, i,j):
        return m.w[i,j] >= m.x[i]-self.grid_values[i-1,0] -(self.grid_values[i-1,-1]-self.grid_values[i-1,0])*(1-m.D[i,0,j])
    
    def _exp_step_function(self,model,i):
        return model.y[i] == sum(self.f_intercepts[i-1,j-1]* model.D[i,0,j] + self.f_slopes[i-1,j-1]*(model.w[i,j]+self.grid_values[i-1,0]) for j in model.I_intervals)
    
    
    def _exp_abs1(self,m, i):
        return m.z[i] >= m.y[i]-m.x[i]

    def _exp_abs2(self,m,i):
        return m.z[i] >= m.x[i]-m.y[i]
    
    def _exp_obj(self, m):
        return sum(m.z[i] for i in m.I_its)
    
    def _exp_link_xy(self,m, i):
        return m.y[i-1] == m.x[i]

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
            setattr(self.model, "step_"+str(i), Piecewise(
                self.model.y[i+1], 
                self.model.y[i],
                pw_pts = self.grid_values[i,:],
                f_rule = self.f_values[i,:],
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

    def Pl1(n, l):
        return [i+1 for i in range(n) if i%(2**l)<2**(l-1)]

    print(Pl1(16,2))

    np.random.seed(5)
    def I_i(i, n):
        m = int(np.log2(n))
        C_temp = [int(x) for x in bin(i-1)[2:]]
        C = [0 for i in range(m-len(C_temp))]+C_temp
        I_i = m-np.argwhere(np.array(C)==1).flatten()
        I_i_C = m- np.argwhere(np.array(C)==0).flatten()
        return (I_i, I_i_C)


    


    def gen_data(n_its,n_intervals):
        grid_values = np.array([np.linspace(0,1,n_intervals+1).tolist() for i in range(n_its)])
        f_delta = np.around(np.random.rand(n_its,n_intervals+1),4)
        f_delta_sorted = np.sort(f_delta, axis = 1)
        f_cum = np.cumsum(f_delta_sorted, axis = 1)
        f_values = 1- np.array([row/row[-1] for row in f_cum])
        f_slopes = (f_values[:,1:]-f_values[:,:-1])/(grid_values[:,1:]-grid_values[:,:-1])
        f_intercepts = f_values[:,:-1] - f_slopes*grid_values[:,:-1]
        return (f_values, f_intercepts, f_slopes,grid_values)
    
    n = 4
    m = int(np.log2(n))
    N =2
    n =4
    f_values, f_intercepts, f_slopes, grid_values = gen_data(N,n)
    Model = DLog_Concrete()
    inst = Model.instantiate(N, n, f_values,f_intercepts, f_slopes, grid_values)
    #inst.pprint()
    #inst.pprint()
    #inst2 = Indicator_Weak().instantiate(N,n,a,xi)
    # print([str(i) for i in instance.I_intervals])
    #instance_step.pprint()
    opt = pyo.SolverFactory("gurobi")
    opt.options['TimeLimit'] =5
    tic = perf_counter()
    #try:
    results = opt.solve(inst, tee = True)
    #print(results)
    # except:
    #     print(perf_counter()-tic)

    #results2 = opt.solve(inst2)

    #print(results2.problem.number_of_binary_variables)
    #print([inst.y[i].value for i in range(N+1)])
    #print([inst.w[1,i].value for i in inst.I_intervals])


    #print([inst.lambd[1,i].value for i in inst.I_steps])
    #print([inst.B[1,i].value for i in inst.I_short])
    #print([instance_weak.y[i].value for i in instance_weak.I_y])
    # print([instance.B[i,j].value for i in range(1,N+1) for j in range(1,n+1)] )
    #print(results)
    #print(results.problem.lower_bound)
    #print([pyo.value(instance.y[i]) for i in range(0,N)])