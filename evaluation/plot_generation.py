import numpy as np
import matplotlib.pyplot as plt

"""
Kurzes Skript um die Graphik der Treppenfunktionen in BA zu erzeugen
"""


n_its = 1
n_intervals = 8

sdelta1 = np.around(np.random.rand(n_its,n_intervals),4)
scum1 = np.cumsum(sdelta1, axis = 1)
sdelta2 = np.around(np.random.rand(n_its,n_intervals),4)
sdelta2_sorted = np.sort(sdelta2, axis = 1)
scum2 = np.cumsum(sdelta2_sorted, axis = 1)

sa1 =  np.around(np.random.rand(n_its,n_intervals),4)
sa2 = 1- np.array([row/row[-1] for row in scum1])
sa3 = 1- np.array([row/row[-1] for row in scum2])
xi = np.array([np.linspace(0,1,n_intervals+1).tolist() for i in range(n_its)])

pdelta1 = np.around(np.random.rand(n_its,n_intervals+1),4)
pcum1 = np.cumsum(pdelta1, axis = 1)
pdelta2 = np.around(np.random.rand(n_its,n_intervals+1),4)
pdelta2_sorted = np.sort(pdelta2, axis = 1)
pcum2 = np.cumsum(pdelta2_sorted, axis = 1)

pa1 =  np.around(np.random.rand(n_its,n_intervals+1),4)
pa2 = 1- np.array([row/row[-1] for row in pcum1])
pa3 = 1- np.array([row/row[-1] for row in pcum2])

fig, (step, plin) = plt.subplots(1,2, figsize = (10,5))
for i, array in enumerate([sa1,sa2,sa3]):
    for j in range(n_intervals):
        if j == 0:
            if i ==0:
                step.plot([xi[0,j], xi[0,j+1]], [array[0,j], array[0,j]], color = "C"+str(i), lw = 4, label = "Vollständig zufällig")
            if i ==1:
                step.plot([xi[0,j], xi[0,j+1]], [array[0,j], array[0,j]], color = "C"+str(i), lw = 4, label = "Monoton")
            if i ==2:
                step.plot([xi[0,j], xi[0,j+1]], [array[0,j], array[0,j]], color = "C"+str(i), lw = 4, label = "Annähernd konkav")    
        else:
            step.plot([xi[0,j], xi[0,j+1]], [array[0,j], array[0,j]], color = "C"+str(i), lw = 4)

plin.plot(xi[0,:], pa1[0,:], color = "C0", lw = 4, label = "Vollständig zufällig")
plin.plot(xi[0,:], pa2[0,:], color = "C1", lw = 4, label = "Monoton")
plin.plot(xi[0,:], pa3[0,:], color = "C2", lw = 4, label = "Konkav")




step.set_ylim(0,1)
step.set_xlim(0,1)
plin.set_ylim(0,1)
plin.set_xlim(0,1)
step.legend()
plin.legend()
plt.show()