import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

data = np.random.random((100,2))
diagrams = ripser(data)['dgms']
plot_diagrams(diagrams, show=True)
plt.scatter(data[:,0], data[:,1])
plt.show()
