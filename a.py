import numpy as np


a = np.array([1, 0, 3])
b = np.zeros((a.size, a.max() + 1))
b[np.arange(a.size), a] = 1

print(b)

