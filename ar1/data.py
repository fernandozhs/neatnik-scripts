import numpy  as np
import pickle as p


# Initializes the process array.
x = np.empty(2000)

# Populates array with AR(1) process: x[t] = A + B*x[t-1] + N[t]
A = 0.5
B = 0.85
x[0] = 0
for t in range(1, len(x)):
    N = np.random.normal()
    x[t] = A + B*x[t-1] + N

# Reshapes and saves the tail of the resulting process realization.
data = np.reshape(x[-1000:], (1,1000,1))
p.dump(data, open('./data.p', 'wb'))