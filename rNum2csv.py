import numpy as np

n = 10000

d1 = 2.5 * np.random.randn(n, 1) + 3
d2 = 0.016+0.1*np.random.rand(n,1)
d3 = np.random.lognormal(3, 1, n)

d = np.column_stack((d1, d2, d3))

np.savetxt('np.csv', d, fmt='%.2f', delimiter=',', header=" #1, #2, #3")

print(d)
