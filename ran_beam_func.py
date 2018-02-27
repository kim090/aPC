import numpy as np

n = 10000

L = 0.5 * np.random.randn(n, 1) + 5 #beam length normal distribution with mean 5 m
E = 200*10**9+20*10**9*np.random.rand(n,1) # Youngs modulus 200 GPa = 200*10^9 Pa
q = 10000+100*np.random.rand(n,1) # force per m

d = np.column_stack((L, E, q))

np.savetxt('np.csv', d, fmt='%.2f', delimiter=',', header=" #1, #2, #3")

print(d)
