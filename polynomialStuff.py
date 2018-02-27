import numpy as np
from aPC import rawMoments, polyCoeff
from computeSets import allSets
import matplotlib.pyplot as plt
from beam_func import beam_func

def polynomial_creator(*coefficients):

	def polynomial(x):
		res = 0
		for index, coeff in enumerate(coefficients):
			res += coeff * x** index
		return res
	return polynomial

def monomial_creator(*p):

	def monomial(*var):
		res = 1
		for i in range(len(p)):
			res = res*p[i](var[i])
		return res

	return monomial

D = np.loadtxt('np.csv', dtype='float', delimiter=',', usecols=(0, 1, 2), unpack=False, skiprows=1)

k=3
M=3

NQ = 1000 # size of actual simulations performed i.e. input data and output data
idx = np.random.randint(len(D), size=NQ) # select NQ rows from the simulated data D
R = D[idx,:] #the actual input data used in the test function.
U = beam_func(R[:,0], R[:,1], R[:,2])

aSets = allSets(k,M)
m = []

for i in range(len(aSets)):
    coeff = []
    for j in range(M):
        coeff.append(polyCoeff(rawMoments(R[:,j], aSets[i][j]), aSets[i][j]))
        
    p = []

    for c in range(len(coeff)):
        p.append(polynomial_creator(*coeff[c])) #the * is for unpacking arrays
    
    m.append(monomial_creator(*p))


# create Q vector    
Q = np.zeros((R.shape[0], len(m)))

for i in range(len(m)):
    for j in range(R.shape[0]):
        Q[j,i]=(m[i](*R[j,:]))    

#compute C coeff for the PCE
A = np.dot(Q.T,Q)
b = np.dot(Q.T, U)
C = np.linalg.solve(A,b)

#evaluate PCE at D
PCE = np.zeros((D.shape[0],len(m)))

for i in range(len(m)):
    for j in range(D.shape[0]):
        PCE[j,i]=(C[i]*m[i](*D[j,:])) 
        
pce = sum(PCE.T)

plt.hist(pce, bins=100, normed=True)
plt.hist(beam_func(D[:,0], D[:,1], D[:,2]), bins=100, normed=True)
labels= ["aPCnew","True"]
plt.legend(labels)
plt.show()
