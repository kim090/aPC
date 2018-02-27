import itertools
import numpy as np
import matplotlib.pyplot as plt
from testFun import testFun
from computeSets import allSets

""" 
Arbitrary Polynomial Chaos: instead of using polynomials that depends on the distributions 
of respective input variables, we compute the polynomial coefficients from the raw momements of the 
data that is actually used. Once the PC expansion is done we feed it with more data from 
the different distributions.
"""

def rawMoments(R, k):
	"""Computes 2k raw statistical moments from data R and order k

	Params:
		R: the input data used
		k: the order k

	Return:
		rMom: the raw moments
	"""
	rMom = np.zeros(2*k)
	for i in range(2*k):
		rMom[i]=sum(R**i)/len(R)
	return rMom

def polyCoeff(rMom, k):
	"""Compute polynomial coefficients from the raw statistical moments
	
	Params:
		rMom: the raw moments
		k: the polynomial order
	
	Return:
		p: the polynomial coeffecients
	"""
	mu = np.zeros((k+1, k+1)) 
	mu[-1][-1]=1
	g = 0

	for i in range(k):
		for j in range(k+1):
			mu[i][j]= rMom[j+g] 
		g=g+1

	b = np.zeros(k+1)
	b[-1]=1

	p = np.linalg.solve(mu,b)
	#p = np.flip(p,0) # leading coefficients in polynomial is 1

	return p

def computeQ(NQ, k, M, R, U):
	"""Computes the coeffcients for the terms in the PC expansion

	Params:
		NQ: number of used variables in the simulation
		k: order of expansion
		M: dimensions
		R: the input numbers used in the simultation
		U: the output of the simulation

	Return:
		C: the coefficeints of the PC expansion
	"""
	aSet = allSets(k, M)
	Q = np.zeros((NQ, len(aSet)))
	prod = 1

	for i in range(len(aSet)):
		for j in range(M):
			rMom = rawMoments(R[:,j], aSet[i][j])
			p = polyCoeff(rMom, aSet[i][j])
			temp = np.polyval(p, R[:,j])
			prod = np.multiply(temp, prod)
		Q[:,i]=prod
		prod=1

	A = np.dot(Q.T,Q)
	b = np.dot(Q.T, U)
	C = np.linalg.solve(A,b)

	return C

def computePCE(k, M, D, R, C):
	""" Computes the PC expansions

	Params:
		k: order of the expansion
		M: dimensions
		D: n generated data from the known input distributions
		R: the input numbers used in the simultation, used to compute raw statistical moments
		C: coeffecients used in the PC expansion

	Return:
		sum(PCE.T): the output of the PC expansion
	"""
	aSet = allSets(k, M)
	PCE = np.zeros((len(D),len(aSet)))
	prod = 1

	for i in range(len(aSet)):
		for j in range(M):
			rMom = rawMoments(R[:,j], aSet[i][j]) #use the actual input data R to estimate raw moments
			p = polyCoeff(rMom, aSet[i][j])      
			temp = np.polyval(p, D[:,j])          #use simulated data D from the distributions to feed the PCE
			prod = np.multiply(temp, prod)
		PCE[:,i]=C[i]*prod
		prod = 1
	
	return sum(PCE.T)

#k - Order
k=2

# M = 3 dimensions for test
M = 3

# data in 3 variables
D = np.loadtxt('np.csv', dtype='float', delimiter=',', usecols=(0, 1, 2), unpack=False, skiprows=1)

NQ = 100 # size of actual simulations performed i.e. input data and output data
idx = np.random.randint(len(D), size=NQ) # select NQ rows from the simulated data D
R = D[idx,:] #the actual input data used in the test function.

# Our output vector containing result from simulations of D1-D3
U = testFun(R[:,0], R[:,1], R[:,2])

C = computeQ(NQ, k, M, R, U)
res = computePCE(k, M, D, R, C)

#some plotting
plt.hist(res, normed=True)
plt.hist(testFun(D[:,0], D[:,1], D[:,2]), normed=True)
labels= ["aPC","True"]
plt.legend(labels)
plt.show()
