import numpy as np
import itertools

def multiSet(k, M):
	"""Computes all combinations of M bins containing 
	number from 0-k keep those who sums upp to k

	Params:
		k: max order of polynomial
		M: number of dimensions (bins)

	Return:
		mSet: list of arrays which sets sums up to order k
	"""
	result = itertools.product(range(k+1),repeat=M) # compute all possible combinations of k number in M bins
	mSet = [np.array(i) for i in result if sum(i)==k] # computes all the combinations where the number in the bins sums up to k
	return mSet

def allSets(k,M):
	""" Append all multi sets 

	Params:
		k: order of polynomial
		M: number of variables

	Return:
		aSet: list of arrays containg all multiSets from 0-k
	"""
	aSet = multiSet(0,M)

	for i in range(1,k+1):
		aSet = np.append(aSet, multiSet(i, M), 0)

	return aSet
