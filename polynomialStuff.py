import numpy as np
import matplotlib.pyplot as plt

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

coeff = []

p1 = polynomial_creator(*[3,2,1,4])
p2 = polynomial_creator(*[2,1,0])
p3 = polynomial_creator(*[2,3])

coeff.append([3,2,1,4]) 
coeff.append([2,1,0])
coeff.append([2,3])

p = []

for i in range(len(coeff)):
    p.append(polynomial_creator(*coeff[i])) #the * is for unpacking arrays

m1 = monomial_creator(p1, p2, p3)
m2 = monomial_creator(*p) #the * is for unpacking list into functions

print(p1(2)*p2(2)*p3(2)) 
print(m1(2,2,2)) 
print(m2(2,2,2))

plt.plot(2,2)
plt.show()