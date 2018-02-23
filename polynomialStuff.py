import numpy as np

def polynomial_creator(*coefficients):

	def polynomial(x):
		res = 0
		for index, coeff in enumerate(coefficients):
			res += coeff * x** index
		return res
	return polynomial

p1 = polynomial_creator(3,2,1,4)
p2 = polynomial_creator(2,1,0)
p3 = polynomial_creator(2,3)

def monomial_creator(*p):

	def monomial(*var):
		res = 1
		for i in range(len(p)):
			res = res*p[i](var[i])
		return res

	return monomial

m1 = monomial_creator(p1, p2, p3)

print(p1(2)*p2(2)*p3(2)) # 43, 4
print(m1(2,2,2)) # 172