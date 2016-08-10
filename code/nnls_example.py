import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from cvxpy import *

# Generate problem data
s = 1
m = 10
n = 30
mu = [0] * 9
Sigma = np.array([[1.6484, -0.2096, -0.0771, -0.4088, 0.0678, -0.6337, 0.9720, -1.2158, -1.3219],
                  [-0.2096, 1.9274, 0.7059, 1.3051, 0.4479, 0.7384, -0.6342, 1.4291, -0.4723],
                  [-0.0771, 0.7059, 2.5503, 0.9047, 0.9280, 0.0566, -2.5292, 0.4776, -0.4552],
                  [-0.4088, 1.3051, 0.9047, 2.7638, 0.7607, 1.2465, -1.8116, 2.0076, -0.3377],
                  [0.0678, 0.4479, 0.9280, 0.7607, 3.8453, -0.2098, -2.0078, -0.1715, -0.3952],
                  [-0.6337, 0.7384, 0.0566, 1.2465, -0.2098, 2.0432, -1.0666,  1.7536, -0.1845],
                  [0.9720, -0.6342, -2.5292, -1.8116, -2.0078, -1.0666, 4.0882,  -1.3587, 0.7287],
                  [-1.2158, 1.4291, 0.4776, 2.0076, -0.1715, 1.7536, -1.3587, 2.8789, 0.4094],
                  [-1.3219, -0.4723, -0.4552, -0.3377, -0.3952, -0.1845, 0.7287, 0.4094, 4.8406]])

np.random.seed(1)
X = np.random.multivariate_normal(mu, Sigma, n)
X = np.insert(X, 0, [1] * n, axis = 1)
b = np.array([0, 0.8, 0, 1, 0.2, 0, 0.4, 1, 0, 0.7]).T
y = np.dot(X, b) + np.random.normal(0, s, n)
print 'True coefficients\n', b

# Construct the unconstrained OLS problem
beta = Variable(m)
objective = Minimize(sum_squares(y - X*beta))
p = Problem(objective)

# Solve for the OLS beta coefficients
result = p.solve()
fit1 = np.dot(X,beta.value)
beta1 = np.squeeze(np.asarray(beta.value))
print '\nFitted coefficients\n', beta1

# Add non-negativity constraint to problem
constraints = [beta >= 0]
p = Problem(objective, constraints)

# Solve the problem with non-negativity constraint
result2 = p.solve()
fit2 = np.dot(X,beta.value)
beta2 = np.squeeze(np.asarray(beta.value))
print '\nFitted coefficients with constraint\n', beta2

if True:
	ind = np.arange(m)
	width = 0.25
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, beta2, width = width, color = 'b', label = 'Non-negative Least Squares')
	rects2 = ax.bar(ind + width, b, width = width, color = 'r', label = 'True Coefficients')
	rects3 = ax.bar(ind + 2*width, beta1, width = width, color = 'y', label = 'Least Squares')
	
	ax.set_ylabel('Coefficients')
	ax.set_ylim([-1, 3])
	ax.set_xticks(ind + 1.5*width)
	ax.set_xticklabels([r'$\beta_{}$'.format(s) for s in ind])
	ax.legend(loc='upper right')
	plt.show()
else:
	# Compare fitted coefficients
	fig = plt.figure(figsize=(12, 6))
	sub1 = fig.add_subplot(121)
	sub2 = fig.add_subplot(122)
	t = np.arange(1, m+1)

	# Plot true coefficients
	sub1.plot(t, b, 'b^')
	sub2.plot(t, b, 'b^')

	# Subplots of fitted coefficients
	sub1.vlines(t, [0], beta1)
	sub1.set_title("Fitted Coefficients of Least Squares")
	sub2.vlines(t, [0], beta2)
	sub2.set_title("Fitted Coefficients of Non-negative Least Squares")

plt.show()
