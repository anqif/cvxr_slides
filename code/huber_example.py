import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

# Problem data
np.random.seed(1)
n = 300
m = 450

beta_true = 5*np.random.normal(size=(n,1))
X = np.random.randn(n,m)
Y = np.zeros((m,1))
v = np.random.normal(size=(m,1))

# Generate data for different values of p.
TESTS = 50
lsq_data = np.zeros(TESTS)
huber_data = np.zeros(TESTS)
prescient_data = np.zeros(TESTS)
p_vals = np.linspace(0, 0.15, num=TESTS)

for idx, p in enumerate(p_vals):
	# Generate the sign changes.
	factor = 2*np.random.binomial(1, 1-p, size=(m,1)) - 1
	Y = factor*X.T.dot(beta_true) + v

	# Form and solve a standard regression problem.
	beta = Variable(n)
	fit = norm(beta - beta_true)/norm(beta_true)
	cost = norm(X.T*beta - Y)
	prob = Problem(Minimize(cost))
	prob.solve()
	lsq_data[idx] = fit.value
    
	# Form and solve a prescient regression problem, i.e., where the sign changes are known.
	cost = norm(mul_elemwise(factor, X.T*beta) - Y)
	Problem(Minimize(cost)).solve()
	prescient_data[idx] = fit.value
    
	# Form and solve the Huber regression problem.
	cost = sum_entries(huber(X.T*beta - Y, 1))
	Problem(Minimize(cost)).solve()
	huber_data[idx] = fit.value

if True:
	# Plot the relative reconstruction error for least-squares, prescient, and Huber regression.
	plt.plot(p_vals, lsq_data, label='Least squares')
	plt.plot(p_vals, huber_data, label='Huber')
	plt.plot(p_vals, prescient_data, label='Prescient')

	plt.title('Relative Reconstruction Error')
	plt.ylabel(r'$\||\beta - \beta^{\mathrm{true}}\||_2/\||\beta^{\mathrm{true}}\||_2$')
	plt.xlabel('p')
	plt.legend(loc='upper left')
	plt.show()

	# Plot the relative reconstruction error for Huber and prescient regression, zooming in on smaller values of p.
	indices = np.where(p_vals <= 0.08)
	
	plt.plot(p_vals[indices], huber_data[indices], 'g', label='Huber')
	plt.plot(p_vals[indices], prescient_data[indices], 'r', label='Prescient')
	
	plt.title('Relative Reconstruction Error')
	plt.ylabel(r'$\||\beta - \beta^{\mathrm{true}}\||_2/\||\beta^{\mathrm{true}}\||_2$')
	plt.xlabel('p')
	plt.xlim([0, 0.07])
	plt.ylim([0, 0.05])
	plt.legend(loc='upper left')
	plt.show()
else:
	# Plot the relative reconstruction error for least-squares, prescient, and Huber regression.
	width = 0.25
	bar_ind = np.arange(TESTS)
	plt.bar(bar_ind, lsq_data, width, color='b', label='Least squares')
	plt.bar(bar_ind + width, huber_data, width, color='g', label='Huber')
	plt.bar(bar_ind + 2*width, prescient_data, width, color='r', label='Prescient')
	
	plt.title('Relative Reconstruction Error')
	plt.ylabel(r'$\||\beta - \beta^{\mathrm{true}}\||_2/\||\beta^{\mathrm{true}}\||_2$')
	plt.xlabel('p')
	plt.legend(loc='upper left')
	plt.xticks(bar_ind + width, p_vals)
	plt.show()
	
	# Plot the relative reconstruction error for Huber and prescient regression, zooming in on smaller values of p.
	width = 0.35
	indices = np.where(p_vals <= 0.08)
	bar_ind = np.arange(len(p_vals[indices]))
	
	plt.bar(bar_ind + width, huber_data[indices], width, color='g', label='Huber')
	plt.bar(bar_ind + 2*width, prescient_data[indices], width, color='r', label='Prescient')
	
	plt.title('Relative Reconstruction Error')
	plt.ylabel(r'$\||\beta - \beta^{\mathrm{true}}\||_2/\||\beta^{\mathrm{true}}\||_2$')
	plt.xlabel('p')
	plt.xlim([0, 0.07])
	plt.ylim([0, 0.05])
	plt.legend(loc='upper left')
	plt.xticks(bar_ind + width, p_vals[indices])
	plt.show()
