import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats.kde import gaussian_kde
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from tabulate import tabulate

def skew_sample(data, bias = None):
	if bias is None:
		bias = [1.0] * data.shape[1]
	num = np.exp(np.dot(data, bias))
	return num / sum(num)

def plot_pdf(data, weights = None, label = None):
	if weights is None:
		kde = gaussian_kde(data)
		dist_space = np.linspace(min(data), max(data), 100)
		plt.plot(dist_space, kde(dist_space), label = label)
	else:
		dens = KDEUnivariate(data)
		dens.fit(weights = weights, fft = False)
		dist_space = np.linspace(min(data), max(data), 100)
		plt.plot(dist_space, dens.evaluate(dist_space), label = label)

def plot_cdf(data, probs = None, label = None):
	if probs is None:
		probs = [1.0/len(data)] * len(data)
	distro = np.vstack((data, probs)).T
	dsort = np.sort(distro, axis = 0)
	ecdf = np.cumsum(dsort[:,1]).T
	plt.plot(dsort[:,0], ecdf, label = label)
	# ecdf = sm.distributions.ECDF(data)
	# dist_space = np.linspace(min(data), max(data), 100)
	# plt.plot(dist_space, ecdf(dist_space), label = label)

# Problem data
np.random.seed(1)
n = 2
m = 1000
msub = 100
rv_orig = bernoulli(0.5)
rv_skew = bernoulli(0.8)

# Generate original distribution
sex = np.random.choice([0,1], m, p = rv_orig.pmf([0,1]))
age = np.random.randint(10, 61, m)
mu = 5 * sex + 0.1 * age
X = np.array([sex, age]).T
y = np.random.normal(mu, 1.0)
b = np.mean(X, axis = 0)

# Generate skewed subsample
# skew = rv_skew.pmf(sex) * norm(20,15).pdf(age)
# skew = skew / sum(skew)
skew = skew_sample(X, [-0.95, -0.05])
sub = np.random.choice(m, msub, p = skew)

# Solve direct standardization problem
w = Variable(msub)
cost = sum_entries(entr(w))
constr = [w >= 0, sum_entries(w) == 1, X[sub,:].T * w == b]
prob = Problem(Maximize(cost), constr)
prob.solve()

# Compare probability of male/female
bsub = np.mean(X[sub,:], axis = 0)
table = [['Male', 1-b[0], 1-bsub[0], sum(w.value[sex[sub] == 0])[0,0]], ['Female', b[0], bsub[0], sum(w.value[sex[sub] == 1])[0,0]]]
print tabulate(table, headers = ['Sex', 'True', 'Sample', 'Estimate'])

# Plot the PDF of original and subsample
plot_pdf(y, label='True')
plot_pdf(y[sub], label='Sample')
plot_pdf(y[sub], weights = w.value, label='Estimate')
plt.xlabel('y')
plt.legend(loc='upper right')
plt.show()

# Plot the CDF of original, subsample, and estimate
plot_cdf(y, label='True')
plot_cdf(y[sub], label='Sample')
plot_cdf(y[sub], w.value.T[0], label='Estimate')
plt.xlabel('y')
plt.ylim([0, 1])
plt.legend(loc='upper left')
plt.show()
