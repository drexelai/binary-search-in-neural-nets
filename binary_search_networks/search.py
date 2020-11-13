# Author: Ethan Moyer
# Date: 2020/11/10
# Purpose: perform binary search from 1 to n

from binary_search_networks.pipeline import run_pipe

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def get_output_space(**args):
	accuracys = []
	a = 0
	b = args['n']
	for i in range(a, b + 1):
		train_acc, test_acc = run_pipe(**args)
		accuracys.append(train_acc)
	return a, b, accuracys


def plot_output_space(a, b, accuracys):
	x = [i for i in range(a, b + 1)]
	y = accuracys
	
	tck = interpolate.splrep(x, y, s=0, t=[np.argmax(y)])
	ynew = interpolate.splev(x, tck, der=0)
	plt.scatter(x=x, y=y)
	plt.plot(x, ynew, '--')
	plt.xlabel("Number of hidden layer units")
	plt.ylabel("Accuracy")

	plt.show()


def get_slope(**args):
	args['n'] = args['ni']
	_, yi = run_pipe(**args)
	args['n'] = args['nj']
	_, yj = run_pipe(**args)
	return (yj - yi) / (args['ni'] - args['nj'])


def get_posterior_prob(gamma1, gamma2, mx, my, delta, sigma=0.5):

	xi = mx[-1]
	yi = my[-1]

	del mx[-1]
	del my[-1]

	x = np.array(mx).reshape((-1, 1))
	y = np.array(my)

	model = LinearRegression().fit(x, y)
	y_pred = model.predict(xi)

	likelihood = norm(y_pred, sigma).pdf(yi)

	pior = delta / (gamma2 - gamma1)

	return likelihood * pior


def binary_search(**args):
	gamma1 = 1
	gamma2 = args['n']

	delta = args['delta']

	posterior_alpha = args['posterior_alpha']

	m1 = []
	m2 = []

	mid1 = []
	mid2 = []
	print(gamma1)
	print(gamma2)
	while gamma1 <= gamma2:
		mid = (gamma1 + gamma2)//2
		args['ni'] = mid - delta//2
		args['nj'] = mid + delta//2

		mi = get_slope(**args)

		if mi > 0:
			m1.append(mi)
			mid1.append(mid)
			args['ni'] = mid

			if get_posterior_prob(gamma1, gamma2, mid1, m1, delta) > posterior_alpha and get_slope(**args) < mi: # check if the slopes in between?

				if_found = True
				print("Maximum accuracy found at index {}".format(mid))
				return mid
			else:
				gamma1 = mid # + 1?
		else:
			m2.append(mi)
			mid2.append(mid)
			args['nj'] = mid

			if get_posterior_prob(gamma1, gamma2, mid2, m2, delta) > posterior_alpha and get_slope(**args) > mi:

				if_found = True
				print("Maximum accuracy found at index {}".format(mid))
				return mid

			else:
				gamm2 = mid # - 1?
			


