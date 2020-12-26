# Author: Ethan Moyer
# Date: 2020/11/10
# Purpose: perform binary search from 1 to n

from binary_search_networks.pipeline import run_pipe, run_model
from binary_search_networks.util import get_cusp


import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def get_output_space(**args):

	'''
	Purpose: 
	... retrieve the output space of the model from 1 to end.
	Returns: 
	... a: low bound of n
	... b: high bound of n
	... accuracy: a list of recorded accuracies at each recorded ni
	'''

	accuracys = []
	a = 1
	b = args['n']
	for ni in range(a, b + 1):
		args['n'] = ni
		train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1 = run_pipe(**args)
		accuracys.append(train_accuracy)
	return a, b, accuracys


def plot_output_space(a, b, accuracys):
	'''
	Purpose: display the output space using a scatter plot and draw a cubic spline where t0 is at the maximum observed accuracy.
	Returns: None
	'''

	x = [i for i in range(a, b + 1)]
	y = accuracys
	
	# TODO: not sure if t is the correct parameter to select where the splines are broken up
	tck = interpolate.splrep(x, y, s=0, t=[np.argmax(y)])
	ynew = interpolate.splev(x, tck, der=0)
	plt.scatter(x=x, y=y)
	plt.plot(x, ynew, '--')
	plt.xlabel("Number of hidden layer units")
	plt.ylabel("Accuracy")

	plt.show()


def plot_slopes(mx, my, model):
	'''
	Purpose: display the historically calculated slopes and visualize the linear regression through them.
	Returns: None
	'''

	plt.scatter(x=mx, y=my)
	my_pred = model.predict(mx)
	plt.plot(mx, my_pred, '--')
	plt.xlabel("Number of hidden layer units")
	plt.ylabel("Slope of secant line")

	plt.show()


def get_slope(**args):

	'''
	Purpose: given ni & nj in args, run the pipeline for both and calculate the slope of their respective train accuracies.
	Returns:
	... slope: the recorded slope between the accuracy at ni & nj separated by delta
	'''
	if args['use_cusp_dist']:
		dist = get_cusp(args['n'])
		return get_dist_slope(dist, **args)
	args['n'] = args['ni']
	_, val_acc_i, _, _, _, _, _ = run_pipe(**args)
	args['n'] = args['nj']
	_, val_acc_j, _, _, _, _, _ = run_pipe(**args)
	return (val_acc_i - val_acc_j) / (args['ni'] - args['nj'])

def get_dist_slope(dist, **args):
	return (dist[args['ni']]-dist[args['nj']]) / (args['ni'] - args['nj'])

# Problems:
# 1. 1/yi causes infinity to happen when yi is 0
# 2. Getting the normal distribution +std vertically doesn't measure how possible the value
# is when put in a normal distribution. If this was a case, a line with a larger slope, higher B_1
# will always have a higher prob. What I think is better is to get the distance of the points from the line by 
# doing
# a. Projection of the points onto the line
# b. Deducting the projections from the points to get the vectors which are perpendicular to the
# line denoting the distance
# c. Using b to calculate std and normal distribution prob

def get_posterior_prob(gamma1, gamma2, mx, my, delta, sigma=0.5):
	'''
	Purpose: calculate the posterior probability according to the following beysian equation:
		P(𝑚𝑎𝑥𝑖𝑚𝑢𝑚│𝑚_𝐿, m_𝑈, 𝛾_𝐿, 𝛾_𝑈, Δ) = P(𝑚_𝐿, 𝑚_𝑈│𝑚𝑎𝑥𝑖𝑚𝑢𝑚) * P(𝑚𝑎𝑥𝑖𝑚𝑢𝑚|𝛾_𝐿, 𝛾_𝑈, Δ)
		posterior = likelihood * prior
	where
		P(𝑦=𝑚_𝐿,𝑚_𝑈│𝑚𝑎𝑥𝑖𝑚𝑢𝑚) ~ N(𝑦 ̌=𝛽_0+𝛽_1 ∗ 𝑥, 𝜎) and
		P(𝑚𝑎𝑥𝑖𝑚𝑢𝑚│𝛾_𝐿, 𝛾_𝑈, Δ) = Δ / (𝛾_𝑈 − 𝛾_𝐿)
	Returns:
	... posterior: the product of the likelihood and prior which represents the probability that a maximum is between ni & nj
	'''

	# Compare the most recent slope to the past recorded slopes
	xi = mx[-1]
	yi = my[-1]

	del mx[-1]
	del my[-1]

	# Have to introduce three separate cases depending on the size of the previously recorded slopes

	# If there are not previously recorded slopes, model the probability if 1/yi because we expect a low probability of a nonzero slope but a high probability of a nonzero slope
	if len(mx) == 0:
		likelihood = norm(1/yi, sigma).pdf(1/yi)

	# If there is only one previously recorded slope, model the probability simply with the first recorded slope and a general sigma
	elif len(mx) == 1:
		likelihood = norm(my[0], sigma).pdf(yi)

	# If there are more than one recorded slopes, then model the probability using the linear regression relationship... this may be adapted to be a polynomial if it does not fit it well
	else:
		x = np.array(mx).reshape((-1, 1))
		y = np.array(my)

		model = LinearRegression().fit(x, y)
		my_pred = model.predict(mx)
		sigma = np.std(my_pred, my)

		y_pred = model.predict(xi)

		likelihood = norm(y_pred, sigma).pdf(yi)

	if yi < 0:
		likelihood *= -1

	pior = delta / (gamma2 - gamma1)

	return likelihood * pior


def binary_search(**args):
	'''
	Purpose: find the maximum accuracy using binary search
	Returns:
	... mid: the found maximum accuracy
	'''

	# Gamma1 & gamma2 keep track of the current recorded upper and lower bounds to the number of units.
	gamma1 = 1
	gamma2 = args['n']

	delta = args['delta']

	# This is a threshold for when there is sufficient evidence that there is a maximum between ni & nj
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

		# When we are on the left side of the maximum
		if mi > 0:
			m1.append(mi)
			mid1.append(mid)
			args['ni'] = mid
			# Get posterior probability (and if its sufficient, check the secant line on the respective side)
			if get_posterior_prob(gamma1, gamma2, mid1, m1, delta) > posterior_alpha:
				if get_slope(**args) < mi: # check if the slopes in between?
					print("Maximum accuracy found at index {}".format(mid))
					# TODO: decide if delta is sufficiently small than we can stop the search
					return mid
				# if delta is large (~50) such that the posterior begins to increase, decrease delta
				else:
					if delta > 3:
						delta /= 2
					elif delta < 6:
						delta = 3
			else:
				gamma1 = mid # + 1?
		# When we are on the right side of the maximum
		else:
			m2.append(mi)
			mid2.append(mid)
			args['nj'] = mid

			# Get posterior probability (and if its sufficient, check the secant line on the respective side)
			if get_posterior_prob(gamma1, gamma2, mid2, m2, delta) > posterior_alpha:
				if get_slope(**args) > mi:
					print("Maximum accuracy found at index {}".format(mid))
					# TODO: decide if delta is sufficiently small than we can stop the search
					return mid
				# if delta is large (~50) such that the posterior begins to increase, decrease delta
				else:
					if delta > 3:
						delta /= 2
					elif delta < 6:
						delta = 3

			else:
				gamm2 = mid # - 1?
			


