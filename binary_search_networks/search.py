# Author: Ethan Moyer, Isamu Isozaki
# Date: 2020/11/10
# Purpose: perform binary search from 1 to n

from binary_search_networks.pipeline import run_pipe
from binary_search_networks.util import get_cusp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import random

# Add function to set seed
#random.seed(seed)

def get_output_space(**args):

	'''
	Purpose: 
	... retrieve the output space of the model from 1 to end.
	... save models and experiment data
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
	plt.savefig(f"{args['fig_save_dir']}/{args['fig_save_name']}")


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
		dist = get_cusp(args['n'], x=MID, seed=seed)
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

def get_posterior_prob(gamma1, gamma2, mid, m, delta, side, sigma=0.5):
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

	xi = mid[-1]
	yi = m[-1]

	del mid[-1]
	del m[-1]

	# Have to introduce three separate cases depending on the size of the previously recorded slopes

	# If there are not previously recorded slopes, model the probability if 1/yi because we expect a low probability of a nonzero slope but a high probability of a nonzero slope

	if len(m) == 0:
		likelihood = 0.5

	# If there is only one previously recorded slope, model the probability simply with the first recorded slope and a general sigma
	elif len(m) == 1:
		likelihood = norm(m[0], sigma).cdf(yi)

	# If there are more than one recorded slopes, then model the probability using the linear regression relationship... this may be adapted to be a polynomial if it does not fit it well
	# TODO: add in a condition for when the length is 2 (because sigma will be 0) and then else (>2)
	else:	
		x = np.array([mid]).reshape((-1, 1))
		y = np.array(m)

		model = LinearRegression().fit(x, y)
		y_pred = model.predict(x)
		my_pred = model.predict(np.array([xi]).reshape((-1, 1)))
		sigma = np.std(y_pred)
		likelihood = norm(my_pred, sigma).cdf(yi)

	if side == 1:
			likelihood = 1 - likelihood

	pior = delta / (gamma2 - gamma1)

	print("Likelihood: {}".format(likelihood))
	print("Pior: {}".format(pior))

	mid.append(xi)
	m.append(yi)

	return likelihood * pior


def binary_search(**args):
	'''
	Purpose: find the maximum accuracy using binary search
	Returns:
	... mid: the found maximum accuracy
	'''
	global MID
	global seed
	MID = random.random()
	seed = random.random()

	# NOTE: TEMP
	itereration = 0
	# Gamma1 & gamma2 keep track of the current recorded upper and lower bounds to the number of units.
	gamma1 = 1
	gamma2 = args['n']

	delta = args['delta']

	# This is a threshold for when there is sufficient evidence that there is a maximum between ni & nj
	prior = delta / (gamma2 - gamma1)
	posterior_alpha = 0.9 * prior #args['posterior_alpha']

	m1 = []
	m2 = []

	mid1 = []
	mid2 = []
	while gamma1 <= gamma2 and itereration < 10:
		#if len(mid1) > 0:
		#	if mid1.count(mid1[-1]) < 1:	break
		print("Gamma L: {}".format(gamma1))
		print("Gamma U: {}".format(gamma2))
		mid = (gamma1 + gamma2)//2

		args['ni'] = int(mid - delta//2)
		args['nj'] = int(mid + delta//2)
		print("ni: {}".format(args['ni']))
		print("nj: {}".format(args['nj']))

		mi = get_slope(**args)
		print("Slope: {}".format(mi))
		# When we are on the left side of the maximum
		if mi > 0:
			m1.append(mi)
			mid1.append(mid)
			print("Mid1 values:", mid1)
			print("m1 values:", m1)
			print("Mid2 values:", mid2)
			print("m2 values:", m2)
			args['ni'] = mid
			# Get posterior probability (and if its sufficient, check the secant line on the respective side)
			posterior_prob = get_posterior_prob(1, args['n'], mid1, m1, delta, side=1)
			print("probability: {}".format(posterior_prob))
			print("Itererations", itereration)
			if posterior_prob > posterior_alpha:
				#if get_slope(**args) < mi: # check if the slopes in between?
				print("Maximum accuracy found at index {}".format(mid))
					# TODO: decide if delta is sufficiently small than we can stop the search
				y = get_cusp(args['n'], x=MID, seed=seed)
				actual_max = np.argmax(y)
				#plt.plot(y)
				#plt.show()
				return abs(mid - actual_max), itereration
				# if delta is large (~50) such that the posterior begins to increase, decrease delta
				#else:
				#	if delta > 3:
				#		delta /= 2
				#	elif delta < 6:
				#		delta = 3
			else:
				gamma1 = mid # + 1?
				
				
		# When we are on the right side of the maximum
		else:
			m2.append(mi)
			mid2.append(mid)
			print("Mid1 values:", mid1)
			print("Mid2 values:", mid2)
			args['nj'] = mid

			# Get posterior probability (and if its sufficient, check the secant line on the respective side)
			posterior_prob = get_posterior_prob(1, args['n'], mid2, m2, delta, side=2) 
			print("probability: {}".format(posterior_prob))
			if posterior_prob > posterior_alpha:
				#if get_slope(**args) > mi:
				print("Maximum accuracy found at index {}".format(mid))
					# TODO: decide if delta is sufficiently small than we can stop the search
				y = get_cusp(args['n'], x=MID, seed=seed)
				actual_max = np.argmax(y)
				#print(np.argmax(y))
				#plt.plot(y)
				#plt.show()
				return abs(mid - actual_max), itereration
				# if delta is large (~50) such that the posterior begins to increase, decrease delta
				#else:
				#	if delta > 3:
				#		delta /= 2
				#	elif delta < 6:
				#		delta = 3

			else:
				gamma2 = mid # - 1?
		y = get_cusp(args['n'], x=MID, seed=seed)
		actual_max = np.argmax(y)
		if gamma2 - gamma1 < delta:
			return abs(actual_max - mid), itereration

		itereration += 1
		
		print("Actual Maximum:", )
		#	plt.plot(y)
		#	plt.show()
		#	exit(0)
		print("-"*20)
	

