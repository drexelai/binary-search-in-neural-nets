# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take n as input and run program

from binary_search_networks.pipeline import run_pipe
from binary_search_networks.util import parse_arguments

from binary_search_networks.search import binary_search, get_output_space, plot_output_space

import sys
def main(args):
	args = parse_arguments(args)
	a, b, train_accuracies, test_accuracies = get_output_space(**args)
	args['a'] = a
	args['b'] = b
	args['train_accuracies'] = train_accuracies
	args['test_accuracies'] = test_accuracies
	plot_output_space(**args)
	exit(0)

	# differences = []
	# itererations = []

	# for i in range(1000):
	# 	print(i)
	# 	difference, itereration = binary_search(**args)
	# 	print(difference, itereration)
	# 	differences.append(difference)
	# 	itererations.append(itereration)

	# print(differences)
	# print(itererations)
	# print(sum(differences) / len(differences))
	# print(sum(itererations) / len(itererations))
	#run_pipe(**args)

	# binary_search(**args)

	# run_pipe(**args)


if __name__ == "__main__":
	main(sys.argv)
