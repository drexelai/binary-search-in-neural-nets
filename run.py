# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take n as input and run program

from binary_search_networks.pipeline import run_pipe
from binary_search_networks.util import parse_arguments

from binary_search_networks.search import binary_search, get_output_space, plot_output_space

import sys
def main(args):
	args = parse_arguments(args)
	# a, b, accuracys = get_output_space(**args)
	# plot_output_space(a, b, accuracys)

	differences = []
	itererations = []

	for i in range(100):
		print(i)
		difference, itereration = binary_search(**args)
		print(difference, itereration)
		differences.append(difference)
		itererations.append(itereration)

	print(differences)
	print(itererations)
	print(sum(differences) / len(differences))
	print(sum(itererations) / len(itererations))
	#run_pipe(**args)


if __name__ == "__main__":
	main(sys.argv)
