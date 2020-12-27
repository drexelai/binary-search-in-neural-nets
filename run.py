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
	binary_search(**args)

	#run_pipe(**args)



	
if __name__ == "__main__":
	main(sys.argv)
