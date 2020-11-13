# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take n as input and run program

from binary_search_networks.pipeline import run_pipe
from binary_search_networks.util import parse_arguments
import sys
def main(args):
	args = parse_arguments(args)
	run_pipe(**args)
	
if __name__ == "__main__":
	main(sys.argv)
