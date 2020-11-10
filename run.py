# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take n as input and run program

from binary_search_networks.binary_search_parser import binary_search_parser 
from binary_search_networks.pipeline import run_pipe
import sys
def main(args):
	arg_parser = binary_search_parser()
	args, _ = arg_parser.parse_known_args(args)#the openai way
	#convert args to dictionary
	args = vars(args)

	run_pipe(**args)
if __name__ == "__main__":
	main(sys.argv)
