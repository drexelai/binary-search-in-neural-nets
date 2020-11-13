# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Utility function folders
from binary_search_networks.binary_search_parser import binary_search_parser 

# Input: list of arguments
# Output: dictionary of arguments
# Parses the arguments
def parse_arguments(args):
    arg_parser = binary_search_parser()
    args, _ = arg_parser.parse_known_args(args)
    args = vars(args)
    return args