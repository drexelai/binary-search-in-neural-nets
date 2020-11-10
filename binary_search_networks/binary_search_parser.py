# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: construct parser with input n
import argparse
import numpy as np

def arg_parser():
	"""
	Create an empty argparse.ArgumentParser.
	"""
	import argparse
	return argparse.ArgumentParser(
		description='Setting up binary search in neural nets project',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
		allow_abbrev=False)

def binary_search_parser():
	parser = arg_parser()
	parser.add_argument('--batch_size', default=10, type=int, help='The batch size to train for.')
	parser.add_argument('--epoch', default=150, type=int, help='The number of epochs to train for.')
	parser.add_argument('--n', default=5, type=int, help='The dimension of the hidden layer.')
	parser.add_argument('--test_size', default=0.2, type=float, help='The proportion of data used for tests. Between 0 and 1.')
	parser.add_argument('--validation_split', default=0.2, type=float, help='Validation split proportion while training. Between 0 and 1.')
	parser.add_argument('--verbose', default=1, type=int, help='Verbose for training. Either 0, 1, or 2.')
	return parser