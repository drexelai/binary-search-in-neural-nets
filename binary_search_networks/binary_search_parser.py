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
	parser.add_argument('--delta', default=50, type=int, help='The spacing between ni and nj when calculating the gradient of accuracy.')
	parser.add_argument('--early_stopping_patience', default=3, type=int, help='Number of epochs with no improvement after which training will be stopped if early stopping is enabled.')
	parser.add_argument('--epoch', default=150, type=int, help='The number of epochs to train for.')
	parser.add_argument('--exp_data_save', default="exp_data.csv", type=str, help='The location the experiment data is saved')
	parser.add_argument('--fig_save_dir', default="figures", type=str, help='The location where plots will be saved')
	parser.add_argument('--fig_save_name', default="figure", type=str, help='The name of the figure that is saved')
	parser.add_argument('--model_save_dir', default="models", type=str, help='The location where models will be saved')

	return parser