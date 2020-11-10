# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Train and test for models with hidden layer at n dimensions
from binary_search_networks.get_data import get_data
from binary_search_networks.train import train
from binary_search_networks.test import test

def run_pipe(**args):
    X_train, X_test, y_train, y_test = get_data(**args)
    model = train(X_train, y_train, **args)
    test(X_test, y_test, model)
