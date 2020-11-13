# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Train and test for models with hidden layer at n dimensions
from binary_search_networks.get_data import get_data
from binary_search_networks.train import train
from binary_search_networks.test import test
from binary_search_networks.util import parse_arguments

# Run pipe and returns train accuracy and test accuracy
def run_pipe(**args):
    X_train, X_test, y_train, y_test = get_data(**args)
    model, train_accuracy = train(X_train, y_train, **args)
    test_accuracy = test(X_test, y_test, model)
    print("Train Accuracy: {:.2f}%\nTest Accuracy: {:.2f}%".format(train_accuracy*100, test_accuracy*100))
    return train_accuracy, test_accuracy

# Run pipe with parameter n and returns train and test accuracy
def run_model(n):
    args = parse_arguments([''])
    args['n'] = n
    train_accuracy, test_accuracy = run_pipe(**args)
    return train_accuracy, test_accuracy