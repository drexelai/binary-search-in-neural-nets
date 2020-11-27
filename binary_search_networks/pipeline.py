# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Train and test for models with hidden layer at n dimensions
from binary_search_networks.get_data import get_titanic_data, get_data_churn_rate
from binary_search_networks.train import train
from binary_search_networks.test import test
from binary_search_networks.util import parse_arguments

# Run pipe and returns train accuracy and test accuracy
def run_pipe(**args):
    data_func = get_data_churn_rate if args["use_churn_data"] else get_titanic_data
    X_train, X_test, y_train, y_test = data_func(**args)
    model, train_accuracy, val_accuracy = train(X_train, y_train, **args)
    test_accuracy, area_under_curve, precision, recall, F1 = test(X_test, y_test, model, **args)
    print("""Train Accuracy: {:.2f}%
    Validation Accuracy: {:.2f}% 
    Test Accuracy: {:.2f}% 
    Precision: {:.2f}
    Recall: {:.2f}
    F1 Score: {:.2f}"""
    .format(train_accuracy*100, val_accuracy*100, test_accuracy*100, precision, recall, F1))
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1

# Run pipe with parameter n and returns train and test accuracy
def run_model(n):
    args = parse_arguments([''])
    args['n'] = n
    train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1 = run_pipe(**args)
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1