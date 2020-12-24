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
    model, train_accuracy, val_accuracy = train(X_train, y_train, **args)

    # Plot training accuracy
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig('results/accuracy', dpi=300)
    plt.close()

    test_accuracy, area_under_curve, precision, recall, F1 = test(X_test, y_test, model, **args)
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model

# Run pipe with parameter n and returns train and test accuracy
def run_model(n):
    args = parse_arguments([''])
    args['n'] = n
    train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1 = run_pipe(**args)

    # Plot training accuracy
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig('results/accuracy', dpi=300)
    plt.close()
    
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model