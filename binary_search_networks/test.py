# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take X, y and model as input and returns test accuracy

def test(X, y, model):
    _, accuracy = model.evaluate(X, y)
    return accuracy