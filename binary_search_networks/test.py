# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take X, y and model as input and print accuracy

def test(X, y, model):
    _, accuracy = model.evaluate(X, y)
    print('Test Accuracy: %.2f' % (accuracy*100))
    return model