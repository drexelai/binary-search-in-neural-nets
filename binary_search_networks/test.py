# Author: Isamu Isozaki, Yigit
# Date: 2020/11/10
# Purpose: Take X, y and model as input and returns test accuracy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

def test(X, y, model, **args):
    y_pred = model.predict(X)
    _, accuracy = model.evaluate(X, y)
    # Just for those who are new to ml,
    # True positive (TP) = the number of times model correctly outputted 1
    # False positive (FP) = the number of times model incorrectly outputted 1
    # True negative (TN) = the number of times model correctly outputted 0
    # False negative (FN) = the number of times model incorrectly outputted 0
    # precision = TP/ (TP + FP)
    # Recall = TP / (TP + FN)
    # If model only outputs 0 for 99% of the time and it only outputs 1 once when it's really confident, percison=1
    # If model only outputs 1, recall=1
    # ROC curve is Recall, true positive rate, for the y axis and false positive rate, FP/(FP+TN), for the x axis
    # The best classifier will have true positive rate = 1 and false positive rate = 0. So if the curve is going upper left from the start, it's a good classifier
    # Generates curve by changing the cutoff and getting FP rate and TP rate. The cutoffs are in the thresholds
    fpr, tpr, thresholds = roc_curve(y, y_pred) 
    # Area under the curve: best=1
    area_under_curve = auc(fpr, tpr)
    cr = classification_report(y, y_pred > args["threshold"])
    # Confusion matrix outputs 
    # TP FP
    # FN TN
    tn, fp, fn, tp = confusion_matrix(y, y_pred > args["threshold"]).ravel()
    precision = tp / (tp+fp)
    recall = tp/ (tp + fn)
    # F1 score is the harmonic mean of precision and recall
    F1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, area_under_curve, precision, recall, F1