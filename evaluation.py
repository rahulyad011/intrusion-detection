# evaluation metrices
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import det_curve

from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.metrics import classification_report # for generating a classification report of model

import matplotlib.pyplot as plt

def plot_svm_prediction(y_pred, y_test):
    plt.figure(figsize=(20,8))
    plt.plot(y_pred[300:500], label="prediction", linewidth=2.0,color='blue')
    plt.plot(y_test[300:500].values, label="real_values", linewidth=2.0,color='lightcoral')
    plt.legend(loc="best")
    plt.ylim((-1,2))
    plt.title("Linear SVM Binary Classification")
    plt.savefig('Plots/lsvm_real_pred_bin.png')
    plt.show()

def evaluate():
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    fpr, fnr, thresholds = det_curve(y_test, y_pred)
    print("False Positive Rate - ", fpr)
    print("False Negative Rate - ", fnr)
    print("Thresholds - ", thresholds)

def plot_confusion_matrix(model, y_test, y_pred):
    clf = model
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=clf.classes_)
    disp.plot()

    plt.show()