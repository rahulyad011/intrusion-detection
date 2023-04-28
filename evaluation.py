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

import numpy as np

import matplotlib.pyplot as plt

# local imports
from utility import save_eval_in_csv

def plot_svm_prediction(y_pred, y_test, model_type):
    plt.figure(figsize=(20,8))
    plt.plot(y_pred[300:500], label="prediction", linewidth=2.0,color='blue')
    plt.plot(y_test[300:500].values, label="real_values", linewidth=2.0,color='lightcoral')
    plt.legend(loc="best")
    plt.ylim((-1,2))
    plt.title(model_type+"Binary Classification")
    plt.savefig('Plots/'+model_type+'_pred_bin.png')
    # plt.show()

def evaluate(model, dataset_used, model_type, y_test, y_pred, plot):
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    clf = model
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("confusion matrix:")
    print(cm)
    # Compute TP, TN, FP, FN for each class
    tp = np.diagonal(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Compute FPR and FNR for each class
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print("False Positive Rate - ", fpr)
    print("False Negative Rate - ", fnr)
    print()
    c_accuracy = accuracy_score(y_test, y_pred)*100
    print("accuracy Score - ",c_accuracy) 
    print("Recall Score - ",recall_score(y_test,y_pred))
    c_f1_score = f1_score(y_test,y_pred)
    print("F1 Score - ",c_f1_score)
    print("Precision Score - ",precision_score(y_test,y_pred))

    save_eval_in_csv(dataset_used, model_type, c_accuracy, c_f1_score, fnr[1], fpr[0])

    if plot == True:
        plot_confusion_matrix(model, y_test, y_pred, cm, model_type)
        if model_type == "SVM":
            plot_svm_prediction(y_pred, y_test, model_type)

def plot_confusion_matrix(model, y_test, y_pred, cm, model_type):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=model.classes_)
    disp.plot()
    plt.title(model_type+" Binary Classification")
    plt.savefig('Plots/'+model_type+'_confusion_matrix.png')
    # plt.show()