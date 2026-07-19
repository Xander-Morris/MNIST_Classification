import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml 
import matplotlib as mpl 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score 
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE_SEED = 666

def _setup_data():
    mnist = fetch_openml("mnist_784", version=1)
    keys = mnist.keys()

    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    some_digit = X.iloc[[0]]

    train_test_threshold = 60000 # training set is first 60K images, test set is last 10K images 
    X_train, X_test, y_train, y_test = X[:train_test_threshold], X[train_test_threshold:], y[:train_test_threshold], y[train_test_threshold:]

    return (X_train, y_train, X_test, y_test, some_digit)

def _plot_precision_recall_vs_threshold(y_train, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.savefig("precision_recall_vs_threshold.png")

def _cross_validate(some_digit, X_train, y_train, sgd_clf):
    skfolds = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE_SEED, shuffle=True)

    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    
    print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
    print(confusion_matrix(y_train, y_train_pred))
    print(precision_score(y_train, y_train_pred))
    print(recall_score(y_train, y_train_pred))
    print(f1_score(y_train, y_train_pred))

"""
    The Receiver operating characteristic (ROC) curve is a common tool used with binary classifiers. 
    It plots the true positive rate (TPR, also called recall) against the false positive rate (FPR).
"""
def _plot_roc_curves(y_train, y_scores):
    plt.clf()
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plt.plot(fpr, tpr, "b:", label="SGD")
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal through the middle

    # create the random forest classifier 
    forest_clf = RandomForestClassifier(random_state=RANDOM_STATE_SEED)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class 
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)
    plt.plot(fpr_forest, tpr_forest, linewidth=2, label="Random Forest")
    plt.legend(loc="lower right")
    plt.savefig("roc_curves.png")

def _run_svm_classifier(X_train, y_train, some_digit):
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    print(svm_clf.predict(some_digit))

def _run_ovr_classifier(X_train, y_train, some_digit):
    ovr_clf = OneVsRestClassifier(SVC())
    ovr_clf.fit(X_train, y_train)
    print(ovr_clf.predict(some_digit))
    print(len(ovr_clf.estimators_))

def _run_sgd_classifier(X_train, y_train, some_digit):
    sgd_clf = SGDClassifier(max_iter=2000, tol=1e-3)
    sgd_clf.fit(X_train, y_train)
    print(sgd_clf.predict(some_digit))
    print(sgd_clf.decision_function(some_digit))
    print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

    # analyze errors 
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=1)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    plt.clf()
    """
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    """
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

def _run_multioutput_classification(X_train, X_test, knn_clf):
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise 
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise 
    y_train_mod = X_train 
    y_test_mod = X_test 
    knn_clf.fit(X_train_mod, y_train_mod)
    """
        TODO: Make this actually work 
        clean_digit = knn_clf.predict([X_test_mod[some_index]])
        plot_digit(clean_digit)    
    """

def _run_multilabel_classification(X_train, y_train, some_digit):
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    print(knn_clf.predict(some_digit))
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

if __name__ == "__main__":
    (X_train, y_train, X_test, y_test, some_digit) = _setup_data()
    """
    _cross_validate(some_digit, X_train, y_train_5, sgd_clf)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    _plot_precision_recall_vs_threshold(y_train_5, y_scores)
    _plot_roc_curves(y_train_5, y_scores)
    """
    _run_svm_classifier(X_train, y_train, some_digit)
    _run_ovr_classifier(X_train, y_train, some_digit)
    _run_sgd_classifier(X_train, y_train, some_digit)
    _run_multilabel_classification(X_train, y_train, some_digit)