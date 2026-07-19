from typing import List, Literal

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from sklearn.datasets import fetch_openml 
import matplotlib as mpl 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score 
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE_SEED = 666
DIRECTIONS = ("left", "right", "up", "down")

def _setup_data():
    mnist = fetch_openml("mnist_784", version=1)

    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    #some_digit = X.iloc[[0]]

    train_test_threshold = 60000 # training set is first 60K images, test set is last 10K images 
    X_train, X_test, y_train, y_test = X[:train_test_threshold], X[train_test_threshold:], y[:train_test_threshold], y[train_test_threshold:]

    return (X_train, y_train, X_test, y_test)

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

def _run_kneighbors_classifier(X_train, y_train, X_test, y_test, search_sample_size=15000):
    # No scaling: MNIST pixels share one scale (0-255). Scaling amplifies
    # near-zero-variance border pixels into noise and drags KNN accuracy down.
    # Use numpy arrays (not DataFrames) so each worker gets a cheaper pickle,
    # and so fancy integer indexing below selects rows, not columns.
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.to_numpy()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.to_numpy()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.to_numpy()

    # Grid search on small random sample only. KNN cost scales with
    # dataset size for both fit(cheap) and predict(expensive, brute-force
    # in 784-dim). Searching on the full 300K augmented set makes each of
    # the 5-fold x N-param fits crawl. Best k/weights found on 15K holds
    # up on the full set, so search cheap, refit big.
    rng = np.random.RandomState(RANDOM_STATE_SEED)
    sample_idx = rng.choice(len(X_train), size=min(search_sample_size, len(X_train)), replace=False)
    X_search, y_search = X_train[sample_idx], y_train[sample_idx]

    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5],   # trimmed from [3,4,5,6]
        'weights': ['uniform', 'distance'],
    }

    # cv=3 instead of 5, n_jobs capped to avoid Windows named-pipe
    # exhaustion when shipping data to workers
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=4)
    grid_search.fit(X_search, y_search)

    print(f"Best Hyperparameters (from {search_sample_size}-sample search): {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    return grid_search

def _shift_images(images_3d, dir):
    first_val = second_val = 0
    if dir in ("down", "up"):
        first_val = 1 if dir == "down" else -1
    else:
        second_val = 1 if dir == "right" else -1
    # shift=0 on axis 0 = batch dim untouched, every image shifted same way
    return ndi.shift(images_3d, (0, first_val, second_val), mode="constant", cval=0)

if __name__ == "__main__":
    (X_train, y_train, X_test, y_test) = _setup_data()
    """
    _cross_validate(some_digit, X_train, y_train_5, sgd_clf)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    _plot_precision_recall_vs_threshold(y_train_5, y_scores)
    _plot_roc_curves(y_train_5, y_scores)
    """
    #_run_svm_classifier(X_train, y_train, some_digit)
    #_run_ovr_classifier(X_train, y_train, some_digit)
    #_run_sgd_classifier(X_train, y_train, some_digit)
    #_run_multilabel_classification(X_train, y_train, some_digit)
    grid_search = _run_kneighbors_classifier(X_train, y_train, X_test, y_test)
    X_train_np = X_train.to_numpy()
    X_train_3d = X_train_np.reshape(-1, 28, 28)
    augmented_X = [X_train]
    augmented_y = [y_train]
    for direction in DIRECTIONS:
        shifted_3d = _shift_images(X_train_3d, direction)
        augmented_X.append(shifted_3d.reshape(-1, 784))
        augmented_y.append(y_train)
    X_train = np.vstack(augmented_X)
    y_train = np.concatenate(augmented_y)

    # Refit winning params once on the full augmented training set.
    # NOT grid_search.fit() again here -- that would rerun the entire
    # cv=3 x 4-combo search on the 300K augmented rows, the exact
    # slowdown this two-stage split exists to avoid.
    best_model = KNeighborsClassifier(**grid_search.best_params_)
    best_model.fit(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")