from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from pprint import pprint
import logging
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("ieor222_logger")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def rand_5(lst):
    return np.random.choice(lst, 5, replace=False)


def get_correct_wrong_pred_indices(true_labels, pred):
    correct_labels_indices = np.intersect1d(
        true_labels, pred)
    wrong_labels_indices = np.setdiff1d(pred, true_labels)

    return rand_5(correct_labels_indices), rand_5(wrong_labels_indices)


if __name__ == "__main__":
    features_filename = sys.argv[1]
    df = pd.read_csv(features_filename)
    df = df[1000:-1000]
    n_rows, n_cols = df.shape

    labels = df["labels"]
    df.drop('labels', axis=1, inplace=True)
    time = df["Time"]
    df.drop('Time', axis=1, inplace=True)

    eighty_percentile = int(0.9 * n_rows)

    training_data = df[:eighty_percentile]
    training_labels = labels[:eighty_percentile]
    val_data = df[eighty_percentile:]
    val_labels = labels[eighty_percentile:]
    val_times = time[eighty_percentile:]

    plt.plot(val_times, val_data["weighted_mid_price_0"] / 10000)
    plt.savefig("1.png")
    # quit()

    clf = RandomForestClassifier(
        n_estimators=50, max_features="auto", bootstrap=True)
    tuned_params = [{'criterion': ['gini', 'entropy'],
                     "min_samples_split": [1, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "max_depth": [3, 10, None]
                     }]

    def f(y1, y2):
        return f1_score(y1, y2, average="weighted", labels=[-1, 0, 1])

    grid_search = GridSearchCV(
        clf, param_grid=tuned_params, n_jobs=-1, verbose=2, cv=3, scoring=make_scorer(f))
    grid_search.fit(training_data, training_labels)
    report(grid_search.cv_results_)
    pred = grid_search.predict(val_data)
    print(classification_report(val_labels, pred,
                                labels=[-1, 0, 1], target_names=['down', 'stationary', 'up']))

    features_importance = zip(
        grid_search.best_estimator_.feature_importances_, training_data.columns.values)

    pprint(sorted(features_importance, key=lambda y: y[0], reverse=True)[:10])

    # label correctly predicted up labels
    up_indices_true = np.where(val_labels.values == 1)
    up_indices_pred = np.where(pred == 1)
    down_indices_true = np.where(val_labels.values == -1)
    down_indices_pred = np.where(pred == -1)
    stationary_indices_true = np.where(val_labels.values == 0)
    stationary_indices_pred = np.where(pred == 0)

    correct_up_indices, wrong_up_indices = get_correct_wrong_pred_indices(
        up_indices_true, up_indices_pred)

    correct_down_indices, wrong_down_indices = get_correct_wrong_pred_indices(
        down_indices_true, down_indices_pred)

    correct_stationary_indices, wrong_stationary_indices = get_correct_wrong_pred_indices(
        stationary_indices_true, stationary_indices_pred)

    plt.plot(val_times.take(correct_up_indices), val_data.take(correct_up_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="^", color="g", markersize=10)
    plt.plot(val_times.take(wrong_up_indices), val_data.take(wrong_up_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="v", color="r", markersize=10)
    plt.plot(val_times.take(correct_down_indices), val_data.take(correct_down_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="^", color="g", markersize=10)
    plt.plot(val_times.take(wrong_down_indices), val_data.take(wrong_down_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="v", color="r", markersize=10)
    plt.plot(val_times.take(correct_stationary_indices), val_data.take(correct_stationary_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="o", color="g", markersize=10)
    plt.plot(val_times.take(wrong_stationary_indices), val_data.take(wrong_stationary_indices)[
             "weighted_mid_price_0"] / 10000, linestyle='None', marker="o", color="r", markersize=10)

    plt.savefig("1.png")

    '''run_simple_trading_strategy'''
    inventory = 0
    balance = 0
    TRANSACT_QUANTITY = 10

    for i, p in enumerate(pred):
        if i == len(pred) - 1:
            if inventory < 0:
                balance -= -inventory * val_data.iloc[i]["PA0"]
                logger.info("BUYING %d @ %f" %
                            (-inventory, val_data.iloc[i]["PA0"] / 10000))
            else:
                balance += inventory * val_data.iloc[i]["PB0"]
                logger.info("SELLING %d @ %f" %
                            (inventory, val_data.iloc[i]["PB0"] / 10000))
            inventory = 0
        elif p == 1:
            if np.random.binomial(1, 0.5):
                inventory += TRANSACT_QUANTITY
                balance -= val_data.iloc[i]["PA0"] * TRANSACT_QUANTITY
                logger.info("BUYING %d @ %f" %
                            (TRANSACT_QUANTITY, val_data.iloc[i]["PA0"] / 10000))
        elif p == -1:
            if np.random.binomial(1, 0.5):
                inventory -= TRANSACT_QUANTITY
                balance += val_data.iloc[i]["PB0"] * TRANSACT_QUANTITY
                logger.info("SELLING %d @ %f" %
                            (TRANSACT_QUANTITY, val_data.iloc[i]["PB0"] / 10000))
        logger.info("BALANCE = %f" % (balance / 10000))
