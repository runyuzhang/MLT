import sys
import pandas as pd
import logging
import numpy as np
import yaml
from functools import partial

logger = logging.getLogger("ieor222_logger")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


pd.set_option('display.max_rows', 100)

PA_names = ["PA%d" % i for i in xrange(10)]
PB_names = ["PB%d" % i for i in xrange(10)]
VA_names = ["VA%d" % i for i in xrange(10)]
VB_names = ["VB%d" % i for i in xrange(10)]

intensity_feature_names = ["buy_execute_intensity", "sell_execute_intensity", "buy_limit_intensity",
                           "sell_limit_intensity", "buy_cancel_intensity", "sell_cancel_intensity",
                           "buy_delete_intensity", "sell_delete_intensity"]

basic_feature_set = PA_names + PB_names + VA_names + VB_names


def PA(i):
    return PA_names[i]


def PB(i):
    return PB_names[i]


def VA(i):
    return VA_names[i]


def VB(i):
    return VB_names[i]


def D(f):
    return "D%s" % f


def DD(x, y):
    return "D%sD%s" % (x, y)


def weighted_mid_price(pa, va, pb, vb, sentiment=True):
    if sentiment:
        return (pa * va + pb * vb) / (va + vb)
    else:
        return (pa * vb + pb * va) / (va + vb)


def add_col(df, col_name, col):
    df.loc[:, col_name] = col


def generate_features(df, mid_price_func):
    '''basic feature set'''
    df = df[df["DTime"] != 0]
    ret = df[basic_feature_set]
    add_col(ret, "Time", df["Time"])
    '''use log values'''
    # ret = ret.applymap(np.log)

    '''time sensitive features'''

    # spread
    for i in xrange(0, 10):
        add_col(ret, "spread_%d" % i, df[PA(i)] - df[PB(i)])

    # weighted_mid_price
    for i in xrange(0, 10):
        add_col(ret, "weighted_mid_price_%d" % i, mid_price_func(
            df[PA(i)], df[VA(i)], df[PB(i)], df[VB(i)]))

    # price difference
    add_col(ret, "price_diff_ask_max", df[PA(9)] - df[PA(0)])
    add_col(ret, "price_diff_bid_max", df[PB(0)] - df[PB(9)])
    for i in xrange(0, 9):
        add_col(ret, "price_diff_ask_%d" %
                i, np.abs(df[PA(i + 1)] - df[PA(i)]))
        add_col(ret, "price_diff_ask_%d" %
                i, np.abs(df[PA(i + 1)] - df[PA(i)]))

    # mean price
    add_col(ret, "mean_price_ask_%d" % i, df[PA_names].mean(axis=1))
    add_col(ret, "mean_price_buy_%d" % i, df[PB_names].mean(axis=1))
    add_col(ret, "mean_volume_ask_%d" % i, df[VA_names].mean(axis=1))
    add_col(ret, "mean_volume_buy_%d" % i, df[VB_names].mean(axis=1))

    # accumulated price difference
    sum_price_ask = df[PA_names].sum(axis=1)
    sum_price_bid = df[PB_names].sum(axis=1)
    sum_vol_ask = df[VA_names].sum(axis=1)
    sum_vol_bid = df[VB_names].sum(axis=1)
    add_col(ret, "accumulated_price_diff", sum_price_ask - sum_price_bid)
    add_col(ret, "accumulated_vol_diff", sum_vol_ask - sum_vol_bid)

    '''time sensitive'''

    # price and volume derivative
    for f in basic_feature_set:
        x = df[D(f)] / df[D("Time")]
        add_col(ret, DD(f, "Time"), x)
    for f in intensity_feature_names:
        x = df[D(f)] / df[D("Time")]
        add_col(ret, DD(f, "Time"), x)

    # # dummy variables
    ret.loc[:, "Type"] = df["Type"]
    ret.loc[:, "Direction"] = df["Direction"]

    return ret


def generate_labels(df):
    weighted_mid_prices = df["weighted_mid_price_0"]
    weighted_mid_prices_1_lookahead = weighted_mid_prices[1:]

    labels = np.vectorize(lambda x: 0 if x == 0 else 1 if x >
                          0 else -1)(weighted_mid_prices_1_lookahead.values - weighted_mid_prices[:-1].values)
    df = df.iloc[1:]
    add_col(df, "labels", labels)
    return df


if __name__ == "__main__":
    input_filename = sys.argv[1]
    features_filename = sys.argv[2]

    configs = yaml.load(open("config.yaml"))

    df = pd.read_csv(input_filename)

    df = df.iloc[::10]

    mid_price_func = partial(weighted_mid_price, sentiment=configs[
                             'use_sentiment_based_weighted_mid_price'])

    df = generate_features(df, mid_price_func=mid_price_func)
    df = generate_labels(df)

    df.to_csv(features_filename)
