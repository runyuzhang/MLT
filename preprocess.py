import pandas as pd
import numpy as np
import sys
from multiprocessing import Pool
from functools import partial
import multiprocessing
import logging

CPU_COUNT = multiprocessing.cpu_count()
logger = logging.getLogger("ieor222_logger")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')

# setting display option
pd.set_option('display.max_rows', 100)

# file names
message_filename = sys.argv[1]
orderbook_filename = sys.argv[2]
output_filename = sys.argv[3]


PA_names = ["PA%d" % i for i in xrange(10)]
PB_names = ["PB%d" % i for i in xrange(10)]
VA_names = ["VA%d" % i for i in xrange(10)]
VB_names = ["VB%d" % i for i in xrange(10)]

intensity_feature_names = ["buy_execute_intensity", "sell_execute_intensity", "buy_limit_intensity",
                           "sell_limit_intensity", "buy_cancel_intensity", "sell_cancel_intensity",
                           "buy_delete_intensity", "sell_delete_intensity"]

EXECUTE_ORDER = 4
LIMIT_ORDER = 1
CANCELLATION_ORDER = 2
DELETE_ORDER = 3

BUY_DIRECTION = 1
SELL_DIRECTION = -1


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
    return "d%s" % f


def DD(x, y):
    return "d%sd%s" % (x, y)


def weighted_mid_price(pa, va, pb, vb, sentiment=True):
    if sentiment:
        return (pa * va + pb * vb) / (va + vb)
    else:
        return (pa * vb + pb * va) / (va + vb)


def add_col(df, col_name, col):
    df.loc[:, col_name] = col


def get_value_first_row(df, name):
    return df.iloc[0][name]


def calculate_deltas(df, features):
    ret = df[1:]
    df_1_behind = df[:-1]
    for f in features:
        add_col(ret, "d%s" % f, ret[f].values - df_1_behind[f].values)
    return ret

df_msg = pd.read_csv(message_filename, sep=',')
df_lob = pd.read_csv(orderbook_filename, sep=',')

df = pd.concat([df_msg, df_lob], axis=1)

training_size_per_label = 5000
val_size = 1000


def calc_delta(df, col_name, window):
    def d(val, time):
        try:
            return val - df[col_name][(df["Time"] > time - window) & (df["Time"] < time)].iloc[0]
        except:
            return 0
    return df.apply(lambda x: d(x[col_name], x["Time"]), axis=1)


def calc_average_intensity(df, direction, order_type, window):
    def count_orders_after_time(time):
        return len(df[(df["Type"] == order_type) & (df["Direction"] == direction) & (df["Time"] > time - window) & (df["Time"] < time)])
    return df.apply(lambda x: count_orders_after_time(x["Time"]), axis=1)


def parallel_apply(df, func, num_partitions=30):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(CPU_COUNT)

    df_ret = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df_ret


def compute_deltas_on_features(df, features, window=1):
    for f in features:
        logger.info("computing delta on feature %s" % f)
        add_col(df, "D%s" % f, parallel_apply(
            df, partial(calc_delta, col_name=f, window=window)))
    return df


def compute_average_intensities(df, window=1):
    '''Executed order'''
    logger.info("computing execute order intensity on the buy direction")
    add_col(df, "buy_execute_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=BUY_DIRECTION, order_type=EXECUTE_ORDER, window=window)))

    logger.info("computing execute order intensity on the sell direction")
    add_col(df, "sell_execute_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=SELL_DIRECTION, order_type=EXECUTE_ORDER, window=window)))

    '''Limit order'''
    logger.info("computing limit order intensity on the buy direction")
    add_col(df, "buy_limit_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=BUY_DIRECTION, order_type=LIMIT_ORDER, window=window)))

    logger.info("computing limit order intensity on the sell direction")
    add_col(df, "sell_limit_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=SELL_DIRECTION, order_type=LIMIT_ORDER, window=window)))

    '''Cancellation order'''
    logger.info("computing cancellation order intensity on the buy direction")
    add_col(df, "buy_cancel_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=BUY_DIRECTION, order_type=CANCELLATION_ORDER, window=window)))

    logger.info("computing cancellation order intensity on the sell direction")
    add_col(df, "sell_cancel_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=SELL_DIRECTION, order_type=CANCELLATION_ORDER, window=window)))

    '''Deletion order'''
    logger.info("computing deletion order intensity on the buy direction")
    add_col(df, "buy_delete_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=BUY_DIRECTION, order_type=DELETE_ORDER, window=window)))

    logger.info("computing deletion  order intensity on the sell direction")
    add_col(df, "sell_delete_intensity", parallel_apply(df, partial(
        calc_average_intensity, direction=SELL_DIRECTION, order_type=DELETE_ORDER, window=window)))

    return df
if __name__ == "__main__":
    df = compute_average_intensities(df, window=1)

    df = compute_deltas_on_features(df, ["Time"])

    df = compute_deltas_on_features(df, basic_feature_set)

    df = compute_deltas_on_features(df, intensity_feature_names)

    df.to_csv(output_filename)
