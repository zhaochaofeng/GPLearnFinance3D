import numpy as np
import genetic
import qlib
from qlib.data import D
from utils import make_XY
import os
from add_ts_function import ts_std_10, ts_max_10,  ts_mean_10
from add_ts_function import dynamic_ts_std, dynamic_ts_mean,dynamic_ts_max

qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', kernels=1)

config = D.instruments(market='csi300')
instruments = D.list_instruments(config, start_time='2017-01-01', end_time='2018-01-01', as_list=True)

def get_df(start_time, end_time):
    fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']
    df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
    df['return'] = df['$close'].shift(-2) / df['$close'].shift(-1) - 1

    total_df = df.reset_index()
    total_df.columns = ["股票代码","交易日期","开盘价","最高价","最低价","收盘价","成交量","成交额","收益率"]
    return total_df

train_df = get_df('2017-01-01', '2018-01-01')
eval_df = get_df('2018-01-01', '2019-01-01')


different_axis = ("交易日期","股票代码", "收益率",)

train_X, train_Y, feature_names = make_XY(train_df, *different_axis)
eval_X, eval_Y, _ = make_XY(eval_df, *different_axis)


X = np.concatenate([train_X, eval_X], axis=0)
Y = np.concatenate([train_Y, eval_Y], axis=0)
X_feature_names = feature_names
sample_weight = []
sample_weight.extend([1]*train_X.shape[0])
sample_weight.extend([0]*eval_X.shape[0])
sample_weight = np.array(sample_weight)


function_set_sample = ['common_add', 'common_sub', 'common_mul', 'common_div',
                       'common_log', 'common_sqrt', 'common_abs', 'common_inv', 'common_max', 'common_min', 'common_tan',] #'std_10'

# my_function = [ts_std_10, ts_max_10,  ts_mean_10,]
my_function = [dynamic_ts_std, dynamic_ts_mean,dynamic_ts_max]
function_set = function_set_sample + my_function


gp_sample = genetic.SymbolicTransformer(generations=2,
                                        population_size=200,
                                        tournament_size=10,
                                        init_depth=(1, 3),
                                        hall_of_fame=100,
                                        n_components=10,
                                        function_set=function_set,
                                        metric="pearson_3d",
                                        const_range=(-1, 1),
                                        p_crossover=0.4,
                                        p_hoist_mutation=0.001,
                                        p_subtree_mutation=0.01,
                                        p_point_mutation=0.01,
                                        p_point_replace=0.4,
                                        parsimony_coefficient="auto",
                                        feature_names=X_feature_names,
                                        max_samples=1, verbose=1,
                                        random_state=0, n_jobs=os.cpu_count()-2)

gp_sample.fit_3D(X, Y, feature_names, sample_weight=sample_weight,
                 standard_expression="TRA ((pearson_3d>=0.02) and (spearman_3d >=0.002)) OOB (pearson_3d>0.0002)",
                 need_parallel=True)

result = gp_sample.show_program(X,Y,
                                sample_weight=sample_weight,
                                feature_names=X_feature_names,
                                baseIC=False,
                                show_tracing=(True,"./show_tracing.csv"))
result.to_csv("./result_only10.csv")
