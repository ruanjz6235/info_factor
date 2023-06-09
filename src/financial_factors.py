import pandas as pd
import numpy as np
from itertools import product


def pos(df):
    df1 = pd.DataFrame([range(len(df.columns))] * len(df),
                       columns=df.columns[::-1], index=df.index)
    df1 = df1.where(~df.isna(), np.nan)
    # cov = (df.T - df.mean(axis=1)) * (df1.T - df1.mean(axis=1)).mean().T
    exy = df.mul(df1, axis=1).mean(axis=1)
    exey = df.mean(axis=1) * df1.mean(axis=1)
    df_std2 = df.std(ddof=0, axis=1)
    df1_std2 = df1.std(ddof=0, axis=1)
    return (exy - exey) / (df_std2 * df1_std2)


cols = ['stock_name', 'stock_code', 'ths_indsutry_name_l1', 'ths_indsutry_name_l2', 'ths_indsutry_code_l2',
        'ths_indsutry_name_l3', 'ths_indsutry_code_l3', 'enddate', 'pub_date', 'roe_ttm', 'net_profit_ratio_ttm',
        'profit_ratio_ttm', 'gross_profit_ratio', 'equity_belong_mgs_increase_ratio',
        'net_profit_belong_mgs_increase_ratio_ttm', 'operating_income_increase_ratio_ttm',
        'operating_profit_increase_ratio_ttm', 'accounts_receivable_turnover', 'inventory_turnover',
        'accounts_payable_turnover', 'total_asset_turnover', 'operating_income_cash_ratio', 'fcf_equity_ratio',
        'net_profit_cash_ratio', 'debt_contain_interest_ratio', 'quick_ratio', 'liquidity_ratio', 'inventory_ratio',
        'accounts_receivable_ratio', 'goodwill_ratio']
basic_cols = ['stock_name', 'stock_code', 'ths_indsutry_name_l1', 'ths_indsutry_name_l2', 'ths_indsutry_code_l2',
              'ths_indsutry_name_l3', 'ths_indsutry_code_l3', 'enddate', 'pub_date']
bb_cols = ['stock_code', 'enddate']
# todo: roe_ttm: 净资产收益率
# todo: net_profit_ratio_ttm: 净利率
# todo: profit_ratio_ttm: 利润率
# todo: gross_profit_ratio: 毛利率
# todo: equity_belong_mgs_increase_ratio: 归母所有者权益增长率，净资产同比
# todo: net_profit_belong_mgs_increase_ratio_ttm: 归母净利润增长率，净利润同比
# todo: operating_income_increase_ratio_ttm: 营业收入增长率
# todo: operating_profit_increase_ratio_ttm: 营业利润增长率
# todo: accounts_receivable_turnover: 应收账款周转率
# todo: inventory_turnover: 存货周转率
# todo: accounts_payable_turnover: 应付账款周转率
# todo: total_asset_turnover: 总资产周转率
# todo: operating_income_cash_ratio: 营业收入现金比率
# todo: fcf_equity_ratio: 自由现金流股权比率
# todo: net_profit_cash_ratio: 净利润现金比率
# todo: debt_contain_interest_ratio: 债务包含利息比率
# todo: quick_ratio: 速动比率
# todo: liquidity_ratio: 流动性比率
# todo: inventory_ratio: 库存比率
# todo: accounts_receivable_ratio: 应收账款比率
# todo: goodwill_ratio: 商誉比率
index_cols = cols[len(basic_cols):]
financial_data = pd.read_csv(r'A股十年财务数据.csv')
financial_data['enddate'] = pd.to_datetime(financial_data['enddate'].apply(
    lambda x: str(int(x)) if not np.isnan(x) else np.nan))
cols1 = bb_cols + index_cols
# 1. 排序
fi_data = financial_data[cols1]
fi_data = fi_data.sort_values(['stock_code', 'enddate'])
# fi_data.to_csv('A股十年财务数据_sort.csv', index=False)
# 2. 原始因子——最新值、均值、方差、单调性
print('0------------raw_data')
for i in range(1, 5):
    print(i)
    fi_data[f'enddate_lag{i}'] = fi_data['enddate'] - i * pd.offsets.YearEnd()
    cols2 = [c+f'_lag{i}' for c in cols1[1:]]
    fi_data0 = fi_data[cols1].rename(columns=dict(zip(cols1[1:], cols2)))
    fi_data = fi_data.merge(fi_data0, on=['stock_code', f'enddate_lag{i}'], how='left')

for i, index_col in enumerate(index_cols):
    # if i > 0:
    #     break
    print(index_col)
    sub_col = [index_col, index_col+'_lag1', index_col+'_lag2', index_col+'_lag3', index_col+'_lag4']
    sub_col_new = [index_col+'_mean', index_col+'_std', index_col+'_pos']
    fi_data[index_col+'_mean'] = fi_data[sub_col].mean(axis=1)
    fi_data[index_col+'_std'] = fi_data[sub_col].std(ddof=0, axis=1)
    fi_data[index_col+'_pos'] = pos(fi_data[sub_col])
    # fi_data[bb_cols + sub_col + sub_col_new].to_csv(f'0-{index_col}.csv', index=False)

    fi_data_index_count = fi_data[sub_col].count(axis=1)
    fi_data.loc[fi_data_index_count <= 1, index_col + '_std'] = 1e10
    fi_data[index_col + '_pos'] = fi_data[index_col + '_pos'].fillna(-1e10)


# %%
# 3. P1
print('1------------P1')
index_cols_mean = [x+'_mean' for x in index_cols]
index_cols_std = [x+'_std' for x in index_cols]
index_cols_pos = [x+'_pos' for x in index_cols]
fi_data1 = fi_data[bb_cols + index_cols + index_cols_mean + index_cols_std + index_cols_pos]
# 1. 总体逻辑是，做完数据预处理后，按照从小到大排序，数值越大越好，数值最小者为0，数据最大者为1，而后五分位判断时直接乘以5即可。
# 2. 应收账款周转率 / 存货周转率, 空值第一名; 应付账款周转率 / 付息债务比例 / 应收账款比例 / 存货比例 / 商誉比例, 虽然空值最后一名,
# 但是均越小越好, 所以这几类空值均设为极大值1e10, 后续后面几个指标再加上负号即可。
# 标准差只需要修改应收账款周转率 / 存货周转率为-1e10即可。单调性应与原始值和均值同等处理。
# 3. 对于其他指标，空值为最后一名，所以给空值设极小值-1e10。
# 4. 对于应付账款周转率 / 付息债务比例 / 应收账款比例 / 存货比例 / 商誉比例，数值越小越好，故要在其前加上负号。均值和单调性都是如此。
# 5. 对于应付账款周转率 / 付息债务比例 / 应收账款比例 / 存货比例，负值（加上负号后即为正值）也都是最后一名，此时的最后一名是-1e10。
c0 = ['accounts_receivable_turnover', 'inventory_turnover', 'accounts_payable_turnover', 'debt_contain_interest_ratio',
      'accounts_receivable_ratio', 'inventory_ratio', 'goodwill_ratio']
# 2
for index_col in c0:
    fi_data1[index_col] = fi_data1[index_col].fillna(1e10)
    fi_data1[index_col + '_mean'] = fi_data1[index_col + '_mean'].fillna(1e10)
    fi_data1.loc[fi_data1[index_col + '_pos'] == - 1e10, index_col + '_pos'] = 1e10
    if index_col in c0[:2]:
        fi_data1.loc[fi_data1[index_col + '_std'] == 1e10, index_col + '_std'] = - 1e10
# 3
for index_col in list(set(index_cols) - set(c0)):
    fi_data1[index_col] = fi_data1[index_col].fillna(-1e10)
    fi_data1[index_col + '_mean'] = fi_data1[index_col + '_mean'].fillna(-1e10)
# 4
for index_col in c0[2:]:
    fi_data1[index_col] = - fi_data1[index_col]
    fi_data1[index_col + '_mean'] = - fi_data1[index_col + '_mean']
    fi_data1[index_col + '_pos'] = - fi_data1[index_col + '_pos']
# 5
for index_col in c0[2:-1]:
    fi_data1.loc[fi_data1[index_col] > 0, index_col] = -1e10
    fi_data1.loc[fi_data1[index_col + '_mean'] > 0, index_col + '_mean'] = -1e10
merge_cols = ['stock_name', 'stock_code', 'enddate', 'ths_indsutry_name_l2']
fi_data1 = financial_data[merge_cols].merge(fi_data1, on=bb_cols, how='outer')
# 1
fi_data2 = fi_data1.copy()
for i, index_col in enumerate(index_cols):
    # if i > 0:
    #     break
    print(index_col)
    sub_col0 = [index_col, index_col + '_mean', index_col + '_std', index_col + '_pos']
    # 标准差处理
    fi_data2[index_col + '_std'] = - fi_data2[index_col + '_std']
    # 统一排名
    sub_rank_col = [x+'_rank' for x in sub_col0]
    sub_percent_col = [x+'_percent' for x in sub_col0]
    fi_data2[sub_rank_col] = fi_data2.groupby(merge_cols[2:])[sub_col0].rank(method='max')
    df_max = fi_data2.groupby(merge_cols[2:])[sub_rank_col].max()
    df_max.columns = [x+'_max' for x in df_max.columns]
    fi_data2 = fi_data2.merge(df_max.reset_index(), on=merge_cols[2:], how='left')
    fi_data2[sub_percent_col] = fi_data2[sub_rank_col].values / fi_data2[df_max.columns].values
    # fi_data2[merge_cols + sub_col0 + sub_rank_col + list(df_max.columns) + sub_percent_col].sort_values(
    #     ['enddate', 'ths_indsutry_name_l2', f'{index_col}_percent']).to_csv(f'1-{index_col}_percent.csv', index=False)


# %%
# 4. P2
print('2------------P2')
index_cols_sub = ['', '_mean', '_std', '_pos']
index_cols_percent = [index_col + index_col_sub + '_percent'
                      for index_col, index_col_sub in product(index_cols, index_cols_sub)]
cols_percent = merge_cols + index_cols_percent
per_mean = [index_col + '_percent_mean' for index_col in index_cols]
factor_dict = {'profit': per_mean[:4], 'growth': per_mean[4:8], 'operate': per_mean[8:12],
               'cashflow': per_mean[12:15], 'debt': per_mean[15:18], 'quality': per_mean[18:21]}
fi_data3 = fi_data2[cols_percent].set_index(merge_cols)
# -----------------------------------------------------
# 参数设置
# w1
w1 = [[0.3, 0.3, 0.3, 0]] * len(index_cols)
# w1 = [[0.25, 0.25, 0.25, 0.25]] * len(index_cols)
# w1 = [[0.5, 0.25, 0.25, 0]] * len(index_cols)
# w1 = [[0.5, 1/6, 1/6, 1/6]] * len(index_cols)

# w2
w2 = []
w2_old = np.ones(21)
n = 0
for i in range(6):
    num = len(list(factor_dict.values())[i])
    sub_w2 = w2_old[n: n+num]
    w2.append(sub_w2 / sub_w2.sum())
    n += num
# -----------------------------------------------------
for i, index_col in enumerate(index_cols):
    print(index_col)
    sub_w1 = w1[i]
    sub_percent_col = [index_col + index_col_sub + '_percent' for index_col_sub in index_cols_sub]
    sub_fi_data3 = fi_data3[sub_percent_col]
    sub_fi_data3_df = pd.DataFrame([sub_w1] * len(sub_fi_data3),
                                   index=sub_fi_data3.index,
                                   columns=sub_fi_data3.columns)
    fi_data3[index_col + '_percent_mean'] = (sub_fi_data3 * sub_fi_data3_df).sum(axis=1)
    # fi_data3[sub_percent_col + [index_col + '_percent_mean']].to_csv(f'2-1-{index_col}_percent_mean.csv')
fi_data4 = fi_data3[per_mean]
fi_data4_df = pd.DataFrame([np.hstack(w2)] * len(fi_data4), index=fi_data4.index, columns=fi_data4.columns)
factor_df = (fi_data4 * fi_data4_df)
for factor, sub_factor_lst in factor_dict.items():
    print(factor)
    factor_df[factor] = factor_df[sub_factor_lst].sum(axis=1)
    # factor_df[sub_factor_lst + [factor]].to_csv(f'2-2-{factor}_percent_mean.csv')


# %%
# P3
print('3------------P3')
factor_lst = list(factor_dict.keys())
factor_df1 = factor_df[factor_lst].reset_index()
factor_rank_lst = [factor+'_rank' for factor in factor_lst]
factor_percent_lst = [factor+'_percent' for factor in factor_lst]
factor_df1[factor_rank_lst] = factor_df1.groupby(merge_cols[2:])[factor_lst].rank(method='max')
factor_df_max = factor_df1.groupby(merge_cols[2:])[factor_rank_lst].max()
factor_df_max.columns = [x + '_max' for x in factor_df_max.columns]
factor_df1 = factor_df1.merge(factor_df_max.reset_index(), on=merge_cols[2:], how='left')
factor_df1[factor_percent_lst] = factor_df1[factor_rank_lst].values / factor_df1[factor_df_max.columns].values
# factor_df1[merge_cols + factor_lst + factor_rank_lst + list(factor_df_max.columns) + factor_percent_lst].sort_values(
#         ['enddate', 'ths_indsutry_name_l2', factor_lst[0]]).to_csv('3-factor_raw.csv')

# P4
print('4------------P4')
factor_df2 = factor_df1.set_index(merge_cols)
factor_df2[factor_percent_lst] = factor_df2[factor_percent_lst] * 5

# P5
print('5------------P5')
w3 = np.ones(6)
w3 /= w3.sum()
factor_df21 = pd.DataFrame([w3] * len(factor_df2), index=factor_df2.index, columns=factor_percent_lst)
factor_df3 = (factor_df2[factor_percent_lst] * factor_df21).sum(axis=1).rename('raw_score').reset_index()

# P6
print('6------------P6')
factor_df3['score'] = - ((- round(factor_df3['raw_score'], 3)) // 1)
factor_df3.to_csv('6-factor_last.csv')


# %%
# backtest1
def get_backtest1(score, price_df, days='3M'):
    """

    :param stock_df: columns = ['stock_code', 'date', 'score']
    :param price_df: columns = ['stock_code', 'date', 'close']
    :param days:
    :return:
    """
    price_df_new = price_df[score.columns]
    for score_num in range(5):
        port_ts, port_count = get_port_ts(score, score_num)


def get_port_ts(score, score_num):
    port_ts_tmp = score.where((score > score_num) & (score <= score_num), 0)
    port_ts = port_ts_tmp.where(port_ts_tmp == 0, 1)
    port_count = port_ts.sum(axis=1).rename('num_' + str(score_num))
    port_ts = (port_ts.T / port_count).T
    return port_ts, port_count


def get_price_df():
    today = str(pd.to_datetime('today'))[:10]
    stock_detail = query_iwencai("上市日期小于%s" % today)
    stock_list = list(set(stock_detail['股票代码']))
    price_df = get_price(stock_list, '1990-01-01', today, '1d', ['close'], fq='pre')
    calendars = pd.date_range('1990-06-30', str(pd.to_datetime('today'))[:10], freq='3M')
    price_df = price_df.set_index(['stock_code', 'date']).unstack([-2])
    price_df_other = pd.DataFrame(columns=price_df.columns, index=calendars.difference(price_df.index))
    price_df = pd.concat([price_df, price_df_other]).sort_index().ffill()
    return price_df


factor_df3 = pd.read_csv('6-factor_last.csv')
score = factor_df3.set_index(['stock_code', 'date'])['score'].unstack([-2])






















