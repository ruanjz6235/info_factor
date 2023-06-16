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


if_params = True
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
    cols2 = [c + f'_lag{i}' for c in cols1[1:]]
    fi_data0 = fi_data[cols1].rename(columns=dict(zip(cols1[1:], cols2)))
    fi_data = fi_data.merge(fi_data0, on=['stock_code', f'enddate_lag{i}'], how='left')

for i, index_col in enumerate(index_cols):
    # if i > 0:
    #     break
    print(index_col)
    sub_col = [index_col, index_col + '_lag1', index_col + '_lag2', index_col + '_lag3', index_col + '_lag4']
    sub_col_new = [index_col + '_mean', index_col + '_std', index_col + '_pos']
    fi_data[index_col + '_mean'] = fi_data[sub_col].mean(axis=1)
    fi_data[index_col + '_std'] = fi_data[sub_col].std(ddof=0, axis=1)
    fi_data[index_col + '_pos'] = pos(fi_data[sub_col])
    # fi_data[bb_cols + sub_col + sub_col_new].to_csv(f'0-{index_col}.csv', index=False)

    fi_data_index_count = fi_data[sub_col].count(axis=1)
    fi_data.loc[fi_data_index_count <= 1, index_col + '_std'] = 1e10
    fi_data[index_col + '_pos'] = fi_data[index_col + '_pos'].fillna(-1e10)

# %%
# 3. P1
print('1------------P1')
index_cols_mean = [x + '_mean' for x in index_cols]
index_cols_std = [x + '_std' for x in index_cols]
index_cols_pos = [x + '_pos' for x in index_cols]
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
    sub_rank_col = [x + '_rank' for x in sub_col0]
    sub_percent_col = [x + '_percent' for x in sub_col0]
    fi_data2[sub_rank_col] = fi_data2.groupby(merge_cols[2:])[sub_col0].rank(method='max')
    df_max = fi_data2.groupby(merge_cols[2:])[sub_rank_col].max()
    df_max.columns = [x + '_max' for x in df_max.columns]
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
    sub_w2 = w2_old[n: n + num]
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
factor_rank_lst = [factor + '_rank' for factor in factor_lst]
factor_percent_lst = [factor + '_percent' for factor in factor_lst]
factor_df1[factor_rank_lst] = factor_df1.groupby(merge_cols[2:])[factor_lst].rank(method='max')
factor_df_max = factor_df1.groupby(merge_cols[2:])[factor_rank_lst].max()
factor_df_max.columns = [x + '_max' for x in factor_df_max.columns]
factor_df1 = factor_df1.merge(factor_df_max.reset_index(), on=merge_cols[2:], how='left')
factor_df1[factor_percent_lst] = factor_df1[factor_rank_lst].values / factor_df1[factor_df_max.columns].values
# factor_df1[merge_cols + factor_lst + factor_rank_lst + list(factor_df_max.columns) + factor_percent_lst].sort_values(
#         ['enddate', 'ths_indsutry_name_l2', factor_lst[0]]).to_csv('3-factor_raw.csv')

# %%
# P4
def get_score(dt):
    api = get_arsenal_api('cbas-babel-frondend-pod')
    dt = pd.to_datetime(dt).strftime("%Y%m%d")
    if len(dt) > 0:
        r = api.get('babel/api/query_index_by_time', params={
            'indexId': "500000.40002276, 500000.40002278, 500000.40009199, 500000.40009200, 500000.40013712, 500000.40013714, 500000.40155400, 500000.40000000",
            "time": dt})
        result = r.json()
        # print(result)

        result = pd.DataFrame(result['data'])
        if result.shape[1] >= 3:
            print(dt, end=';')
            result = result.rename(
                columns={"500000_40002276": "net_profit", "500000_40002278": "net_profit_ttm", "500000_40009199": "profit_after",
                         "500000_40009200": "profit_after_ttm", "500000_40013712": "operate_revenue", "500000_40013714": "operate_revenue_ttm",
                         "500000_40155400": "report_date", "500000_40000000": 'symbol', 'date': 'date'})
            result['date'] = dt
            del result['uid']
            print(dt)
            return result


# net_profit, profit_after, operate_revenue
def get_scores():
    tradingdays = get_trade_days('1990-06-30', '2023-06-15')
    score = []
    for date_index in range(len(tradingdays)):
        sub_score = get_score(tradingdays[date_index])
        score.append(sub_score)
    return pd.concat(score)


def get_iwencai_data(input_name, output_name, tps):
    quarter_c = ['一季报', '二季报', '三季报', '年报']
    quarter_e = ['-03-31', '-06-30', '-09-30', '-12-31']
    quarter_d = dict(zip(quarter_c, quarter_e))
    df2 = []
    for year, qc in product(range(1990, 2024), quarter_c):
        sub_df2 = []
        for tp in tps:
            ss_df2 = query_iwencai(f"全市场{year}年{qc}{input_name}{tp}")
            ss_df2.columns = ['stock_code', 'name', output_name]
            ss_df2['tp'] = tp
            sub_df2.append(ss_df2)
        sub_df2 = pd.concat(sub_df2)
        sub_df2['calendar'] = str(year) + quarter_d[qc]
        df2.append(sub_df2)
    df2 = pd.concat(df2)
    df2 = df2[~df2[output_name].isna()]
    df2.loc[df2['tp'] == '', 'tp'] = 'mrq'
    return df2


def data_process():
    mrq_data = get_scores()
    audit_data = get_iwencai_data('审计意见', 'audit', [''])
    ctrl_data = pd.read_excel(r'内控意见.xlsx')
    ctrl_cols = [x.split(' ')[-1] for x in ctrl_data.columns]


print('4------------P4')
factor_df2 = factor_df1.set_index(merge_cols)
factor_df2[factor_percent_lst] = factor_df2[factor_percent_lst] * 5
if if_params:
    pass

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
def get_backtest1(score, price_df, price_df_index, dates):
    """

    :param stock_df: columns = ['stock_code', 'date', 'score']
    :param price_df: columns = ['stock_code', 'date', 'close']
    :param days:
    :return:
    """
    cols = score.columns.intersection(price_df.columns)
    price_df_new = price_df[cols].loc[dates]
    ret_df = price_df_new.shift(-12).ffill() / price_df_new - 1
    ret_df = ret_df.loc[score.index]
    ret_port = pd.DataFrame(index=score.index)

    score = score[cols]
    for score_num in range(5):
        port_ts, port_count = get_port_ts(score, score_num)
        ret_attr = (port_ts * ret_df).sum(axis=1).rename('score_' + str(score_num + 1))
        ret_port = pd.concat([ret_port, pd.DataFrame(ret_attr), pd.DataFrame(port_count)], axis=1)
    score_cols = ret_port.columns[ret_port.columns.str.startswith('score_')]
    num_cols = ret_port.columns[ret_port.columns.str.startswith('num_')]
    ret_score = ret_port[score_cols]
    ret_score_df = pd.DataFrame([np.arange(1, 6)] * len(ret_score), columns=ret_score.columns, index=ret_score.index)
    ret_port['pos'] = (ret_score * ret_score_df).sum(axis=1)
    ret_port['futuredate'] = ret_port.index + pd.DateOffset(years=1)
    ret_port.loc[ret_port['futuredate'] > ret_port.index[-1], 'futuredate'] = ret_port.index[-1]

    price_df_index_new = price_df_index.loc[dates]
    ret_index_df = price_df_index_new.shift(-12).ffill() / price_df_index_new - 1
    ret_index_df = ret_index_df.loc[score.index]
    ret_port = pd.concat([ret_port, ret_index_df], axis=1)
    ret_port = ret_port[['futuredate', 'pos'] + list(num_cols) + list(score_cols) + list(price_df_index.columns)]
    ret_port.to_csv('backtest1_band.csv')


def get_port_ts(score, score_num):
    port_ts_tmp = score.where((score > score_num) & (score <= score_num + 1), 0)
    port_ts = port_ts_tmp.where(port_ts_tmp == 0, 1)
    port_count = port_ts.sum(axis=1).rename('num_' + str(score_num + 1))
    port_ts = (port_ts.T / port_count).T
    return port_ts, port_count


def get_price_df():
    today = str(pd.to_datetime('today'))[:10]
    stock_detail = query_iwencai("上市日期小于%s" % today)
    stock_list = list(set(stock_detail['股票代码']))
    price_df = get_price_df_index(stock_list)
    return price_df


def get_price_df_index(stock_list):
    today = str(pd.to_datetime('today'))[:10]
    price_df = get_price(stock_list, '1990-01-01', today, '1d', ['close'], fq='post')
    calendars = pd.date_range('1990-06-30', str(pd.to_datetime('today'))[:10], freq='M')
    price_df = pd.concat(price_df)['close'].unstack([-2])
    price_df_other = pd.DataFrame(columns=price_df.columns, index=calendars.difference(price_df.index))
    price_df = pd.concat([price_df, price_df_other]).sort_index()
    price_df = price_df.where(price_df != 0, np.nan)
    return price_df.ffill()


def normalize_dates(df_pivot_old, dates, calendar='df'):
    df_pivot = df_pivot_old.copy()
    # 日期归为月底
    df_pivot.index = pd.to_datetime(df_pivot.index)
    index1 = df_pivot.index[~df_pivot.index.is_month_end]
    index2 = df_pivot.index[df_pivot.index.is_month_end]
    index_dict = dict(zip(index1, index1 + pd.offsets.MonthEnd()))
    index_dict.update(dict(zip(index2, index2)))
    df_pivot.index = df_pivot.index.map(index_dict)
    # 只取季度末数据
    df_pivot = df_pivot.loc[df_pivot.index.is_quarter_end]
    if calendar == 'month':
        # 填充到每个月的数据(按季度填充)
        df_pivot_other = pd.DataFrame(columns=df_pivot.columns, index=dates.difference(df_pivot.index))
        df_pivot = pd.concat([df_pivot, df_pivot_other]).sort_index()
        # 按季度填充
        df_pivot['calendar'] = df_pivot.index.map(lambda x: str(x)[:5] + str((x.month + 2) // 3))
        df_pivot['calendar'] = df_pivot['calendar'].shift(-1).ffill()
        df_pivot[df_pivot.columns[:-1]] = df_pivot.groupby('calendar').ffill()
    elif calendar == 'day':
        df_pivot = df_pivot_old.copy()
        df_pivot_other = pd.DataFrame(columns=df_pivot.columns, index=dates.difference(df_pivot.index))
        df_pivot = pd.concat([df_pivot, df_pivot_other]).sort_index().ffill()

        df_pivot['calendar'] = df_pivot.index.map(lambda x: str(x)[:5] + str((x.month + 2) // 3))
        df_pivot['calendar'] = df_pivot['calendar'].shift(-1).ffill()
        df_pivot[df_pivot.columns[:-1]] = df_pivot.groupby('calendar').ffill()
    else:
        df_pivot['calendar'] = range(len(df_pivot))
        df_pivot_other = pd.DataFrame(columns=df_pivot.columns, index=dates.difference(df_pivot.index))
        df_pivot = pd.concat([df_pivot, df_pivot_other]).sort_index()
        df_pivot['calendar'] = df_pivot['calendar'].ffill()
        df_pivot[df_pivot.columns[:-1]] = df_pivot.groupby('calendar').ffill()
    return df_pivot[df_pivot.columns[:-1]].dropna(how='all')


# %%
def get_backtest2(score, price_df, price_df_index):
    cols = score.columns.intersection(price_df.columns)
    price_df_new = price_df[cols]
    score_new = score[cols]

    ret_df = price_df_new.pct_change()
    ret_df_index = price_df_index.pct_change()
    ret_port = pd.DataFrame(index=score.index)
    price_df_new.reset_index().to_feather('price_df.feather')
    ret_df.reset_index().to_feather('ret_df.feather')
    score_new.reset_index().to_feather('score_new.feather')
    for score_num in range(5):
        port_ts, port_count = get_port_ts(score_new, score_num)
        port_ts.reset_index().to_feather(f'{score_num}_port_ts.feather')
        ret_attr = (port_ts * ret_df).sum(axis=1).rename('score_' + str(score_num + 1))
        ret_port = pd.concat([ret_port, pd.DataFrame(ret_attr), pd.DataFrame(port_count)], axis=1)
    score_cols = ret_port.columns[ret_port.columns.str.startswith('score_')]
    num_cols = ret_port.columns[ret_port.columns.str.startswith('num_')]
    ret_port = pd.concat([ret_port, ret_df_index], axis=1)
    ret_port['excess_300'] = ret_port['score_5'] - ret_port['000300.SH']
    ret_port['excess_500'] = ret_port['score_5'] - ret_port['000905.SH']
    ret_port['ls'] = ret_port['score_5'] - ret_port['000905.SH']
    cols2 = ['excess_300', 'excess_500', 'ls']
    # ret_port[score_cols] = (ret_port[score_cols] + 1).cumprod() - 1
    ret_port[list(num_cols) + list(score_cols) + list(ret_df_index.columns) + cols2].to_csv('backtest2_detail.csv')
    # from backtest import ret_risk
    # ret_risk_df = ret_risk(np.log(ret_port[score_cols] + 1), np.log(ret_df_index['000300.SH'] + 1))
    # ret_risk_df.to_csv('backtest2_index.csv')


# %% 回测
factor_df3 = pd.read_csv('6-factor_last.csv')
score = factor_df3.set_index(['stock_code', 'enddate'])['score'].unstack([-2])
dates = pd.date_range('1990-06-30', str(pd.to_datetime('today'))[:10], freq='M')
score = normalize_dates(score, dates)
# score.to_feather('score.feather')
price_df_index = get_price_df_index(stock_list=['000300.SH', '000905.SH'])
price_df = get_price_df()
# get_backtest1(score, price_df, price_df_index, dates)

score_new = normalize_dates(score, dates=price_df.index, calendar='day')
get_backtest2(score_new, price_df, price_df_index)

# %%
