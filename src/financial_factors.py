#!/usr/bin/env python
# coding: utf-8
import matplotlib as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import datetime
import tqdm
import os


# initialization
start_date = '2001-01-01'
end_date = (datetime.datetime.today() - pd.Timedelta('1d')).strftime('%Y-%m-%d')
print(start_date, end_date)
y_now = int(datetime.datetime.today().year)
m_now = int(datetime.datetime.today().month)
# x = stats.percentileofscore([1, 2, 3, 4], 3)


# 取出季度日历
statDate_sq = []
for y in range(1990, y_now + 1):
    for q in range(1, 5):
        if (y == y_now) and (q > 1):  # FIXME
            continue
        else:
            statDate_sq.append(str(y) + 'q' + str(q))
statDate_sq = statDate_sq
print(statDate_sq[0], statDate_sq[-1])

statDate_y = [str(i) for i in range(1990, y_now)]


# calculate the score
def data_processing(df1, i_name):
    # mean and standard deviation

    df1.index = df1['date']
    tmp = df1.groupby('symbol').resample('Y').last()
    temp0 = tmp[df1.columns[:4]].reset_index(drop=True)
    tmp1 = tmp[df1.columns[4:]].reset_index()
    temp1 = tmp1.groupby('symbol')[tmp1.columns[2:]].rolling(5, min_periods=1).mean()
    temp2 = tmp1.groupby('symbol')[tmp1.columns[2:]].rolling(5, min_periods=1).std()

    temp1.columns = [i + "_mean" for i in temp1.columns]
    temp2.columns = [i + "_std" for i in temp2.columns]

    # industry
    df0 = pd.concat([temp0, temp1.reset_index(drop=True), temp2.reset_index(drop=True)], axis=1)
    # df0 = pd.merge(temp1.reset_index(), temp2.reset_index(), how='left', on=['symbol', 'report_date', 'date'])
    df2 = pd.merge(df1.reset_index(drop=True), df0, how='left', on=['symbol', 'report_date', 'date', 'SW_2'])
    # df2 = pd.merge(df2, sw2, how='left', left_on=['symbol'], right_on=['symbol'])
    df2['SW_2'] = df2['SW_2'].apply(lambda x: str(x))
    df2 = df2.replace('nan', np.nan)

    df2 = df2[~df2.duplicated()]
    df2 = df2.iloc[:, ~df2.columns.duplicated()]

    # percentage
    df3 = df2.set_index(['symbol', 'report_date', 'date', 'SW_2']).groupby(level=['date', 'SW_2']).apply(
        lambda x: x.rank(axis=0, numeric_only=True, na_option='bottom', ascending=False, pct=True, method='min'))
    # set weights# df3 = cal_m(df3)
    df3 = df3.reset_index(level=[-4, -3, -2, -1]).reset_index(drop=True)

    # ability
    df3_tmp = df3.set_index(['report_date', 'SW_2'])[tmp1.columns]
    df4 = pd.DataFrame(df3_tmp.groupby(['symbol', 'date']).apply(lambda x: x.mean(axis=1)), columns=[i_name]).reset_index()
    rank = df4.groupby(['SW_2', 'date']).apply(lambda x: x.rank(numeric_only=True, na_option='top', pct=True, method='min'))

    if pd.__version__ < '2.0.0':
        df5 = df4.copy()
        df5[i_name] = rank.reset_index()[i_name]
        rank = df5.set_index(['report_date', 'symbol'])
    else:
        rank = rank.reset_index(level=['SW_2', 'date']).sort_index()

    norm_rank = rank.groupby(['SW_2', 'date']).apply(lambda x: (1 - x / x.max()) * 5).reset_index(level=[0, 1])[i_name]
    df4['%s_n5' % i_name] = norm_rank
    df4 = pd.merge(df4, df3, on=['date', 'report_date', 'symbol', 'SW_2'])
    # exec('df_%s = df4' % i_name)
    df4.to_csv('%s.csv' % i_name, encoding='utf-8')
    # eval('df_%s' % i_name, globals())


def get_data_from_local_csv(df, i_name, cls):

    df1 = df[cls]
    df1.sort_values(['symbol', 'date', 'SW_2'], inplace=True)
    print(i_name)
    data_processing(df1, i_name)


# %%
path = r'D:\work\20230524\A股十年财务数据\A股十年财务数据.csv'
df = pd.read_csv(path, header=0, index_col=None, ).query(
    "ths_indsutry_name_l2=='房地产开发'")
df = df.rename(
    columns={'stock_code': 'symbol', 'ths_indsutry_name_l2': 'SW_2', 'enddate': 'date', 'pub_date': 'report_date'})
df['date'] = pd.to_datetime(df['date'].astype(str))
df['report_date'] = pd.to_datetime(df['report_date'].astype(str).str[:8])
# 数据计算
# profit
i_name = 'profit'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['roe_ttm', 'gross_profit_ratio', 'profit_ratio_ttm', 'net_profit_ratio_ttm']
get_data_from_local_csv(df, i_name, cls)

# growth
i_name = 'growth'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['operating_income_increase_ratio_ttm', 'operating_profit_increase_ratio_ttm',
             'net_profit_increase_ratio_ttm', 'equity_belong_mgs_increase_ratio']
get_data_from_local_csv(df, i_name, cls)

# operate
i_name = 'operate'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['accounts_receivable_turnover', 'inventory_turnover', 'accounts_payable_turnover', 'total_asset_turnover']
get_data_from_local_csv(df, i_name, cls)

# cashflow
i_name = 'cashflow'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['net_profit_cash_ratio', 'fcf_equity_ratio', ]
get_data_from_local_csv(df, i_name, cls)

# debt
i_name = 'debt'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['quick_ratio', 'liquidity_ratio', 'debt_contain_interest_ratio']
get_data_from_local_csv(df, i_name, cls)

# quality
i_name = 'quality'
cls = ['report_date', 'symbol', 'date', 'SW_2']
cls = cls + ['accounts_receivable_ratio', 'inventory_ratio', 'goodwill_ratio', ]
get_data_from_local_csv(df, i_name, cls)


# consolidation
df = pd.DataFrame([], columns=['date', 'report_date', 'symbol', 'il2_name'])
for i in ['profit', 'growth', 'operate', 'cashflow', 'debt', 'quality']:
    exec("df_%s = pd.read_csv('./%s.csv', index_col=0,header=0)" % (i, i))
    exec("df_%s = df_%s[['date', 'report_date','symbol','il2_name','%s_n5']]" % (i, i, i))
    exec("df = pd.merge(df, df_%s, how='outer',on=['date', 'report_date','symbol','il2_name'])" % i)
df = df.query(
    "il2_name!='银行' & il2_name!='保险及其他' & il2_name!='证券'")  # ['il2_name'].unique()#to_csv('test.csv', encoding='utf-8-sig')
df = df[~df['symbol'].str.contains('BJ')]
