import pandas as pd
import numpy as np

start, end = pd.to_datetime('2018-06-03'), pd.to_datetime('2023-06-02')
args1, args2 = [0, 15, 40, 80], [1, 3, 10, 20]
codes = ['000300.SH', '000001.SH', '399001.SZ', '399006.SZ']


# babel取数
def get_score(dt):
    api = get_arsenal_api('cbas-babel-frondend-pod')
    dt = pd.to_datetime(dt).strftime("%Y%m%d")
    if len(dt) > 0:
        r = api.get('babel/api/query_index_by_time', params={
            'indexId': "500000.40002321, 500000.40000030, 500000.40008019, 500000.40008010, 500000.40026140, 500000.40008004, 500000.40000000",
            "time": dt})
        result = r.json()
        # print(result)

        result = pd.DataFrame(result['data'])
        if result.shape[1] >= 3:
            print(dt, end=';')
            result = result.rename(
                columns={"500000_40002321": "roettm", "500000_40000030": "score", "500000_40008019": "ps",
                         "500000_40008010": "pe", "500000_40026140": "pb", "500000_40008004": "pcfo",
                         "500000_40000000": 'symbol', 'date': 'date'})
            result['date'] = dt
            del result['uid']
            print(dt)
            return result.dropna(subset=['score'])


# %%
def map_pe(x, *args):
    x0, x1, x2, x3 = args
    if x < x0:
        return f'-{x0}'
    elif x < x1:
        return f'{x0}-{x1}'
    elif x < x2:
        return f'{x1}-{x2}'
    elif x < x3:
        return f'{x2}-{x3}'
    else:
        return f'{x3}-'


dt_list = get_trade_days(start, end)
score = []
for dt in dt_list:
    print(dt)
    score.append(get_score(dt))
score = pd.concat(score)
score.reset_index(drop=True).to_feather('all_score.feather')

# %%
score = pd.read_feather('all_score.feather')
score['date'] = pd.to_datetime(score['date'])

score['score_'] = score['score'].apply(lambda x: 'score_' + str(int(np.floor(x))))
score.loc[score['score_'] == 'score_5', 'score_'] = 'score_4'
a = score.groupby(['date', 'score_'])['symbol'].count().unstack()
a = (a.T / a.sum(axis=1)).T

score['pe_'] = score['pe'].apply(map_pe, args=args1)
score['pe_'] = 'pe_' + score['pe_']
b = score.groupby(['date', 'pe_'])['symbol'].count().unstack()
b = (b.T / b.sum(axis=1)).T

score['pb_'] = score['pb'].apply(map_pe, args=args2)
score['pb_'] = 'pb_' + score['pb_']
c = score.groupby(['date', 'pb_'])['symbol'].count().unstack()
c = (c.T / c.sum(axis=1)).T

index_ret = np.exp(np.log(pd.concat(get_price(codes,
                                              start - pd.DateOffset(days=2),
                                              end,
                                              '1d',
                                              ['close'],
                                              fq='pre')
                                    )['close'].unstack([-2]).pct_change().fillna(0).iloc[1:] + 1).cumsum()) - 1

all_score_count = pd.concat([a, b, c, index_ret], axis=1).dropna()
all_score_count.to_excel('all_score_count.xlsx')
