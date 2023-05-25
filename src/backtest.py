import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def cal_annualized_return(asset_return, multiplier):
    if len(asset_return) > 1:
        annualized_return = multiplier * asset_return.mean()
        annualized_return = np.exp(annualized_return) - 1
    else:
        annualized_return = np.nan
    return annualized_return


def cal_annualized_excess_return(asset_return, index_return, multiplier):
    if len(asset_return) > 1:
        annualized_return = multiplier * (asset_return - index_return).mean()
        annualized_return = np.exp(annualized_return) - 1
    else:
        annualized_return = np.nan
    return annualized_return


def cal_beta(asset_return, index_return, multiplier, interest=0.03):
    if len(asset_return) > 1 and len(index_return) > 1:
        rf = interest / multiplier
        y = asset_return - rf
        x = index_return - rf
        x = sm.add_constant(x)
        z = pd.concat([x, y], axis=1).fillna(0)
        x = z[z.columns[:2]]
        y = z[z.columns[2]]
        # print('y', y.tolist(), 'x', x['close'].tolist())
        model = sm.OLS(y, x).fit()
        if len(y) > 1:
            beta = model.params.values[1]
            alpha = model.params.values[0]
            alpha = np.exp(multiplier * alpha) - 1
        else:
            beta = np.nan
            alpha = np.nan
    else:
        beta = np.nan
        alpha = np.nan
    return beta, alpha


def cal_sharpe_ratio(asset_return, multiplier, interest=0.03):
    if len(asset_return) > 1:
        annualized_return = multiplier * asset_return.mean()
        annualized_return = np.exp(annualized_return) - 1
        annualized_vol = asset_return.std(ddof=1) * np.sqrt(multiplier)
        sharpe_ratio = (annualized_return - interest) / annualized_vol
    else:
        sharpe_ratio = np.nan
    return sharpe_ratio


def cal_information_ratio(asset_return, index_return, multiplier):
    if len(asset_return) > 1 and len(index_return) > 1:
        active_return = asset_return - index_return
        tracking_error = (active_return.std(ddof=1)) * np.sqrt(multiplier)
        asset_annualized_return = multiplier * asset_return.mean()
        index_annualized_return = multiplier * index_return.mean()
        information_ratio = (asset_annualized_return - index_annualized_return) / tracking_error
    else:
        information_ratio = np.nan
    return information_ratio


def cal_mdd(asset_return):
    """
    :param asset_return: 计算区间内的收益率序列
    """
    if len(asset_return) > 1:
        asset_return = np.log(asset_return + 1)
        asset_return.dropna(inplace=True)
        running_max = np.maximum.accumulate(asset_return.cumsum())
        underwater = asset_return.cumsum() - running_max
        underwater = np.exp(underwater) - 1
        mdd = underwater.min()
    else:
        mdd = np.nan
    return - mdd


def cal_excess_winning_rate(asset_return, index_return):
    if len(asset_return) > 1:
        return_diff = asset_return - index_return
        winning_rate = len(return_diff[return_diff > 0]) / len(return_diff)
    else:
        winning_rate = np.nan
    return winning_rate


def cal_winning_rate(asset_return):
    if len(asset_return) > 1:
        return_diff = asset_return
        winning_rate = len(return_diff[return_diff > 0]) / len(return_diff)
    else:
        winning_rate = np.nan
    return winning_rate


def cal_wr_di_ret(asset_return):
    wr = cal_winning_rate(asset_return)
    ret = cal_annualized_return(asset_return, multiplier=1)
    return wr / ret


def cum(ret):
    return np.exp(np.log(ret + 1).cumsum()) - 1


def get_port_ts(score, i):
    cond = (score > i) & (score <= (i + 1))
    port = score.where(cond, 0).where(~cond, 1)
    port = port.T / port.sum(axis=1)
    return port.T


def resample_data(port, m, k):
    dates = np.arange(len(port))
    resample_dates = dates[[x % m != k for x in dates]]
    port.iloc[resample_dates] = np.nan
    return port.ffill()


def resample_ret(ret, window):
    ret = np.log(ret + 1)
    freq_ret = ret.resample(window).sum()
    freq_ret = np.exp(freq_ret) - 1
    return freq_ret


def plot(series_lst, name):
    plt.figure(figsize=(15, 8))
    plt.rcParams["font.family"] = 'Songti SC'
    for series_name, series in series_lst.items():
        plt.plot(series, label=series_name)
    plt.title(name)
    plt.legend()
    name1 = name.split('-')[0]
    plt.savefig(f"./{name1}/{name}.png")


def ret_risk(asset_ret, index_ret):
    """"""
    # print('asset_ret, index_ret')
    # print(asset_ret, index_ret)
    annual_ret = cal_annualized_return(asset_ret, 252)
    annual_excess_ret = cal_annualized_return(asset_ret - index_ret, 252)
    beta, alpha = cal_beta(asset_ret, index_ret, 252)
    sharpe = cal_sharpe_ratio(asset_ret, 252)
    ir = cal_information_ratio(asset_ret, index_ret, 252)
    mdd = cal_mdd(asset_ret)
    freq_ret = []
    for window in ['D', 'W', 'M', 'Y']:
        asset_freq = resample_ret(asset_ret, window)
        index_freq = resample_ret(index_ret, window)
        wr_ret_asset = cal_wr_di_ret(asset_freq)
        wr_ret_exces = cal_wr_di_ret(asset_freq - index_freq)
        freq_ret.extend([wr_ret_asset, wr_ret_exces])
    # print('[annual_ret, annual_excess_ret, beta, alpha, sharpe, ir, mdd] + freq_ret')
    # print([annual_ret, annual_excess_ret, beta, alpha, sharpe, ir, mdd] + freq_ret)
    return [annual_ret, annual_excess_ret, beta, alpha, sharpe, ir, mdd] + freq_ret


def backtest(score, ret, index_ret, name, k=0):
    ret_risk_data = []
    x, y = np.floor(score.min().min()), np.floor(score.max().max())
    asset_cum = {}
    for i in range(int(x), int(y)+1):  # 0-1, 1-2, 2-3, 3-4, 4-5
        for m in [1, 3, 5, 10, 20]:  # 一日、三日、五日、十日、二十日持仓
            port_ts = resample_data(get_port_ts(score, i), m, k)
            # print('port_ts, ret', port_ts, ret, port_ts.sum(axis=1))
            # port_ts.to_csv('a.csv')
            # print(port_ts.iloc[1].values)
            # print(port_ts.columns)
            # ret1 = ret[port_ts.columns[:-1]]
            # print(ret1.iloc[1].values)
            # print(ret1.iloc[1].values[port_ts.iloc[1].values[:-1] == 0.25])
            asset_ret = (port_ts.shift(1) * ret).sum(axis=1)
            # print('port_ts, asset_ret')
            # print(port_ts.T.sort_values('2018-01-02').T, asset_ret.iloc[:50])
            asset_cum.update({str(i) + '-' + str(m): cum(asset_ret)})
            plot({'asset': cum(asset_ret), 'index': cum(index_ret)}, name + '-' + str(i) + '-' + str(m))
            ret_risk_data.append(ret_risk(np.log(asset_ret+1), np.log(index_ret+1)))
    asset_cum.update({'index': cum(index_ret)})
    plot(asset_cum, name)
    col = ['年化收益率', '超额年化收益', '夏普比率', '最大回撤', 'Alpha', 'Beta', '信息比率',
           '日胜率/日平均涨幅', '日超额胜率/日平均超跌涨幅', '周胜率/周平均涨幅', '周超额胜率/周平均超跌涨幅',
           '月胜率/月平均涨幅', '月超额胜率/月平均超跌涨幅', '年胜率/年平均涨幅', '年超额胜率/年平均超额涨幅']
    a = [f'{i}_{i+1}' for i in range(int(x), int(y)+1)]
    b = ['1', '3', '5', '10', '20']
    index = pd.MultiIndex.from_product([a, b], names=['score', 'period'])
    ret_risk_data = pd.DataFrame(ret_risk_data, columns=col, index=index)
    ret_risk_data.to_excel(f'./{name}/{name}_backtest.xlsx')

