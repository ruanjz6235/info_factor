import pandas as pd
import numpy as np
from functools import lru_cache


class BaseData:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    @property
    def trade_days(self):
        return pd.DataFrame(get_trade_days(self.start_date, self.end_date))

    @trade_days.setter
    def trade_days(self, dates):
        if isinstance(dates, pd.DataFrame) or isinstance(dates, pd.Series):
            return dates
        elif isinstance(dates, list) and len(dates) == 2:
            return pd.DataFrame(get_trade_days(dates[0], dates[1]))
        else:
            return pd.DataFrame(dates)

    @property
    def stock_detail(self):
        return query_iwencai("上市日期小于等于%s" % self.end_date)

    @stock_detail.setter
    def stock_detail(self, end_date):
        return query_iwencai("上市日期小于等于%s" % end_date)

    @property
    def stock_list(self):
        return list(set(self.stock_detail['股票代码']))

    @property
    @lru_cache(maxsize=999)
    def price_detail(self):
        price_detail = get_price(self.stock_list,
                                 self.start_date,
                                 pd.to_datetime('today').strftime("%Y-%m-%d"),
                                 '1d',
                                 ['open', 'close'],
                                 fq='pre')
        price_detail = pd.concat(price_detail)
        open_price, close_price = price_detail['open'], price_detail['close']

        price_detail_nfq = get_price(self.stock_list,
                                     self.start_date,
                                     pd.to_datetime('today').strftime("%Y-%m-%d"),
                                     '1d',
                                     ['prev_close'],
                                     fq=None)
        prev_close = pd.concat(price_detail_nfq)['prev_close']

        return open_price, close_price, prev_close

    @property
    @lru_cache(maxsize=999)
    def ret(self):
        open_price, close_price, prev_close = self.price_detail
        open_price, close_price, prev_close = open_price.ffill(), close_price.ffill(), prev_close.ffill()
        ret = np.log(open_price.pct_change() + 1)
        opens = ret, ret.rolling(3).sum(), ret.rolling(5).sum(), ret.rolling(10).sum(), ret.rolling(20).sum()
        return [np.exp(sub_open) - 1 for sub_open in opens]





