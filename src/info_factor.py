import pandas as pd
import numpy as np
import torch
import re
from DataAPI.openapi.open_api import OpenAPI
from functools import reduce
from backtest import backtest
# from const import periods
periods = [1, 3, 5, 10, 20]


def get_df_union_col_query(df1, df2, col):
    df1[col] = df1[col].astype(str)
    df2[col] = df2[col].astype(str)
    return df1[reduce(lambda x, y: x + y, [df1[i] for i in col]).isin(
        reduce(lambda x, y: x + y, [df2[i] for i in col]))]


class BaseGenerator:
    def __init__(self, dates, sentence=''):
        """"""
        self.dates = dates
        self.sentence = sentence.format(**dict(zip(['start_date', 'end_date'], dates)))
        self.wencai_data = False
        self.api = OpenAPI()

        self.stock_detail = None
        self.price_detail = None
        self.tradeday = None
        self.map_tp = 'df'
        self.lst_tp = 'df'  # 'df' or 'price'

    def get_price(self, price_df=None, stock_detail=None, stock_list=None):
        if price_df is not None:
            self.stock_detail = stock_detail
            self.stock_list = stock_list
            self.price_detail = price_df
        else:
            if stock_list:
                self.stock_detail = query_iwencai("上市日期小于%s" % pd.to_datetime('today').strftime("%Y-%m-%d"))
                self.stock_list = list(set(self.stock_detail['股票代码']))
            else:
                self.stock_list = stock_list
            price_detail = get_price(self.stock_list,
                                     self.dates[0],
                                     self.dates[1],
                                     '1d',
                                     ['open', 'close'],
                                     fq='pre')
            self.price_detail = pd.concat(price_detail)
            self.price_detail.index.names = ['stock_code', 'date']
        self.tradeday = self.price_detail.index

    def get_nfq_prev_close(self, stock_list=None):
        if not stock_list:
            self.stock_detail = query_iwencai("上市日期小于%s" % pd.to_datetime('today').strftime("%Y-%m-%d"))
            stock_list = list(set(self.stock_detail['股票代码']))
        price_detail_nfq = get_price(stock_list,
                                     self.dates[0],
                                     self.dates[1],
                                     '1d',
                                     ['prev_close'],
                                     fq=None)
        self.price_detail_nfq = pd.concat(price_detail_nfq)
        self.price_detail_nfq.index.names = ['stock_code', 'date']
        return self.price_detail_nfq.reset_index()

    def get_type_ret(self):
        for period in periods:
            self.price_detail[f'{period}_open'] = self.price_detail.groupby('stock_code')['open'].apply(
                lambda x: x.shift(-period) / x - 1)
            self.price_detail[f'{period}_close'] = self.price_detail.groupby('stock_code')['close'].apply(
                lambda x: x.shift(-period) / x - 1)
        return self.price_detail

    def normalize_date(self, stock_df):
        import datetime
        if len(stock_df) > 0:
            cols1 = stock_df.columns[np.array([type(x) for x in stock_df.iloc[0].values]) == datetime.date]
            cols2_cond1 = np.array([type(x) for x in stock_df.iloc[0].values]) == str
            cols2_cond2 = pd.DataFrame(
                [stock_df.columns.str.contains(x) for x in ['date', 'day', 'Date', 'Day', '日期', '日', '时间']]
            ).any().values
            cols2 = stock_df.columns[cols2_cond1 & cols2_cond2]
            cols = np.append(cols1, cols2)
            if len(cols) > 0:
                for col in cols:
                    stock_df[col] = pd.to_datetime(stock_df[col])
        return stock_df

    @property
    def stock_df(self):
        if self.wencai_data:
            stock_df = query_iwencai(self.sentence)
        else:
            stock_df = getattr(self.api, self.sentence)(*self.dates)
        stock_df = self.normalize_date(stock_df)
        return stock_df

    @stock_df.setter
    def stock_df(self, sentence):
        if self.wencai_data:
            return query_iwencai(sentence)
        else:
            return getattr(self.api, sentence)(*self.dates)

    # @property
    # def stock_list(self):
    #     if self.lst_tp == 'df':
    #         return self.stock_df['thscode'].drop_duplicates().tolist()
    #     else:
    #         return self.price_detail['stock_code'].drop_duplicates().tolist()
    #
    # @stock_list.setter
    # def stock_list(self, lst_tp):
    #     return getattr(self, lst_tp)

    def cal_raw_data(self, stock_df, **kwargs):
        raise NotImplementedError

    def map_data(self, x, *args):
        raise NotImplementedError

    def generate_raw_data(self, *args, **kwargs):
        """

        :param args: 超参数调整
        :param kwargs: 其他参数入参，如起止日期等，用于self.cal_raw_data的计算
        :return: 生成因子score
        """
        stock_df = self.cal_raw_data(self.stock_df, **kwargs)
        if self.map_tp == 'series':
            stock_df['score'] = stock_df['score'].apply(self.map_data, args=args)
        else:
            stock_df['score'] = stock_df.apply(self.map_data, axis=1, args=args)
        return stock_df

    def get_decay_score(self, stock_df):
        score = stock_df.set_index(['stock_code', 'date']).unstack([-2])
        score_new = score.fillna(0)
        for ll in range(1, len(score_new)):
            score_new.iloc[ll] = score_new.iloc[ll - 1] * 0.8 + score_new.iloc[ll]
        return score_new.stack([-2]).reset_index()

    def get_next_day(self, stock_df):
        price_detail = self.price_detail['close'].unstack().reset_index().rename(columns={'index': 'stock_code'})

        # stock_df = price_detail[['stock_code']]
        # stock_df['date'] = pd.date_range(price_detail.columns[1],
        #                                  price_detail.columns[-1],
        #                                  freq='5H').normalize()[:5166]

        tmp = stock_df.merge(price_detail, on=['stock_code'], how='left')
        tmp['date_new'] = tmp['date']
        tmp = tmp.set_index(['stock_code', 'date_new']).T
        dates, codes = tmp.index, tmp.columns
        tmp_new1 = tmp.copy()
        tmp_new2 = pd.DataFrame(data=np.array(list(dates) * len(codes)).reshape(len(codes), len(dates)).T,
                                index=dates, columns=codes)
        tmp_new2.loc['date'] = tmp_new1.loc['date']
        tmp_new = tmp_new1.where(tmp_new1.isna(), tmp_new2).where(~tmp_new1.isna(), pd.NaT)

        # tmp_new[('300851.SZ', pd.to_datetime('2022-01-02'))] = tmp_new[tmp_new.columns[4]]
        # tmp_new.loc['date'][[('300851.SZ', pd.to_datetime('2022-01-02'))]] = pd.to_datetime('2022-01-02')
        def get_first_date(x):
            return x[(x > x[0]) & (~x.isna())][0] if len(x[(x > x[0]) & (~x.isna())]) > 0 else np.nan

        next_day = tmp_new.apply(get_first_date).rename('date').reset_index()
        return next_day

    def get_stock_next_ret(self, stock_df_old, codes=None, decay=False):
        stock_df = stock_df_old.copy()
        if decay:
            stock_df = self.get_decay_score(stock_df)
        if codes:
            stock_df = stock_df[stock_df['stock_code'].isin(codes)]
        next_day = self.get_next_day(stock_df[['stock_code', 'date']])
        next_day.columns = ['stock_code', 'date', 'date_new']
        stock_df = stock_df.merge(next_day, on=['stock_code', 'date'], how='inner')[
            ['stock_code', 'date_new', 'score']].rename(columns={'date_new': 'date'})
        stock_df = stock_df[~stock_df['date'].isna()]
        ret = self.get_type_ret()
        df_ret = stock_df.merge(ret, on=['stock_code', 'date'], how='left')
        return df_ret

    def get_next_index(self, df_ret, name):
        def get_index(df):
            s1 = df.count()
            s2 = df.where(df > 0, np.nan).count() / s1
            s3 = df.mean()
            s = pd.concat({'样本数': s1, '胜率': s2, '均值': s3})
            return s

        df_ret_index = df_ret.groupby(['score'])[df_ret.columns[3:]].apply(get_index).stack([-2]).reset_index()
        count_df = df_ret_index['score'].describe()
        df_ret_index.to_csv(f'{name}_df_ret_index.csv')
        count_df.to_csv(f'{name}_count_df.csv')
        return df_ret_index, count_df

    def backtest(self, stock_df):
        score = stock_df.drop_duplicates().set_index(['stock_code', 'date'])['score'].unstack([-2])
        score_new = score.fillna(0)
        for i in range(1, len(score_new)):
            score_new.iloc[i] = score_new.iloc[i - 1] * 0.8 + score_new.iloc[i]
        index_ret = get_price('000300.SH',
                              self.dates[0],
                              self.dates[1],
                              '1d',
                              ['close'],
                              fq='pre')['close'].pct_change().fillna(0)
        backtest(score_new, self.price_detail['close'].unstack([-2]).pct_change().fillna(0), index_ret, self.__class__.__name__)


bg = BaseGenerator(sentence='', dates=['2018-01-01', '2023-05-19'])
bg.get_price()


# %%重大合同
# finished
class ZDHT(BaseGenerator):
    def __init__(self, dates):
        super(ZDHT, self).__init__(sentence="{start_date}以来，重大合同发布时间，重大合同金额", dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '股票简称', '重大合同发布时间']
        self.map_tp = 'series'

    @staticmethod
    def get_total_income(stock_df):
        total_income = pd.DataFrame(columns=['股票代码', '股票简称', '营业总收入'])
        for year in stock_df['last_year'].drop_duplicates().tolist():
            sub_total_income = query_iwencai("%s年年报，营业总收入" % year)
            sub_total_income['last_year'] = year
            total_income = total_income.append(sub_total_income)
        return total_income

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[stock_df['重大合同发布时间'] <= self.dates[1]]
        stock_df = stock_df.groupby(self.col)['重大合同金额'].sum().reset_index()
        stock_df['last_year'] = (pd.to_datetime(stock_df['重大合同发布时间']) - pd.DateOffset(years=1)).apply(lambda x: x.year)
        total_income = self.get_total_income(stock_df)
        stock_df = stock_df.merge(total_income[['股票代码', 'last_year', '营业总收入']],
                                  how='left', on=['股票代码', 'last_year'])
        stock_df['score'] = round(stock_df['重大合同金额'] / stock_df['营业总收入'], 2)
        return stock_df.copy()

    def map_data(self, x, *args):
        """

        :param x:
        :param args: 默认值
        :return:
        """
        x1, x2, y1, y2, y3 = args
        if x > x1:
            tmp_r = y1
        elif x > x2:
            tmp_r = y2
        else:
            tmp_r = y3
        return tmp_r

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data(0.2, 0.1, 2, 1, 0.5)
        stock_df = stock_df[['股票代码', '重大合同发布时间', 'score']]
        stock_df.columns = ['stock_code', 'date', 'score']
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        self.get_price(price_df=bg.price_detail, stock_detail=bg.stock_detail, stock_list=bg.stock_list)
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%定向增发
# finished
class DXZF(BaseGenerator):
    def __init__(self, dates):
        """
        注：get_dxzf是一个取数sql
        query = '''
        select b.thscode
              ,a.F002V_STK240
              ,a.F086D_STK240
              ,a.F013N_STK240
              ,a.F011N_STK240
              ,a.F007V_STK240
            --   ,a.f020v_stk240
              ,a.f077d_stk240
              ,a.f078d_stk240
              ,a.f080d_stk240
              ,a.f081d_stk240
        from STK240 a, PUB205 b
        where a.isvalid = 1
        and b.F019V_PUB205 = a.ZQID_STK240
        and a.CTIME >= '2018-01-01'
        and (b.thscode like '%SZ' or b.thscode like '%SH')
        -- and a.f081d_stk240 is not null
        -- and b.thscode = '300094.SZ'
        '''
        :param dates:
        """
        super(DXZF, self).__init__(sentence='get_dxzf', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'f002v_stk240', 'f077d_stk240', 'f078d_stk240', 'f080d_stk240', 'f081d_stk240']

    def get_f007v_stk240(self, x):
        mxgm = re.compile('易方达|富国|兴全|睿远|景顺|中欧|交银|信达|工银|汇添富|嘉实|广发|南方|鹏华').search(x)
        mxsm = re.compile(
            '高毅|高瓴|景林|重阳|源乐晟|淡水泉|东方港湾|红杉资本|正心谷|基石|天津礼仁|珠海煦远鼎峰|千合资本').search(x)
        zmwz = re.compile('高盛|摩根大通|瑞银|摩根士丹利|美林|巴黎银行').search(x)
        if mxgm is not None:
            tmp_r = 'mxgm'
        elif mxsm is not None:
            tmp_r = 'mxsm'
        elif zmwz is not None:
            tmp_r = 'zmwz'
        else:
            tmp_r = 'other'
        return tmp_r

    def get_dxzf_stage(self, stock_df):
        """
        'thscode', 'f002v_stk240', 'f086d_stk240', 'f013n_stk240',
        'f011n_stk240', 'f007v_stk240', 'f077d_stk240', 'f078d_stk240',
        'f080d_stk240', 'f081d_stk240'
        :param stock_df:
        :param kwargs:
        :return:
        """
        stock_df = stock_df.groupby(self.col).agg({'f013n_stk240': 'sum',
                                                   'f011n_stk240': 'min',
                                                   'f007v_stk240': ''.join,
                                                   'f086d_stk240': 'max'}).reset_index()
        stock_df['f007v_stk240'] = stock_df['f007v_stk240'].apply(self.get_f007v_stk240)
        cols = ['f077d_stk240', 'f078d_stk240', 'f080d_stk240', 'f081d_stk240']
        cols_step = ['董事会通过', '股东大会通过', '发审委通过', '证监会核准']
        zips = dict(zip(['thscode', 'f013n_stk240', 'f011n_stk240', 'f007v_stk240', 'f086d_stk240', 'f002v_stk240'],
                        ['stock_code', 'amount', 'price', 'class', 'date', 'step']))
        stock_df_new = stock_df.rename(columns=zips)
        for i, col in enumerate(cols):
            stock_df_new = stock_df_new[~stock_df_new[col].isna()]
            stock_df_new.loc[stock_df_new[col] != stock_df_new['date'], 'date'] = stock_df_new[col]
            stock_df_new.loc[stock_df_new[col] != stock_df_new['date'], 'step'] = cols_step[i]
        return stock_df_new

    def get_total_equity(self):
        zcfzb_his = self.api.get_balance_sheet_data(str(tuple(self.stock_list)), self.dates[0], self.dates[1])
        total_equity = zcfzb_his[['code', 'declare_date', 'report_date', 'total_equity']]
        total_equity.columns = ['stock_code', 'declare_date', 'report_date', 'total_equity']
        total_equity['declare_date'] = total_equity['declare_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        total_equity['report_date'] = total_equity['report_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        total_equity_end = total_equity.groupby(['stock_code', 'declare_date'])['report_date'].max().reset_index()
        total_equity = get_df_union_col_query(total_equity, total_equity_end, total_equity_end.columns)
        return total_equity[['stock_code', 'declare_date', 'total_equity']]

    def cal_raw_data(self, stock_df, **kwargs):
        """
        'thscode', 'f002v_stk240', 'f086d_stk240', 'f013n_stk240',
        'f011n_stk240', 'f007v_stk240', 'f077d_stk240', 'f078d_stk240',
        'f080d_stk240', 'f081d_stk240'
        :param stock_df:
        :param kwargs:
        :return:
        """
        stock_df_new = self.get_dxzf_stage(stock_df)
        total_equity = self.get_total_equity()
        print(stock_df_new.columns)
        print(total_equity.columns)
        stock_df_new = stock_df_new.merge(total_equity, how='left', on='stock_code')
        stock_df_new['gap'] = (pd.to_datetime(stock_df_new['date'])
                               - pd.to_datetime(stock_df_new['declare_date'])).apply(lambda x: x.days)
        stock_df_new = stock_df_new[stock_df_new['gap'] >= 0]

        stock_df_new['gap_rank'] = stock_df_new.groupby(['stock_code', 'date'])['gap'].rank()
        stock_df_new = stock_df_new[stock_df_new['gap_rank'] == 1]
        stock_df_new['amount_eql'] = round(stock_df_new['amount'] / stock_df_new['total_equity'] * 10000, 2)
        stock_df_new['date'] = pd.to_datetime(stock_df_new['date'])
        stock_df_new['declare_date'] = pd.to_datetime(stock_df_new['declare_date'])
        price_detail_nfq = self.get_nfq_prev_close(stock_list=stock_df_new['stock_code'].drop_duplicates().tolist())
        stock_df_new = stock_df_new.merge(price_detail_nfq, on=['stock_code', 'date'], how='left')
        return stock_df_new

    def map_data(self, x, *args):
        """

        :param x:
        :param args: 默认值
        :return:
        """
        x01, x02, y01, y02, y03 = args[0]
        y11, y12, y13 = args[1]
        x21, x22, x23, y21, y22, y23, y24 = args[2]
        y31, y32, y33, y34, y35 = args[3]

        if x[14] > x01:
            zfje = y01
        elif (x[14] >= x02) & (x[14] <= x01):
            zfje = y02
        else:  # x[14] < 0.02:
            zfje = y03

        if (x[8] == 'mxgm') | (x[8] == 'zmwz'):
            zfdx = y11
        elif x[8] == 'mxsm':
            zfdx = y12
        else:
            zfdx = y13

        if x[7] / x[15] <= x21:
            zfjg = y21
        elif (x[7] / x[15] > x21) & (x[7] / x[15] < x22):
            zfjg = y22
        elif (x[7] / x[15] >= x22) & (x[7] / x[15] <= x23):
            zfjg = y23
        else:
            zfjg = y24

        if x[1] in ['董事会通过', '已实施']:
            fxpf = y31
        elif x[1] == '股东大会通过':
            fxpf = y32
        elif x[1] in ['股东大会未通过', '发审委未通过', '到期失效', '停止实施', '发行失败']:
            fxpf = y33
        elif x[1] == '发审委通过':
            fxpf = y34
        else:  # x[5] in ['证监会核准', '证监会注册', '交易所审核通过']
            fxpf = y35

        return round(fxpf * (zfje + zfdx + zfjg) / 3, 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data([0.08, 0.02, 2, 1, 0.5],
                                          [1, 2, 0],
                                          [0.8, 1.1, 1.5, -1, 0.5, 2, 0],
                                          [1, 0.1, -1, 0.2, 0.5])
        stock_df = stock_df[['stock_code', 'date', 'score']]
        self.get_price(price_df=bg.price_detail, stock_detail=bg.stock_detail, stock_list=bg.stock_list)
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%股權激勵
class GQJL_HIS(BaseGenerator):
    def __init__(self, dates):
        """
        注：get_gqjl是一个取数sql
        query = '''
        select thscode
              ,date
              ,f008n_stk313
        from
        (
        select b.thscode
              ,cast(a.ctime as date) as date
              ,a.F008N_STK313
              ,a.f004d_stk313
              ,row_number() over(partition by thscode, cast(a.ctime as date) order by f004d_stk313) as rw
        from stk313 a, PUB205 b
        where a.isvalid = 1
        and b.F019V_PUB205 = a.ZQID_STK313
        and a.CTIME >= '2018-01-01'
        and b.F003V_PUB205 = 'A股'
        -- and b.thscode = '002765.SZ'
        and a.type_stk313 = 1
        order by DECLAREDATE_STK313
        ) a
        where rw = 1
        '''
        :param dates:
        """
        super(GQJL_HIS, self).__init__(sentence='get_equity_incentive_data', dates=dates)
        self.wencai_data = False
        self.col = ['stock_code', 'declare_date', '激励总数占当时总股本比例', '初始行权价']
        # self.stock_df_new = self.cal_raw_data(self.stock_df)

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col]
        stock_df.columns = ['stock_code', 'date', 'jlzs_ratio', 'xq_price']
        next_day = self.get_next_day(stock_df[['stock_code', 'date']])
        next_day.columns = ['stock_code', 'date', 'date_new']
        stock_df = stock_df.merge(next_day, on=['stock_code', 'date'], how='inner')[
            ['stock_code', 'date_new', 'jlzs_ratio', 'xq_price']].rename(columns={'date_new': 'date'})
        stock_df = stock_df[~stock_df['date'].isna()]
        return stock_df.groupby(['stock_code', 'date']).agg({'jlzs_ratio': 'sum', 'xq_price': 'mean'}).reset_index()


class STK313(BaseGenerator):
    def __init__(self, dates):
        """
        注：get_gqjl是一个取数sql
        query = '''
        select thscode
              ,date
              ,f008n_stk313
        from
        (
        select b.thscode
              ,cast(a.ctime as date) as date
              ,a.F008N_STK313
              ,a.f004d_stk313
              ,row_number() over(partition by thscode, cast(a.ctime as date) order by f004d_stk313) as rw
        from stk313 a, PUB205 b
        where a.isvalid = 1
        and b.F019V_PUB205 = a.ZQID_STK313
        and a.CTIME >= '2018-01-01'
        and b.F003V_PUB205 = 'A股'
        -- and b.thscode = '002765.SZ'
        and a.type_stk313 = 1
        order by DECLAREDATE_STK313
        ) a
        where rw = 1
        '''
        :param dates:
        """
        super(STK313, self).__init__(sentence='get_gqjl', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'f008n_stk313']
        # self.stock_df_new = self.cal_raw_data(self.stock_df)

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df.columns = ['stock_code', 'date', 'yjzb']
        stock_df['date'] = pd.to_datetime(stock_df['date']) + pd.DateOffset(days=1)
        next_day = self.get_next_day(stock_df[['stock_code', 'date']])
        next_day.columns = ['stock_code', 'date', 'date_new']
        stock_df = stock_df.merge(next_day, on=['stock_code', 'date'], how='inner')[
            ['stock_code', 'date_new', 'yjzb']].rename(columns={'date_new': 'date'})
        stock_df = stock_df[~stock_df['date'].isna()]
        return stock_df


class GQJL(BaseGenerator):
    def __init__(self, dates):
        super(GQJL, self).__init__(dates=dates)
        gqjl_his = GQJL_HIS(dates)
        gqjl_his.get_price(stock_list=gqjl_his.stock_df['stock_code'].drop_duplicates().tolist())
        # gqjl_his.get_price(price_df=bg.price_detail, stock_detail=bg.stock_detail, stock_list=bg.stock_list)
        stock_df1 = gqjl_his.cal_raw_data(gqjl_his.stock_df)
        stk313 = STK313(dates)
        stk313.get_price(stock_list=stk313.stock_df['thscode'].drop_duplicates().tolist())
        # stk313.get_price(price_df=bg.price_detail, stock_detail=bg.stock_detail, stock_list=bg.stock_list)
        stock_df2 = stk313.cal_raw_data(stk313.stock_df)
        self.stock_df_new = stock_df1.merge(stock_df2, on=['stock_code', 'date'], how='left').merge(
            bg.price_detail, on=['stock_code', 'date'], how='left')

    def map_data(self, x, *args):

        if x[4] > 50:
            yjcn = 2
        elif (x[4] >= 30) & (x[4] <= 50):
            yjcn = 1
        elif (x[4] > 0) & (x[4] < 30):
            yjcn = 0.5
        else:
            yjcn = 0

        if x[2] > 1.5:
            jlje = 2
        elif (x[2] >= 0.5) & (x[2] <= 1.5):
            jlje = 1
        else:
            jlje = 0

        if x[3] / x[6] <= 0.6:
            jljg = 2
        else:
            jljg = 0

        return round((yjcn + jlje + jljg) / 3, 2)

    def __call__(self, *args, **kwargs):
        self.stock_df_new['score'] = self.stock_df_new.apply(self.map_data, axis=1)
        self.get_price(stock_list=self.stock_df_new['stock_code'].drop_duplicates().tolist())
        # self.get_price(price_df=bg.price_detail, stock_detail=bg.stock_detail, stock_list=bg.stock_list)
        df_ret = self.get_stock_next_ret(self.stock_df_new)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, self.stock_df_new


# %%股份回購
# finished
class GFHG(BaseGenerator):
    def __init__(self, dates):
        """
        注：get_gfhg是一个取数sql
        query = '''
        select b.thscode
              ,a.DECLAREDATE_STK301
              ,a.F002V_STK301
              ,a.F004N_STK301 as price_hlimit
              ,nvl(F023N_STK301, a.F010N_STK301) as amount_llimit
              ,a.F016D_STK301
              ,a.F017D_STK301
              ,a.F018D_STK301
              ,a.F019D_STK301
              ,a.F020D_STK301
        from stk301 a, PUB205 b
        where a.isvalid = 1
        and b.F019V_PUB205 = a.ZQID_STK301
        and a.CTIME >= '2018-01-01'
        and b.F003V_PUB205 = 'A股'
        -- and a.f081d_stk240 is not null
        -- and b.thscode = '002362.SZ'
        -- order by DECLAREDATE_STK313
        '''
        :param dates:
        """
        super(GFHG, self).__init__(sentence='get_gfhg', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'price_hlimit', 'amount_llimit', 'f002v_stk301']

    def cal_raw_data(self, stock_df, **kwargs):
        self.get_price(stock_list=self.stock_df['thscode'].drop_duplicates().tolist())
        col_dict = {'f016d_stk301': '董事会预案',
                    'f017d_stk301': '股东大会通过',
                    'f018d_stk301': '实施回购',
                    'f019d_stk301': '回购完成',
                    'f020d_stk301': '回购股份已注销',
                    'declaredate_stk301': '停止回购'}
        cols = ['董事会预案', '股东大会未通过', '股东大会通过', '实施回购', '回购完成', '回购股份已注销', '停止回购']
        col_dict2 = dict(zip(list(cols), range(len(cols))))
        stock_df_new = stock_df.set_index(self.col).stack().reset_index()
        stock_df_new.columns = self.col + ['step', 'date']
        stock_df_new['step'] = stock_df_new['step'].apply(lambda x: col_dict[x])
        stock_df_new = stock_df_new[stock_df_new['step'] != '股东提议']
        stock_df_new.loc[stock_df_new['f002v_stk301'] == '股东大会未通过', 'step'] = '股东大会未通过'
        stock_df_new = stock_df_new.rename(columns={'thscode': 'stock_code'})

        def get_last_action(df):
            df_new = df.set_index(['stock_code', 'date'])
            codes_date = df.groupby(['stock_code', 'date'])['step'].count()
            codes_date = codes_date[codes_date > 1].index
            df1 = df_new[df_new.index.isin(codes_date)].reset_index()
            df2 = df_new[~df_new.index.isin(codes_date)]
            df10 = df1[(df1['step'] == '回购完成') & (df1['f002v_stk301'] == '回购完成')]
            df11_code = list(set(df1['stock_code']) - set(df10['stock_code']))
            df11 = df1[df1['stock_code'].isin(df11_code)]
            df11['count'] = df11['step'].apply(lambda x: col_dict2[x])
            df11 = df11.groupby(['stock_code', 'date']).apply(
                lambda x: x[x['count'] == max(x['count'])]).reset_index(drop=True)
            del df11['count']
            print(df1[df1['stock_code'] == '603801.SH'].sort_values(['date']))
            df_new = pd.concat([df10, df11, df2.reset_index()])
            df_new = df_new.set_index(['stock_code', 'date'])
            codes_date = df.groupby(['stock_code', 'date'])['step'].count()
            codes_date = codes_date[codes_date == 1].index
            return df_new[df_new.index.isin(codes_date)].reset_index()

        stock_df_new = get_last_action(stock_df_new)
        stock_df_new = stock_df_new.merge(self.price_detail, on=['stock_code', 'date'], how='left')
        print(stock_df_new[stock_df_new['stock_code'] == '603801.SH'].sort_values(['date']))
        return stock_df_new

    def map_data(self, x, *args):
        """

        :param x:
        :param args: args = ([1.5, 1, 2, 1, 0.1], [3, 1, 2, 1.5, 1], [1, 0.1, 0.5, -0.2, -0.5])
        :return:
        """
        x01, x02, y01, y02, y03 = args[0]
        x11, x12, y11, y12, y13 = args[1]
        y21, y22, y23, y24, y25, y26 = args[2]

        if x[2] / x[7] > x01:
            hgjg = y01
        elif (x[2] / x[7] >= x02) & (x[2] / x[7] <= x01):
            hgjg = y02
        else:
            hgjg = y03

        if x[3] > x11:
            hgzj = y11
        elif (x[3] >= x12) & (x[3] <= x11):
            hgzj = y12
        else:
            hgzj = y13

        if x[5] == '董事会预案':
            fx = y21
        elif x[5] == '股东大会通过':
            fx = y22
        elif x[5] == '实施回购':
            fx = y23
        elif x[5] == '股东大会未通过':
            fx = y24
        elif x[5] == '停止回购':
            fx = y25
        else:
            fx = y26

        return round((hgjg + hgzj) / 2 * fx, 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data([1.5, 1, 2, 1, 0.1],
                                          [3, 1, 2, 1.5, 1],
                                          [1, 0.1, 0.5, -0.2, -0.5, 0])
        print(stock_df[stock_df['stock_code'] == '600337.SH'].sort_values(['date']))
        stock_df = stock_df[['stock_code', 'date', 'score']]
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%减持計劃
# finished
class JCJH(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(JCJH, self).__init__(sentence='{start_date}至{end_date}减持计划', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '增减持计划股东类别', '增减持计划变动方向', '增减持计划进度', '增减持计划最新公告日期',
                    '增减持计划首次公告日期', '增减持计划变动起始日期', '增减持计划变动截止日期',
                    '预计变动数量占总股本比例']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col]
        set_col = [self.col[0], self.col[1], self.col[3], self.col[8]]
        stack_col = [self.col[4], self.col[5], self.col[6], self.col[7]]
        col_dict = {'增减持计划首次公告日期': '减持计划',
                    '增减持计划变动起始日期': '减持实施',
                    '增减持计划最新公告日期': '减持完成',
                    '增减持计划变动截止日期': '减持完成'}
        stock_df_new = stock_df.set_index(set_col)[stack_col].stack().reset_index()
        stock_df_new.columns = [self.col[0], self.col[1], self.col[3], self.col[8], 'step', 'date']
        stock_df_new = stock_df_new[
            (~((stock_df_new['增减持计划进度'] == '完成')
               & (stock_df_new['step'] == '增减持计划变动截止日期')))
            & (~((stock_df_new['增减持计划进度'] == '进行中')
                 & (stock_df_new['step'] == '增减持计划最新公告日期')))
            ]
        stock_df_new['step'] = stock_df_new['step'].apply(lambda x: col_dict[x])
        stock_df_new.loc[(stock_df_new['增减持计划进度'] == '停止实施')
                         & (stock_df_new['step'] == '减持完成'), 'step'] = '停止实施'
        stock_df_new.columns = ['stock_code', 'type', 'stage', 'ratio', 'step', 'date']
        stock_df_new = stock_df_new.groupby(['stock_code', 'date', 'type', 'stage', 'step'])['ratio'].sum().reset_index()
        return stock_df_new[['stock_code', 'type', 'stage', 'ratio', 'step', 'date']]

    def map_data(self, x, *args):

        if x[4] == '减持计划':
            ggjd = 1
        elif x[4] == '减持实施':
            ggjd = 0.1
        elif x[4] == '减持完成':
            ggjd = -0.2
        elif x[4] == '停止实施':
            ggjd = -0.5
        else:
            ggjd = 0

        type_list = x[1].split(',')

        if '实际控制人' in type_list:
            gdlb = -2
        elif '高管' in type_list:
            gdlb = -1.5
        elif ('员工持股计划' in type_list) | ('持股5%以上一般股东' in type_list):
            gdlb = -1
        else:
            gdlb = -0.5

        if x[3] > 1:
            hgzj = -2
        elif (x[3] >= 0.5) & (x[3] <= 1):
            hgzj = -1
        else:
            hgzj = -0.5

        return round(ggjd * (gdlb + hgzj) / 2, 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%增持計劃
# finished
class ZCJH(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(ZCJH, self).__init__(sentence='{start_date}至{end_date}增持计划', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '增减持计划股东类别', '增减持计划变动方向', '增减持计划进度', '增减持计划最新公告日期',
                    '增减持计划首次公告日期', '增减持计划变动起始日期', '增减持计划变动截止日期',
                    '预计变动数量占总股本比例']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col]
        set_col = [self.col[0], self.col[1], self.col[3], self.col[8]]
        stack_col = [self.col[4], self.col[5], self.col[6], self.col[7]]
        col_dict = {'增减持计划首次公告日期': '增持计划',
                    '增减持计划变动起始日期': '增持实施',
                    '增减持计划最新公告日期': '增持完成',
                    '增减持计划变动截止日期': '增持完成'}
        stock_df_new = stock_df.set_index(set_col)[stack_col].stack().reset_index()
        stock_df_new.columns = [self.col[0], self.col[1], self.col[3], self.col[8], 'step', 'date']
        stock_df_new = stock_df_new[
            (~((stock_df_new['增减持计划进度'] == '完成')
               & (stock_df_new['step'] == '增减持计划变动截止日期')))
            & (~((stock_df_new['增减持计划进度'] == '进行中')
                 & (stock_df_new['step'] == '增减持计划最新公告日期')))
            ]
        stock_df_new['step'] = stock_df_new['step'].apply(lambda x: col_dict[x])
        stock_df_new.loc[(stock_df_new['增减持计划进度'] == '停止实施')
                         & (stock_df_new['step'] == '增持完成'), 'step'] = '停止实施'
        stock_df_new.columns = ['stock_code', 'type', 'stage', 'ratio', 'step', 'date']
        stock_df_new = stock_df_new.groupby(['stock_code', 'date', 'type', 'stage', 'step'])['ratio'].sum().reset_index()
        return stock_df_new[['stock_code', 'type', 'stage', 'ratio', 'step', 'date']]

    def map_data(self, x, *args):

        if x[4] == '增持计划':
            ggjd = 1
        elif x[4] == '增持实施':
            ggjd = 0.1
        elif x[4] == '增持完成':
            ggjd = -0.2
        elif x[4] == '停止实施':
            ggjd = -0.5
        else:
            ggjd = 0

        type_list = x[1].split(',')

        if '实际控制人' in type_list:
            gdlb = -2
        elif '高管' in type_list:
            gdlb = -1.5
        elif ('员工持股计划' in type_list) | ('持股5%以上一般股东' in type_list):
            gdlb = -1
        else:
            gdlb = -0.5

        if x[3] > 1:
            hgzj = -2
        elif (x[3] >= 0.5) & (x[3] <= 1):
            hgzj = -1
        else:
            hgzj = -0.5

        return round(ggjd * (gdlb + hgzj) / 2, 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%限售解禁
# finished
class XSJJ(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(XSJJ, self).__init__(sentence='{start_date}至{end_date}解禁股成本', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '解禁日期', '解禁成本', '解禁比例']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col]
        stock_df.columns = ['stock_code', 'date', 'cost', 'ratio']
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        stock_df = stock_df.merge(self.price_detail, on=['stock_code', 'date'], how='left')
        return stock_df

    def map_data(self, x, *args):
        # x01, x02, y01, y02, y03 = args[0]
        # x11, x12, y11, y12, y13 = args[1]

        if x[3] > 10:
            jjbl = -2
        elif (x[3] >= 5) & (x[3] <= 10):
            jjbl = -1
        else:
            jjbl = -0.5

        if x[4] / x[2] > 2:
            jjcb = -2
        elif (x[4] / x[2] >= 1) & (x[4] / x[2] <= 2):
            jjcb = -1
        else:
            jjcb = 0
        return round((jjbl + jjcb) / 2, 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%实际控制人变更
# finished
class SKRBG(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(SKRBG, self).__init__(sentence='{start_date}至{end_date}实控人变更, 上市时间小于{end_date}', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '股票简称', '变更前实控人', '变更后实控人', '实控人变更公告日期', '实控人变更截止日期']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col].rename(columns={'股票代码': 'stock_code', '实控人变更公告日期': 'date'})
        stock_df = stock_df[stock_df['date'] < self.dates[1]]
        return stock_df

    def map_data(self, x, *args):
        if (len(x[3].split('国有')) > 1) | (len(x[3].split('国资')) > 1) | (len(x[3].split('管理委员会')) > 1):
            score = 2
        else:
            score = 0.5

        return score * 0.5

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %%分红派息
# finished
class FHPX(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select a.*
              ,b.F010_JGS060
        from
            (
            select b.thscode
                  ,a.F001D_STK441
                  ,a.F008N_STK441
                  ,a.F047D_STK441
                  ,a.F002D_STK441
                  ,a.F004D_STK441
                  ,a.F006D_STK441
                  ,a.F009D_STK441
                  ,a.F012N_STK441
                  ,nvl(a.F010N_STK441, 0) + nvl(a.F011N_STK441, 0) as F010NF011N_STK441
                  ,a.F036V_STK441
                  ,a.F037D_STK441
            from STK441 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK441
              and a.F002D_STK441 <= '{end_date}'
              and a.F002D_STK441 >= '{start_date}'
              and b.F003V_PUB205 = 'A股'
              and ((a.F012N_STK441 is not null) or (a.F010N_STK441 is not null) or (a.F011N_STK441 is not null))
            ) a
        left join
            (
            select b.thscode
                  ,a.ENDDATE_JGS060
                  ,a.F010_JGS060
            from JGS060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE_JGS060
              and b.F003V_PUB205 = 'A股'
            ) b
        on a.thscode = b.thscode and a.F001D_STK441 = b.ENDDATE_JGS060
        order by thscode, f001d_stk441
        '''
        :param dates:
        """
        super(FHPX, self).__init__(sentence='get_fhpx', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'fhl', 'szl', 'f036v_stk441', 'f037d_stk441']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[stock_df['thscode'].notnull()]
        stock_df[['f012n_stk441', 'f010nf011n_stk441']] = stock_df[['f012n_stk441', 'f010nf011n_stk441']].fillna(0)
        stock_df['fhl'] = round(stock_df['f012n_stk441'] / stock_df['f010_jgs060'] / 10, 3)
        stock_df['szl'] = round(stock_df['f010nf011n_stk441'] / 10, 3)

        stack_col = ['f047d_stk441', 'f002d_stk441', 'f004d_stk441', 'f006d_stk441']
        steps = ['预披露', '董事会预案', '股东大会预案', '实施方案']
        col_dict = dict(zip(stack_col, steps))

        stock_df.loc[stock_df['f047d_stk441'].isna(), 'f047d_stk441'] = stock_df['f002d_stk441']
        stock_df = stock_df.set_index(self.col)[stack_col].stack().reset_index()
        stock_df.columns = self.col + ['step', 'date']
        stock_df['step'] = stock_df['step'].apply(lambda x: col_dict[x])
        stock_df.loc[(stock_df['f036v_stk441'] == '取消分红')
                     & (stock_df['date'] == stock_df['f037d_stk441']),
                     'step'] = '取消分红'
        stock_df = stock_df[((stock_df['f036v_stk441'] == steps[0]) & (stock_df['step'].isin(steps[:1])))
                            | ((stock_df['f036v_stk441'] == steps[1]) & (stock_df['step'].isin(steps[:2])))
                            | ((stock_df['f036v_stk441'] == steps[2]) & (stock_df['step'].isin(steps[:3])))
                            | ((stock_df['f036v_stk441'] == steps[3]) & (stock_df['step'].isin(steps[:4])))
                            | ((stock_df['f036v_stk441'] == '取消分红') & (stock_df['date'] <= stock_df['f037d_stk441']))]
        return stock_df.rename(columns={'thscode': 'stock_code'})

    def map_data(self, x, *args):
        fhjd = 0
        fhbl = 0
        szbl = 0

        if x[5] == '预披露':
            fhjd = 1
        elif x[5] == '董事会预案':
            fhjd = 0.1
        elif x[5] == '股东大会预案':
            fhjd = 0.1
        elif x[5] == '实施方案':
            fhjd = 0.2
        else:
            fhjd = 0

        if x[1] > 0.02:
            fhbl = 1.5
        elif (x[1] <= 0.02) & (x[1] >= 0.005):
            fhbl = 0.5
        elif (x[1] < 0.005) & (x[1] > 0):
            fhbl = 0.1
        else:
            fhbl = 0

        if x[2] > 1:
            szbl = 2
        elif (x[2] <= 1) & (x[2] >= 0.5):
            szbl = 0.5
        elif (x[2] < 0.5) & (x[2] > 0):
            szbl = 0.1
        else:
            szbl = 0

        return round(fhjd * max(fhbl, szbl), 2)

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 业绩预告 && 业绩快报 && 业绩报告
# unfinished
class YJBG(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
            select a.*
                  ,b.F045N_YB026
            from
            (
            select b.thscode
                  ,a.declaredate_stk428
                  ,a.F001D_STK428
                  ,substring(cast(a.F001D_STK428 as text), 1, 4) as end_year
                  ,a.F010N_STK428*10000 as F010N_STK428
                  ,a.F017N_STK428
                  ,round(a.F010N_STK428*10000/a.F017N_STK428*100-100,1)
                  ,'业绩预告' as type
                --   ,a.
            from STK428 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK428
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk428 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk435
                  ,a.enddate_stk435
                  ,substring(cast(a.enddate_stk435 as text), 1, 4) as end_year
                  ,a.F004N_STK435
                  ,a.F015N_STK435
                  ,round(a.F004N_STK435/a.F015N_STK435*100-100,1)
                  ,'业绩快报' as type
            from STK435 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK435
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk435 >= '20180101'
              and b.thscode is not null
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk060
                  ,a.ENDDATE
                  ,substring(cast(a.ENDDATE as text), 1, 4) as end_year
                  ,a.F002
                  ,a.F044N_STK060
                  ,nvl(round(a.F002/a.F044N_STK060 - 1, 2)*100, 0) as rat
                  ,'业绩报告' as type
                --   ,a.
            from STK060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk060 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'
              ) a
              left join
              (
              select thscode_yb026
                  ,enddate_yb026
                  ,substring(cast(enddate_yb026 as text), 1, 4) as end_year
                  ,F045N_YB026*1000000 as F045N_YB026
            from YB026
            where isvalid = 1
              and thscode_yb026 = '300033.SZ'
              ) b
            on a.thscode = b.thscode_yb026 and a.end_year = a.end_year
            '''
        :param dates:
        """
        super(YJBG, self).__init__(sentence='get_yjbg_factor', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'declaredate_stk428', 'score']

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df['declaredate_stk428'] = stock_df['declaredate_stk428'].astype(str)
        stock_df = stock_df[stock_df['declaredate_stk428'] < self.dates[1]]
        stock_df['declaredate_stk428'] = stock_df[['thscode', 'declaredate_stk428']].apply(self.get_tradeday, axis=1)
        stock_df['gap'] = stock_df['declaredate_stk428'].astype('datetime64[ns]') - stock_df['f001d_stk428'].astype(
            'datetime64[ns]')
        stock_df['rank'] = stock_df.groupby(['thscode', 'declaredate_stk428'])['gap'].rank(ascending=True)
        stock_df = stock_df[stock_df['rank'] == 1]
        stock_df['full_tag'] = stock_df['thscode'] + stock_df['declaredate_stk428'].astype(str)
        stock_df['yjbg_score'] = stock_df['score']

    def map_data(self, x, *args):
        # 预测数据最近2021财报，无法回测
        """
        select a.thscode
              ,a.declaredate_stk428
              ,a.F001D_STK428
              ,a.F010N_STK428
              ,a.F017N_STK428
              ,a.rat
              ,a.type
              ,case when a.F017N_STK428 > 0 and a.rat>100 then 2
                    when a.F017N_STK428 > 0 and a.rat>30 and a.rat<=100 then 1
                    when a.F017N_STK428 > 0 and a.rat>=0 and a.rat<=30 then 0.5

                    when a.F017N_STK428 < 0 and a.F010N_STK428>0 then 1.5

                    when a.F017N_STK428 > 0 and a.rat<-100 then -2
                    when a.F017N_STK428 > 0 and a.rat>=-100 and a.rat<-30 then -1
                    when a.F017N_STK428 > 0 and a.rat>=-30 and a.rat<0 then -0.5

                    when a.F017N_STK428 < 0 and a.F010N_STK428<0 and a.rat<=0 then -1
                    when a.F017N_STK428 < 0 and a.F010N_STK428<0 and a.rat>0 then -2
                    else 0 end as score

        from
        (
        select b.thscode
              ,a.declaredate_stk428
              ,a.F001D_STK428
              ,a.F010N_STK428*10000 as F010N_STK428
              ,a.F017N_STK428
              ,round(a.F010N_STK428*10000/(a.F017N_STK428+0.0001)*100-100,1) as rat
              ,'业绩预告' as type
            --   ,a.
        from STK428 a, PUB205 b
        where a.isvalid = 1
          and b.F014V_PUB205 = a.ORGID_STK428
          and b.F003V_PUB205 = 'A股'
          and a.declaredate_stk428 >= '20180101'
          and b.thscode is not null
        --   and a.f003v_stk428 like '%亏%'
        --   and b.thscode = '300033.SZ'

        union all

        select b.thscode
              ,a.declaredate_stk435
              ,a.enddate_stk435
              ,a.F004N_STK435
              ,a.F015N_STK435
              ,round(a.F004N_STK435/(a.F015N_STK435+0.0001)*100-100,1) as rat
              ,'业绩快报' as type
        from STK435 a, PUB205 b
        where a.isvalid = 1
          and b.F014V_PUB205 = a.ORGID_STK435
          and b.F003V_PUB205 = 'A股'
          and a.declaredate_stk435 >= '20180101'
          and b.thscode is not null
        --   and b.thscode = '300033.SZ'

        union all

        select b.thscode
              ,a.declaredate_stk060
              ,a.ENDDATE
              ,a.F002
              ,a.F044N_STK060
              ,nvl(round(a.F002/(a.F044N_STK060+0.0001) - 1, 2)*100, 0) as rat
              ,'业绩报告' as type
            --   ,a.
        from STK060 a, PUB205 b
        where a.isvalid = 1
          and b.F014V_PUB205 = a.COMCODE
          and b.F003V_PUB205 = 'A股'
          and a.declaredate_stk060 >= '20180101'
          and b.thscode is not null
        --   and a.f003v_stk428 like '%亏%'
        --   and b.thscode = '300033.SZ'
          ) a
        """

    def __call__(self, *args, **kwargs):
        stock_df = self.stock_df[self.col]
        stock_df.columns = ['stock_code', 'date', 'score']
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 股份冻结 & 股份质押
# unfinished
class DJZY(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select thscode
              ,mtime
              ,case when dj_rat>0.7 then -2*type_score
                    when (dj_rat>0.3) and (dj_rat<=0.7) then -0.5*type_score
                    when (dj_rat>0) and (dj_rat<=0.3) then 0*type_score
                    when f005=0 then 0.2*type_score
                    else 0 end as dj_score
              ,case when zy_rat>0.7 then -2*type_score
                    when (zy_rat>0.3) and (zy_rat<=0.7) then -0.5*type_score
                    when (zy_rat>0) and (zy_rat<=0.3) then 0*type_score
                    when f006=0 then 0.2*type_score
                    else 0 end as zy_score
        from
        (
        select thscode
              ,mtime
              ,f005
              ,round(f005/f003,2) as dj_rat
              ,f006
              ,round(f006/f003,2) as zy_rat
              ,case when f003/f057n_stk035 > 0.3 then '实际控制人'
                    when f014v_stk032 = '个人' and f003/f057n_stk035 < 0.05 then '高管'
                    else '其它' end as type
              ,case when f003/f057n_stk035 > 0.3 then 1
                    when f014v_stk032 = '个人' and f003/f057n_stk035 < 0.05 then 0.75
                    else 0.5 end as type_score
        from
        (
        select a.*
              ,row_number() over(partition by thscode, mtime order by gap desc) as rw
        from
        (
        select a.F002
              ,a.F003
              ,a.F005
              ,a.F006
              ,a.F014V_STK032
              ,cast(a.MTIME as date) as MTIME
              ,e.DECLAREDATE
              ,cast(e.DECLAREDATE as date)-cast(a.MTIME as date) as gap
              ,e.F057N_STK035
            --   ,a.F025V_STK032
            --   ,b.F007N_STK030
            --   ,a.VSEQ
            --   ,b.SEQ
            --   ,b.COMCODE
            --   ,c.ORGID_PUB203
            --   ,d.F014V_PUB205
              ,d.thscode
        from STK032 a, STK030 b, PUB203 c, PUB205 d, STK035 e
        where a.MTIME >= '2018-01-01'
          and ((a.f005 is not null) or (a.f006 is not null))
          and a.VSEQ = b.SEQ
          and b.COMCODE = c.ORGID_PUB203
          and c.ORGID_PUB203 = d.F014V_PUB205
          and c.ORGID_PUB203 = e.COMCODE
        --   and a.F013V_STK032='015014'
          and a.isvalid = 1
          and d.F003V_PUB205 = 'A股'
        --   and d.thscode = '000008.SZ'
          and d.thscode is not null
          and e.f057n_stk035 > 0
        order by a.MTIME
        ) a
        where gap <= 0
        ) a
        where rw = 1
        ) a'''
        :param dates:
        """
        super(DJZY, self).__init__(sentence='get_djzy', dates=dates)
        self.wencai_data = False

    def cal_raw_data(self, stock_df, **kwargs):
        pass

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 监管函
class JGH(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(JGH, self).__init__(sentence='{start_date}至今监管函', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '监管日期']
        self.map_tp = 'series'

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col].drop_duplicates()
        stock_df['score'] = -1.5
        stock_df.columns = ['stock_code', 'date', 'score']
        return stock_df

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 立案调查
class LADC(BaseGenerator):
    def __init__(self, dates):
        """

        :param dates:
        """
        super(LADC, self).__init__(sentence='{start_date}至今立案调查内容，立案调查对象', dates=dates)
        self.wencai_data = True
        self.col = ['股票代码', '立案调查时间', '立案调查对象']
        self.map_tp = 'series'

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df.groupby(self.col[:2])['立案调查对象'].agg(''.join).reset_index()
        stock_df.columns = ['stock_code', 'date', 'score']
        return stock_df

    def map_data(self, x, *args):

        bgs = re.compile('本公司|控制').search(x)
        gg = re.compile('高管').search(x)
        ybgd = re.compile('持股5').search(x)

        if bgs is not None:
            tmp_r = -2
        elif gg is not None:
            tmp_r = -1.5
        elif ybgd is not None:
            tmp_r = -1
        else:
            tmp_r = -0.5

        return tmp_r

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 年报非标
class NBFB(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select b.THSCODE
              ,a.DECLAREDATE
              ,a.F003
              ,case when a.F003 = 5 then -1.5
                    when a.F003 = 2 then -2
                    else 0 end as nbfb_score
        from STK037 a, PUB205 b
        where a.isvalid = 1
          and a.COMCODE = b.F014V_PUB205
          and a.DECLAREDATE >= '2018-01-01'
          and b.F003V_PUB205 = 'A股'
          and a.F003 in (2, 3, 4, 5)
        '''
        :param dates:
        """
        super(NBFB, self).__init__(sentence='get_nbfb', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'declaredate', 'nbfb_score']
        self.map_tp = 'series'

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[self.col]
        stock_df = stock_df[stock_df['declaredate'].notnull()]
        stock_df.columns = ['stock_code', 'date', 'score']
        return stock_df

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 披星戴帽
class PXDM(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select b.thscode
              ,a.DECLAREDATE_STK019
              ,a.F005N_STK019
              ,case when a.f005n_stk019 in (1,2,7) then -2
                    when a.f005n_stk019 in (3,4,6) then 2
                    end
                    as pxdm_score
        from STK019 a, PUB205 b
        where a.isvalid = 1
          and a.ZQID_STK019 = b.SECCODE_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.F005N_STK019 in (1,2,3,4,6,7)
          and declaredate_stk019>='2018-01-01'

        union all

        select b.thscode
              ,a.DECLAREDATE_STK019
              ,a.F005N_STK019
              ,case when a.f005n_stk019 in (1,2,7) then -2
                    when a.f005n_stk019 in (3,4,6) then 2
                    end
                    as pxdm_score
        from STK019 a, PUB205 b
        where a.isvalid = 1
          and a.ZQID_STK019 = b.SECCODE_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.F005N_STK019 in (1,2,3,4,6,7)
          and declaredate_stk019>='2018-01-01'
        '''
        :param dates:
        """
        super(PXDM, self).__init__(sentence='get_pxdm', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'declaredate_stk019', 'pxdm_score']
        self.map_tp = 'series'

    def cal_raw_data(self, stock_df, **kwargs):
        stock_df = stock_df[stock_df['declaredate_stk019'].notnull()]
        stock_df = stock_df[self.col]
        stock_df.columns = ['stock_code', 'date', 'score']
        return stock_df

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 诉讼仲裁
class SSZC(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select thscode
              ,first_d
              ,min(first_score) as first_score
              ,second_d
              ,min(second_score) as second_score
              ,end_d
              ,min(end_score) as end_score
        from
        (
        select thscode
              ,first_d
              ,case when first_d is not null then rat_score*first_score else null end as first_score
              ,second_d
              ,case when second_d is not null then rat_score*second_score else null end as second_score
              ,f003d_stk658 as end_d
              ,case when f003d_stk658 is not null then rat_score*end_score else null end as end_score
        from
        (
        select a.*
              ,b.DECLAREDATE_STK127
              ,b.F127N_STK127
              ,case when (a.f010n_stk658/b.f127n_stk127*10000*100)>=20 then -2
                    when (a.f010n_stk658/b.f127n_stk127*10000*100)>=5 then -1
                    when (a.f010n_stk658/b.f127n_stk127*10000*100)>0 then -0.5 else 0 end as rat_score
              ,case when (shenfen in ('原告','申请方') and first_r = '原告胜诉')
                    or (shenfen in ('被告','被申请方') and first_r = '被告胜诉') then -1
                    else 1 end as first_score
              ,case when (shenfen in ('原告','申请方') and second_r = '原告胜诉')
                    or (shenfen in ('被告','被申请方') and second_r = '被告胜诉') then -1
                    else 1 end as second_score
              ,case when (shenfen in ('原告','申请方') and end_r = '原告胜诉')
                    or (shenfen in ('被告','被申请方') and end_r = '被告胜诉') then -1
                    else 1 end as end_score
              ,row_number() over(partition by a.thscode, a.order_stk658 order by cast(declaredate_stk127 as date)-cast(f002d_stk658 as date) desc) as rw
        from
        (
        select a.*
              ,b.F003D_STK658
              ,b.jindu
              ,c.end_r
              ,d.first_d
              ,d.first_r
              ,e.second_d
              ,e.second_r
        from
        (
        select  b.thscode
               ,a.F010N_STK658
               ,a.ORDER_STK658
               ,c.f001v_pub201 as shenfen
               ,a.F002D_STK658
        from STK658 a, PUB205 b, PUB201 c
        where a.isvalid = 1
          and a.ORGID_STK658 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.f002d_stk658 >= '2018-01-01'
          and a.F004V_STK658 = c.CODE_PUB201
          and a.F004V_STK658 != '047005'
          and a.f010n_stk658 is not null
          and b.thscode is not null
        --   and b.thscode = '601377.SH'
          ) a
          left join
          (
        select  b.thscode
               ,a.f002d_stk658
               ,a.ORDER_STK658
               ,a.F003D_STK658
               ,c.f001v_pub201 as jindu
        from STK658 a, PUB205 b, PUB201 c
        where a.isvalid = 1
          and a.ORGID_STK658 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.f002d_stk658 >= '2018-01-01'
          and a.F008V_STK658 = c.CODE_PUB201
          and a.F004V_STK658 != '047005'
          and a.f010n_stk658 is not null
          and b.thscode is not null
          ) b
          on a.thscode=b.thscode and a.f002d_stk658=b.f002d_stk658 and a.ORDER_STK658=b.ORDER_STK658
          left join
          (
        select  b.thscode
               ,a.f002d_stk658
               ,a.ORDER_STK658
               ,c.f001v_pub201 as end_r
        from STK658 a, PUB205 b, PUB201 c
        where a.isvalid = 1
          and a.ORGID_STK658 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.f002d_stk658 >= '2018-01-01'
          and a.F009V_STK658 = c.CODE_PUB201
          and a.F004V_STK658 != '047005'
          and a.f010n_stk658 is not null
          and b.thscode is not null
          ) c
          on a.thscode=c.thscode and c.f002d_stk658=c.f002d_stk658 and a.ORDER_STK658=c.ORDER_STK658
          left join
          (
        select  b.thscode
               ,a.f002d_stk658
               ,a.ORDER_STK658
               ,nvl(a.F014D_STK658, a.F003D_STK658) as first_d
               ,c.f001v_pub201 as first_r
        from STK658 a, PUB205 b, PUB201 c
        where a.isvalid = 1
          and a.ORGID_STK658 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.f002d_stk658 >= '2018-01-01'
          and a.F015V_STK658 = c.CODE_PUB201
          and a.F004V_STK658 != '047005'
          and a.f010n_stk658 is not null
          and b.thscode is not null
          and a.F014D_STK658 is not null
          ) d
          on a.thscode=d.thscode and a.f002d_stk658=d.f002d_stk658 and a.ORDER_STK658=d.ORDER_STK658
          left join
          (
        select  b.thscode
               ,a.f002d_stk658
               ,a.ORDER_STK658
               ,nvl(a.F020D_STK658, a.F003D_STK658) as second_d
               ,c.f001v_pub201 as second_r
        from STK658 a, PUB205 b, PUB201 c
        where a.isvalid = 1
          and a.ORGID_STK658 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.f002d_stk658 >= '2018-01-01'
          and a.F021V_STK658 = c.CODE_PUB201
          and a.F004V_STK658 != '047005'
          and a.f010n_stk658 is not null
          and b.thscode is not null
          and a.F020D_STK658 is not null
          ) e
          on a.thscode=e.thscode and a.f002d_stk658=e.f002d_stk658 and a.ORDER_STK658=e.ORDER_STK658
        ) a
        left join
        (
        select b.thscode
              ,a.DECLAREDATE_STK127
              ,a.F127N_STK127
        from STK127 a, PUB205 b
        where a.isvalid = 1
        and b.F014V_PUB205 = a.ORGID_STK127
        and a.DECLAREDATE_STK127 >= '2017-06-01'
        and b.F003V_PUB205 = 'A股'
        and a.REPORTTYPECODE_STK127='HB'
        ) b
        on a.thscode=b.thscode
        where cast(declaredate_stk127 as date)-cast(f002d_stk658 as date)<0
        ) a
        where rw=1
        ) a
        group by thscode, first_d, second_d, end_d
        '''
        :param dates:
        """
        super(SSZC, self).__init__(sentence='get_sszc_factor', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'score']
        self.map_tp = 'series'

    def cal_raw_data(self, stock_df, **kwargs):
        sszc_first = stock_df[['thscode', 'first_d', 'first_score']]
        sszc_first.columns = ['stock_code', 'date', 'score']

        sszc_second = stock_df[['thscode', 'second_d', 'second_score']]
        sszc_second.columns = ['stock_code', 'date', 'score']

        sszc_end = stock_df[['thscode', 'end_d', 'end_score']]
        sszc_end.columns = ['stock_code', 'date', 'score']

        stock_df = pd.concat([sszc_first, sszc_second, sszc_end], axis=0).dropna()
        return stock_df[stock_df['date'].notnull()]

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.normalize_date(self.generate_raw_data())
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 违规处罚
class WGCF(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
        select thscode
              ,discdate_stk656
              ,case when step = '告知' then 1*(cash+dx_type)/2 else 0.5*(cash+dx_type)/2 end as wgcf_score
        from
        (
        select a.*
              ,b.F127N_STK127
              ,case when (a.F003V_STK656*10000/b.F127N_STK127*100) >= 20 then -2
                    when (a.F003V_STK656*10000/b.F127N_STK127*100) >= 5 then -1
                    else -0.5 end as cash
              ,row_number() over(partition by a.thscode, a.discdate_stk656 order by cast(a.discdate_stk656 as date)-cast(b.DECLAREDATE_STK127 as date) asc) as rw
        from
        (
        select b.thscode
              ,case when a.f013t_stk656 like '%告知%' then '告知' else '决定' end as step
              ,a.DISCDATE_STK656
              ,nvl(a.F003V_STK656, 0) as F003V_STK656
              ,case when a.F011V_STK656 like '%公司%' then -2
                    when a.F011V_STK656 like '%实际控制人%' then -1.5
                    when a.F011V_STK656 like '%高管%' then -1.5
                    when a.F011V_STK656 like '%一般%' then -1
                    else 0.5 end
                    as dx_type
        from STK656 a, PUB205 b
        where a.isvalid = 1
          and a.ORGID_STK656 = b.F014V_PUB205
          and b.F003V_PUB205 = 'A股'
          and a.DISCDATE_STK656 >= '2018-01-01'
          and thscode is not null
        --   and thscode = '002500.SZ'
        ) a
        left join
        (
        select b.thscode
              ,a.DECLAREDATE_STK127
              ,a.F127N_STK127
        from STK127 a, PUB205 b
        where a.isvalid = 1
        and b.F014V_PUB205 = a.ORGID_STK127
        and a.DECLAREDATE_STK127 >= '2017-06-01'
        and b.F003V_PUB205 = 'A股'
        and a.REPORTTYPECODE_STK127='HB'
        ) b
        on a.thscode=b.thscode
        where cast(discdate_stk656 as date)-cast(DECLAREDATE_STK127 as date)>0
        ) a
        where rw=1
        '''
        :param dates:
        """
        super(WGCF, self).__init__(sentence='get_wgcf', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'discdate_stk656', 'score']
        self.map_tp = 'series'

    def cal_raw_data(self, **kwargs):
        stock_df = self.normalize_date(self.api.get_wgcf_factor(*self.dates))
        stock_df = stock_df[stock_df['discdate_stk656'].notnull()]
        stock_df.columns = ['stock_code', 'date', 'score']
        return stock_df

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.cal_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 债务违约
class ZWWY(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
            select a.*
                  ,b.F045N_YB026
            from
            (
            select b.thscode
                  ,a.declaredate_stk428
                  ,a.F001D_STK428
                  ,substring(cast(a.F001D_STK428 as text), 1, 4) as end_year
                  ,a.F010N_STK428*10000 as F010N_STK428
                  ,a.F017N_STK428
                  ,round(a.F010N_STK428*10000/a.F017N_STK428*100-100,1)
                  ,'业绩预告' as type
                --   ,a.
            from STK428 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK428
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk428 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk435
                  ,a.enddate_stk435
                  ,substring(cast(a.enddate_stk435 as text), 1, 4) as end_year
                  ,a.F004N_STK435
                  ,a.F015N_STK435
                  ,round(a.F004N_STK435/a.F015N_STK435*100-100,1)
                  ,'业绩快报' as type
            from STK435 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK435
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk435 >= '20180101'
              and b.thscode is not null
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk060
                  ,a.ENDDATE
                  ,substring(cast(a.ENDDATE as text), 1, 4) as end_year
                  ,a.F002
                  ,a.F044N_STK060
                  ,nvl(round(a.F002/a.F044N_STK060 - 1, 2)*100, 0) as rat
                  ,'业绩报告' as type
                --   ,a.
            from STK060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk060 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'
              ) a
              left join
              (
              select thscode_yb026
                  ,enddate_yb026
                  ,substring(cast(enddate_yb026 as text), 1, 4) as end_year
                  ,F045N_YB026*1000000 as F045N_YB026
            from YB026
            where isvalid = 1
              and thscode_yb026 = '300033.SZ'
              ) b
            on a.thscode = b.thscode_yb026 and a.end_year = a.end_year
            '''
        :param dates:
        """
        super(ZWWY, self).__init__(sentence='get_zwwy', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'score']

    def cal_raw_data(self, stock_df, **kwargs):
        pass

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 并购重组
class BGCZ(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
            select a.*
                  ,b.F045N_YB026
            from
            (
            select b.thscode
                  ,a.declaredate_stk428
                  ,a.F001D_STK428
                  ,substring(cast(a.F001D_STK428 as text), 1, 4) as end_year
                  ,a.F010N_STK428*10000 as F010N_STK428
                  ,a.F017N_STK428
                  ,round(a.F010N_STK428*10000/a.F017N_STK428*100-100,1)
                  ,'业绩预告' as type
                --   ,a.
            from STK428 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK428
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk428 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk435
                  ,a.enddate_stk435
                  ,substring(cast(a.enddate_stk435 as text), 1, 4) as end_year
                  ,a.F004N_STK435
                  ,a.F015N_STK435
                  ,round(a.F004N_STK435/a.F015N_STK435*100-100,1)
                  ,'业绩快报' as type
            from STK435 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK435
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk435 >= '20180101'
              and b.thscode is not null
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk060
                  ,a.ENDDATE
                  ,substring(cast(a.ENDDATE as text), 1, 4) as end_year
                  ,a.F002
                  ,a.F044N_STK060
                  ,nvl(round(a.F002/a.F044N_STK060 - 1, 2)*100, 0) as rat
                  ,'业绩报告' as type
                --   ,a.
            from STK060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk060 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'
              ) a
              left join
              (
              select thscode_yb026
                  ,enddate_yb026
                  ,substring(cast(enddate_yb026 as text), 1, 4) as end_year
                  ,F045N_YB026*1000000 as F045N_YB026
            from YB026
            where isvalid = 1
              and thscode_yb026 = '300033.SZ'
              ) b
            on a.thscode = b.thscode_yb026 and a.end_year = a.end_year
            '''
        :param dates:
        """
        super(BGCZ, self).__init__(sentence='get_bgcz', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'score']

    def cal_raw_data(self, stock_df, **kwargs):
        pass

    def map_data(self, x, *args):
        return x

    def __call__(self, *args, **kwargs):
        stock_df = self.generate_raw_data()
        stock_df = stock_df[['stock_code', 'date', 'score']]
        stock_df = stock_df.groupby(['stock_code', 'date'])['score'].sum().reset_index()
        self.get_price(stock_list=stock_df['stock_code'].drop_duplicates().tolist())
        df_ret = self.get_stock_next_ret(stock_df)
        df_ret_index, count_df = self.get_next_index(df_ret, name=self.__class__.__name__)
        return df_ret_index, count_df, stock_df


# %% 研報評級
class YBPJ(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
            select a.*
                  ,b.F045N_YB026
            from
            (
            select b.thscode
                  ,a.declaredate_stk428
                  ,a.F001D_STK428
                  ,substring(cast(a.F001D_STK428 as text), 1, 4) as end_year
                  ,a.F010N_STK428*10000 as F010N_STK428
                  ,a.F017N_STK428
                  ,round(a.F010N_STK428*10000/a.F017N_STK428*100-100,1)
                  ,'业绩预告' as type
                --   ,a.
            from STK428 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK428
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk428 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk435
                  ,a.enddate_stk435
                  ,substring(cast(a.enddate_stk435 as text), 1, 4) as end_year
                  ,a.F004N_STK435
                  ,a.F015N_STK435
                  ,round(a.F004N_STK435/a.F015N_STK435*100-100,1)
                  ,'业绩快报' as type
            from STK435 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK435
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk435 >= '20180101'
              and b.thscode is not null
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk060
                  ,a.ENDDATE
                  ,substring(cast(a.ENDDATE as text), 1, 4) as end_year
                  ,a.F002
                  ,a.F044N_STK060
                  ,nvl(round(a.F002/a.F044N_STK060 - 1, 2)*100, 0) as rat
                  ,'业绩报告' as type
                --   ,a.
            from STK060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk060 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'
              ) a
              left join
              (
              select thscode_yb026
                  ,enddate_yb026
                  ,substring(cast(enddate_yb026 as text), 1, 4) as end_year
                  ,F045N_YB026*1000000 as F045N_YB026
            from YB026
            where isvalid = 1
              and thscode_yb026 = '300033.SZ'
              ) b
            on a.thscode = b.thscode_yb026 and a.end_year = a.end_year
            '''
        :param dates:
        """
        super(YBPJ, self).__init__(sentence='get_ybpj', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'score']

    def cal_raw_data(self, stock_df, **kwargs):
        pass

    def map_data(self, x, *args):
        pass


# %% 綜合評價
class ZHPJ(BaseGenerator):
    def __init__(self, dates):
        """
        query = '''
            select a.*
                  ,b.F045N_YB026
            from
            (
            select b.thscode
                  ,a.declaredate_stk428
                  ,a.F001D_STK428
                  ,substring(cast(a.F001D_STK428 as text), 1, 4) as end_year
                  ,a.F010N_STK428*10000 as F010N_STK428
                  ,a.F017N_STK428
                  ,round(a.F010N_STK428*10000/a.F017N_STK428*100-100,1)
                  ,'业绩预告' as type
                --   ,a.
            from STK428 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK428
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk428 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk435
                  ,a.enddate_stk435
                  ,substring(cast(a.enddate_stk435 as text), 1, 4) as end_year
                  ,a.F004N_STK435
                  ,a.F015N_STK435
                  ,round(a.F004N_STK435/a.F015N_STK435*100-100,1)
                  ,'业绩快报' as type
            from STK435 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.ORGID_STK435
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk435 >= '20180101'
              and b.thscode is not null
              and b.thscode = '300033.SZ'

            union all

            select b.thscode
                  ,a.declaredate_stk060
                  ,a.ENDDATE
                  ,substring(cast(a.ENDDATE as text), 1, 4) as end_year
                  ,a.F002
                  ,a.F044N_STK060
                  ,nvl(round(a.F002/a.F044N_STK060 - 1, 2)*100, 0) as rat
                  ,'业绩报告' as type
                --   ,a.
            from STK060 a, PUB205 b
            where a.isvalid = 1
              and b.F014V_PUB205 = a.COMCODE
              and b.F003V_PUB205 = 'A股'
              and a.declaredate_stk060 >= '20180101'
              and b.thscode is not null
            --   and a.f003v_stk428 like '%亏%'
              and b.thscode = '300033.SZ'
              ) a
              left join
              (
              select thscode_yb026
                  ,enddate_yb026
                  ,substring(cast(enddate_yb026 as text), 1, 4) as end_year
                  ,F045N_YB026*1000000 as F045N_YB026
            from YB026
            where isvalid = 1
              and thscode_yb026 = '300033.SZ'
              ) b
            on a.thscode = b.thscode_yb026 and a.end_year = a.end_year
            '''
        :param dates:
        """
        super(ZHPJ, self).__init__(sentence='get_zhpj', dates=dates)
        self.wencai_data = False
        self.col = ['thscode', 'date', 'score']

    def cal_raw_data(self, stock_df, **kwargs):
        pass

    def map_data(self, x, *args):
        pass


if __name__ == '__main__':
    zdht = ZDHT(dates=['20180101', '20230517'])
    df_ret_index, count_df, stock_df = zdht()
    zdht.backtest(stock_df)
    # stock_df = zdht.generate_raw_data(0.2, 0.1, 2, 1, 0.5)
    # stock_df = stock_df[['股票代码', '重大合同发布时间', 'score']]
    # stock_df.columns = ['stock_code', 'date', 'score']
    # stock_df['date'] = pd.to_datetime(stock_df['date'])
    # zdht.get_price()
    # df_ret = zdht.get_stock_next_ret(stock_df)
    # df_ret_index, count_df = zdht.get_next_index(df_ret, name='zdht')
    #
    # score = stock_df.set_index(['stock_code', 'date'])['score'].unstack([-2])
    # score_new = score.fillna(0)
    # for i in range(1, len(score_new)):
    #     score_new.iloc[i] = score_new.iloc[i - 1] * 0.8 + score_new.iloc[i]
    # index_ret = get_price('000300.SH',
    #                       zdht.dates[0],
    #                       zdht.dates[1],
    #                       '1d',
    #                       ['close'],
    #                       fq='pre').pct_change()
    # backtest(score_new, zdht.price_detail['close'].unstack([-2]), index_ret, 'zdht')

    dxzf = DXZF(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = dxzf()
    dxzf.backtest(stock_df)

    gqjl = GQJL(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = gqjl()
    gqjl.backtest(stock_df)

    gfhg = GFHG(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = gfhg()
    gfhg.backtest(stock_df)

    jcjh = JCJH(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = jcjh()
    jcjh.backtest(stock_df)

    zcjh = ZCJH(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = zcjh()
    zcjh.backtest(stock_df)

    xsjj = XSJJ(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = xsjj()
    xsjj.backtest(stock_df)

    skrbg = SKRBG(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = skrbg()
    skrbg.backtest(stock_df)

    fhpx = FHPX(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = fhpx()
    fhpx.backtest(stock_df)

    jgh = JGH(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = jgh()
    jgh.backtest(stock_df)

    ladc = LADC(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = ladc()
    ladc.backtest(stock_df)

    nbfb = NBFB(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = nbfb()
    nbfb.backtest(stock_df)

    pxdm = PXDM(dates=['20180101', '20230519'])
    df_ret_index, count_df, stock_df = pxdm()
    pxdm.backtest(stock_df)

    wgcf = WGCF(dates=['2018-01-01', '2023-05-19', '2017-06-01'])
    df_ret_index, count_df, stock_df = wgcf()
    wgcf.backtest(stock_df)

    sszc = SSZC(dates=['2018-01-01', '2023-05-19', '2017-06-01'])
    df_ret_index, count_df, stock_df = sszc()
    sszc.backtest(stock_df)

    yjbg = YJBG(dates=['2018-01-01', '2023-05-19'])
    df_ret_index, count_df, stock_df = yjbg()
    yjbg.backtest(stock_df)






