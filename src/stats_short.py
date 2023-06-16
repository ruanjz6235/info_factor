import pandas as pd
import numpy as np


class ShortStats:
    def __init__(self, df):
        self.df = df

    def get_stats(self):
        df_stats = self.df.count(axis=1)
        return df_stats

    def cal_threshold(self):
        raise NotImplementedError


































