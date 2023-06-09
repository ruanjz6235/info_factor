# %% 例子
import pandas as pd
import numpy as np
a = pd.DataFrame(np.random.randn(10, 8))
a0 = pd.DataFrame(np.random.randn(7))
a0 = pd.concat([a0, pd.DataFrame([np.nan], index=[7])])
a = pd.concat([a, a0.T])
a1 = pd.DataFrame([[1e2] * 8] * 2).T
a = pd.concat([a, a1.T])

a['stock_code'] = ['a1'] * 5 + ['a3'] * 3 + ['a2'] * 5

a.columns = ['b'+str(i) for i in a.columns[:-1]] + ['stock_code']

a_col = [x+'_rank' for x in a.columns[:-1]]
a_percent = [x+'_percent' for x in a.columns[:-1]]
a[a_col] = a.groupby('stock_code').rank(method='max')
df_max = a.groupby('stock_code')[a_col].max()

df_max.columns = [x + '_max' for x in df_max.columns]
a = a.merge(df_max.reset_index(), on=['stock_code'], how='left')

a[a_percent] = a[a_col].values / a[df_max.columns].values

































