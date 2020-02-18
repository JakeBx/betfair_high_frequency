import h2o
from utils import dollars_to_ticks
from datetime import datetime as dt
from itertools import product
import numpy as np

class h2oModel:
    def __init__(self, model_path):
        self.mojo_path = model_path

    def process_df_bsp(self, df, end):
        lay_vol_cols = ['batl_v', 'batl_v_1', 'batl_v_2', 'batl_v_3', 'batl_v_4', 'batl_v_5']
        df.fillna(0, inplace=True)
        df['time_to_markt'] = df.starttime - np.datetime64('now')
        df['time_to_markt'] = df['time_to_markt'].astype(np.int) // 1000000000
        df['mid'] = (df.batb + df.batl) / 2

        for c in lay_vol_cols:
            df[c] = df[c] * df.mid

        price_cols = ['batl', 'batb', 'mean_trd', 'mid']
        for c in price_cols:
            df[f'{c}_ticks'] = df[c].apply(dollars_to_ticks)

        df['spread'] = df.batl_ticks - df.batb_ticks

        df['diff'] = df.groupby(['sid']).mid_ticks.diff().cumsum()

        df['ticks_from_mean'] = df.mid_ticks - df.mean_trd_ticks

        ma_cols = ['batl_ticks', 'batl_v', 'batb_v',
                   'batl_v_1', 'batb_v_1', 'batl_v_2', 'batb_v_2', 'batl_v_3', 'batb_v_3',
                   'batl_v_4', 'batb_v_4', 'batl_v_5', 'batb_v_5', 'ticks_from_mean']
        ma_periods = [5, 10, 50]
        for c, l in product(ma_cols, ma_periods):
            df[f'{c}_ma_{l}'] = df[c].rolling(l).mean()

        return df.dropna()

    def process_df(self, df):
        df['mid'] = (df.batb + df.batl) / 2

        price_cols = ['batl', 'batb', 'mean_trd', 'mid']
        for c in price_cols:
            df[f'{c}_ticks'] = df[c].apply(dollars_to_ticks)

        df['spread'] = df.batl_ticks - df.batb_ticks

        lay_vol_cols = ['batl_v', 'batl_v_1', 'batl_v_2', 'batl_v_3', 'batl_v_4', 'batl_v_5']



        df['ticks_from_mean'] = df.mid_ticks - df.mean_trd_ticks

        feature_cols = ['batl_ticks', 'time_to_markt', 'batl_v', 'batb_v',
       'batl_v_1', 'batb_v_1', 'batl_v_2', 'batb_v_2', 'batl_v_3', 'batb_v_3',
       'batl_v_4', 'batb_v_4', 'batl_v_5', 'batb_v_5', 'ticks_from_mean',
       'spread', 'track']

        model_df = df[['sid'] + feature_cols].groupby('sid').last()
        model_df.dropna(inplace=True)

        return model_df

    def get_predictions(self, df, bsp=False, end=None):
        if bsp:
            df = self.process_df_bsp(df, end)
        else:
            df = self.process_df(df)
        if df.shape[0]:
            prediction_df = h2o.mojo_predict_pandas(df, self.mojo_path)
            return prediction_df['True'].values[-1]
        else:
            return None
