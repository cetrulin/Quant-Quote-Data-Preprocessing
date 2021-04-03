
import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from shutil import copyfile

config = {
    'name': 'spy-seconds',
    'lvl_str': 's',
    'path': 'C:\\Users\\suare\\data\\analysis\\quantquote\\',

    # ############ For sec level
    'years_to_explore': ['2016', '2017', '2018', '2019', '2020'],
    'symbol': 'spy',
    'symbol_name': 'S&P500',
    'category': {'unespecified': ['spy']},  # 'gld', 'spy','xle', 'emb','dia', 'qqq', 'ewp'
    # 'desired_abs_mean_tresh': 0.00000000005,
    'output_path': 'C:\\Users\\suare\\data\\tmp\\spy_seeds',
    'prefix': 'table_',
    'extension': '.csv',
    'separator': ';',
    'desired_length': 550,   # for mahab states
    'desired_abs_mean_tresh': 0.01,
    'allowed_outliers_pct': 0.01,
    'ms_field': 'timestamp',  # time
    'dt_field': 'datetime',   # date
    'cols': ['open', 'high', 'low', 'close', 'volume', 'datetime', 'gap', 'timestamp']

    # 'specific_period': True,
    # 'period': '202006'   # , '202006']
    #
    # ############ For sec level
    # For minute level
    #
}


def read_file(filename: str) -> pd.DataFrame:
    path = os.sep.join([config['path'], filename])
    filepath = os.sep.join([path, config['prefix'] + config['symbol'] + config['extension']])
    if os.path.isfile(filepath):
        print(f'Reading {filename}')
        df = pd.read_csv(filepath, names=config['cols'])
        # print(df.head())
        return df


# 1s-level\\S&P500

mahab_m = '07'
dev_m = '10'
first = True
files = dict()


def get_pretraining_states(mahabset_df: pd.DataFrame, config: dict) -> dict:
    """
    This function get the dataframes for pretraining
    :param mahabset_df:
    :param config:
    :return:
    """
    # Generate close price returns and moving average to select the period with a mean close to 0
    log_ret = False  # otherwise percentual
    if log_ret:
        mahabset_df['close_returns'] = np.log(mahabset_df['close'] / mahabset_df['close'].shift(1))
        # mahabset_df['close_returns'] = np.log(1 + mahabset_df['close'].pct_change())
    else:
        mahabset_df['close_returns'] = mahabset_df['close'] / mahabset_df['close'].shift(1) - 1
        # mahabset_df['close_returns'] = mahabset_df['close'].pct_change(1)  # same result
    mahabset_df.dropna(inplace=True)
    sma_col = f"SMA_{config['desired_length']}"
    mahabset_df[sma_col] = \
        mahabset_df['close_returns'].rolling(window=config['desired_length']).mean()
    mahabset_df[sma_col+'_abs'] = mahabset_df[sma_col].abs()
    mahabset_df['SMA_start_date'] = mahabset_df[config['dt_field']].shift(config['desired_length'])
    mahabset_df['SMA_start_ms'] = mahabset_df[config['ms_field']].shift(config['desired_length'])

    # Drop first rows as the moving average is NaN
    len_with_nans = len(mahabset_df)

    # Prepare extra cols
    # mahabset_df['datetime'] = mahabset_df.index.astype(str)   #  maybe for min level is relevant
    mahabset_df.set_index(config['dt_field'], drop=False, inplace=True)
    mahabset_df.dropna(inplace=True)
    assert (len_with_nans - len(mahabset_df)) == config['desired_length'], 'There are non expected NaNs'

    states_dict = dict()
    for i in range(1, 4):  # States 1, 2 and 3 (hardcoded by now)
        selected_df = identify_state(config, mahabset_df, sma_col, state_id=i)
        # assert len(selected_df) == 1, "The maximum value is not a unique value."
        print(len(selected_df))
        print(selected_df)

        for idx, rw in selected_df.iterrows():
            print(f"Current selection from {rw['SMA_start_date']} to {rw[config['dt_field']]} with mean {sma_col}")
            states_dict[i] = \
                mahabset_df[(mahabset_df['SMA_start_date'].between(rw['SMA_start_date'], rw[config['dt_field']]))]
            states_dict[i].sort_index(ascending=True, inplace=True)

    return states_dict


def identify_state(config: dict, mahabset_df: pd.DataFrame, sma_col: str, state_id: int) -> pd.DataFrame:
    """
    This function identifies the last row of a mahab state depending on a logic defined manually for that state id.
    """
    if state_id == 1:  # Select one close to 0 (the closest)
        # Select one close to 0
        # (not the closest cos there are period = 0 due to lack of liquidity at certain frequencies)
        # Filter by desired mean fpr the lateral movement (we'll get the maximum inside a range)
        selected_df = mahabset_df[mahabset_df[sma_col + '_abs'] <= config['desired_abs_mean_tresh']]
        selected_df = selected_df[selected_df[sma_col + '_abs'].max() == selected_df[sma_col + '_abs']]
    elif state_id == 2:  # Select one negative (min)
        selected_df = mahabset_df[mahabset_df[sma_col].min() == mahabset_df[sma_col]]
    elif state_id == 3:  # Select one positive (max)
        selected_df = mahabset_df[mahabset_df[sma_col].max() == mahabset_df[sma_col]]
    else:
        assert False, "The trend/pattern for this state has not been specified"
    return selected_df


period_id = 1
for yr in config['years_to_explore']:
    # this will need to be refactored/changed (maybe to an iterator) for min level
    for q_month in ['01', '04', '07', '10']:  # just for s level

        # 2. it picks the prior period as a devset
        if not first:
            mahab_period = dev_period
            dev_period = period
        else:
            mahab_period = f'{int(yr)-1}-{mahab_m}'
            dev_period = f'{int(yr)-1}-{dev_m}'
            first = False
        period = yr + '-' + q_month
        files[period] = dict()

        # change of period till here

        # 1. it picks a period and changes the name.
        for level in os.listdir(config['path']):
            files[period][level] = dict()
            if config['lvl_str'] in level:
                lvl_path = config['path'] + level + os.sep + config['symbol_name']
                for file in os.listdir(lvl_path):  # all of these loops are not efficient at all
                    if '.csv' in file:
                        if mahab_period in file:
                            files[period][level]['mahab'] = lvl_path + os.sep + file
                        if dev_period in file:
                            files[period][level]['dev'] = lvl_path + os.sep + file
                        if period in file:
                            files[period][level]['train'] = lvl_path + os.sep + file

                # 2. Export all sets in a folder with a period number (like a seed)
                Path(os.sep.join([config['output_path'], level, str(period_id)])).mkdir(parents=True, exist_ok=True)
                print(files[period][level]['dev'])
                print(os.sep.join([config['output_path'], level, str(period_id),
                                   f'{config["symbol"]}_devset.csv']))
                dev_df = pd.read_csv(files[period][level]['dev'], sep=config['separator'])
                dev_df.to_csv(os.sep.join([config['output_path'], level, str(period_id),
                                           f'{config["symbol"]}_devset.csv']))
                train_df = pd.read_csv(files[period][level]['train'], sep=config['separator'])
                train_df.to_csv(os.sep.join([config['output_path'], level, str(period_id),
                                             f'{config["symbol"]}_train.csv']))

                # 3. It creates a moving average of x examples over the previous period to the devset.
                mahabset_df = pd.read_csv(files[period][level]['mahab'], sep=config['separator'])
                states = get_pretraining_states(mahabset_df, config)
                for k in states.keys():
                    states[k].to_csv(os.sep.join([config['output_path'], level, str(period_id),
                                                  f'{config["symbol"]}_mahalanobis_state_{k}.csv']))

                # Then trigger from here the whole convertion (technical indicators).
                # TODO: Let's make it a file that can be imported.
                # Let's generate /txt files too in that lcocation


        period_id = period_id + 1