# Imports
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
import subprocess
from talib import abstract


def get_fullpath(filename: str):
    """

    :param filename:
    :return:
    """
    return os.sep.join(['C:','Users','suare','data', 'raw', 'quantquote', 'minutes', filename])


def get_indicator(ind: str):
    """

    :param ind:
    :return:
    """
    # Add as many indicators as necessary: see all indicators in https://mrjbq7.github.io/ta-lib/
    return abstract.Function(ind)


# Define this to parse the CSVs to ARFF later automatically
def create_arff_file(filename: str, output=None):
    """

    :param filename:
    :param output:
    :return:
    """
    java_mem = '-Xmx3074m'
    filename = filename.replace('.csv', '')
    if output is None:
        output=filename
    else:
        output = output.replace('.csv', '')
    wekadev_libpath = 'C:\\Users\\suare\\Workspace\\phd_cetrulin\\moa-2017.06-sources\\lib\\weka-dev-3.7.12.jar'
    command = ['java', java_mem, '-classpath', wekadev_libpath,
               'weka.core.converters.CSVLoader', filename + '.csv', '>', output + '.arff']
    f = open(filename + '.arff', "w")
    subprocess.call(command, stdout=f)
    print('If the arff is not generated, run the next in the terminal.')
    print(str(' '.join(command)))


def set_label(row):
    """
    This function creates the target feature of the dataset to be created
    :param row:
    :return:
    """
    return 1 if row['close_t+1'] > row['close'] else 0


# Creating a dictionary of technical indicators using TALib abstract API
indicators = ['sma','ema','wma','mom','stoch','macd' ,'rsi' ,'willr',
              'adosc' ,'cci','adx','aroon' ,'bbands','obv' ,'trima',
              'roc' ,'rocr','stochf','medprice','typprice','wclprice',
              'atr','macdfix','mfi' ,'sar' ,'ppo']

# Start of script
sources = ['S&P500']
# levels = ['1min-level', '5min-level', '10min-level', '15min-level', '30min-level', '1h-level']  # TODO
levels = ['1s-level', '5s-level', '10s-level', '15s-level', '30s-level']
modes = ['indicators_best', 'indicators_best_and_times', 'indicators_fullset']
# sets = ['mahalanobis', 'dev', 'train'] #'mahalanobis', 'dev'] #, 'train']  # dates hardcoded later.

# Paths for symbols
RESULT_PATH=os.sep.join(['C:','Users','suare','PycharmProjects','RegimeSwitchingSeriesGenerator','output'])
devsets_path = 'C:\\Users\\suare\\Workspace\\phd_cetrulin\\moa-2017.06-sources\\data\\real\\spy_final_2021_staging_area\\'
input_path = os.sep.join(['C:','Users','suare','data', 'raw', 'quantquote', 'minutes'])


files_for_indicators = list()
load(files_for_indicators)  #  exported file from other script


for mode in modes:
    for file in files_for_indicators:
        #         for level in levels:
        #         for dataset in sets:
        #           filename = file.replace(SOURCE_PATH+os.sep,'')
        filename = file.split(os.sep)[-1]
        FIELD = 'close' # price->'ts' returns->'ret_ts' ts_with_added_noise-> 'ts_n2_post'
        print(f'Start {filename}')
        RESULT_FILEPATH_PROCESSED = \
            os.sep.join([RESULT_PATH, filename.split(os.sep)[-1].replace('.csv', '')+f'{mode}.csv'])

        # Open file
        df = pd.read_csv(file, sep=';')
        print(df.head())
        df = df.drop_duplicates(['datetime', 'open', 'high', 'low', 'close', 'volume'])

        # Add parameters to transform in TS
        timeseries=['close', 'open', 'high', 'low']
        # Length of the TS. How many values do we keep per series.
        # e.g. 1 -> t / 2 -> t,t-1 / 3 -> t,t-1,t-2 / 4 -> t,t-1,t-2,t-3
        length = 5

        # Add lagged times
        for column in timeseries:
            for i in range(1, length):
                df[column+'_t-'+str(i)] = df[column].shift(i)  # it could also be sorted and group by if needed

        # All the numbers here and below assume a default time period for ta params of 10 mins averages
        # change the other numbers (params of 5min and 20min and theremoval of 20 first mins of the day below)
        # if the mins number is changed. In that case change as well the time tag '_10' with the corresponding one.
        default_timerange=10
        # Set extra timeranges for moving averages
        extra_timeranges = [default_timerange//2, default_timerange, default_timerange*2, default_timerange*3]

        # ###########################################
        # Iterate and run list of indicators selected
        # All of them produced for 25 prior mins of data
        # ###########################################
        for ind in list(indicators):
            #                 print(ind)
            if ind not in ['adosc', 'obv', 'mfi']:  # avoiding indicators that need volume
                # For indicators that only return one column
                # (this will need to be modified depending on the selection of indicators)
                if ind in ['ema', 'sma', 'trima']:
                    for timerange in extra_timeranges:
                        df[ind+'_'+str(int(timerange))] = get_indicator(ind)(df, timeperiod=timerange)
                elif ind not in ['bbands', 'aroon', 'stoch', 'macd', 'macdfix', 'stochf']:
                    df[ind+'_'+str(int(default_timerange))] = get_indicator(ind)(df, timeperiod=(default_timerange))
                # Otherwise check the list of columns and return all
                else:
                    key_output = get_indicator(ind)(df, timeperiod=default_timerange)  # , price='close')
                    for j in range(0, len(list(key_output.columns))):
                        df[ind+'_'+key_output.columns[int(j)]] = key_output[key_output.columns[j]]

        # One minute ahead closing price
        df['close_t+1'] = df['close'].shift(-1)

        # Creating label/y to be predicted / independent (predicted) feature 'y'
        df['label'] = df.apply(set_label, axis=1)
        #     df['label'] = df.apply(func, axis=1)

        df.dropna(inplace=True)

        # Encoding cyclical continuous features for the trading day (6.5h a day except for shortened sessions)
        start_market = 34200.0  # 9:30am
        end_market = 57600.0  # 04:00pm
        df['seconds'] = \
            (pd.to_datetime(df.datetime) - pd.to_datetime(df.datetime.str[:10])).dt.total_seconds() - start_market
        # specific to US EFTs, and not considering shortened trading sessions (2 per year).
        seconds_trading_day = end_market - start_market
        df['sin_time'] = np.sin(2*np.pi*df.seconds/seconds_trading_day)
        df['cos_time'] = np.cos(2*np.pi*df.seconds/seconds_trading_day)
        #     df.cos_time.plot(figsize=(15,6))
        #     df.sample(50).plot.scatter('sin_time','cos_time').set_aspect('equal');

        # Day of the week (cyclical also, for weeks of 5 days)
        df['dow'] = pd.to_datetime(df.datetime).dt.dayofweek
        df['sin_dow'] = np.sin(2*np.pi*df.dow/5)
        df['cos_dow'] = np.cos(2*np.pi*df.dow/5)
        #     df.cos_dow.plot(figsize=(15,6))
        #     df.sample(50).plot.scatter('sin_dow','cos_dow').set_aspect('equal');

        # Select columns for output
        if mode == 'indicators_best':
            # best pool found through indicators grid search script
            columns_selected = ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd',
                                'sma_5','sma_10', 'wma_10','ema_10','trima_10','adx_10',
                                'bbands_upperband','bbands_lowerband','roc_10', 'aroon_aroondown','aroon_aroonup',
                                'label']
        elif mode == 'indicators_best_and_times':
            columns_selected = ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd',
                                'sma_5','sma_10', 'wma_10','ema_10','trima_10','adx_10',
                                'bbands_upperband','bbands_lowerband','roc_10', 'aroon_aroondown','aroon_aroonup',
                                'cos_time', 'cos_dow', 'label']
        elif mode == 'indicators_fullset':
            columns_selected = [  # 'datetime',
                'rsi_10','willr_10','macd_macd' ,'cci_10','mom_10',
                'stoch_slowk','stoch_slowd',
                'sma_5','sma_10','sma_20','sma_30',
                'wma_5','wma_10','wma_20','wma_30',
                'ema_5','ema_10','ema_20','ema_30',
                'trima_5','trima_10','trima_20','trima_30',
                'adx_10','bbands_upperband','bbands_middleband','bbands_lowerband',
                'roc_10','rocr_10','stochf_fastd','stochf_fastk',
                'aroon_aroondown','aroon_aroonup','medprice_10','typprice_10','wclprice_10',
                'atr_10','macdfix_macd','sar_10',
                'adosc_10', 'obv_10', 'mfi_10', 'ppo_10',  # ######### commented out previosly
                'volume','volume_t-1','volume_t-2','volume_t-3','volume_t-4',
                'close','close_t-1','close_t-2','close_t-3','close_t-4',
                'high','high_t-1','high_t-2','high_t-3','high_t-4',
                'open','open_t-1','open_t-2','open_t-3','open_t-4',
                'low','low_t-1','low_t-2','low_t-3','low_t-4',
                'cos_time', 'sin_time', 'cos_dow', 'sin_dow',
                #                          # 'binary_label',
                'label']  # ,'gap_t+1','close_t+1'] # + ['adosc_10', 'obv_10', 'mfi_10', 'ppo_10']

        # indicators have dependendies up to 45 mins before.
        # removing records that may look at pre-market or at the previous date
        df.datetime = pd.to_datetime(df.datetime)
        df.set_index('datetime', drop=True, inplace=True)
        df = df.between_time('10:05', '15:58').reset_index()

        # Export processed data
        print(df.head())
        print(len(df))
        print(RESULT_FILEPATH_PROCESSED)
        print('===========================')
        print('===========================')
        output = pd.DataFrame(df, columns=columns_selected)  # [df['datetime'] >= '2017-09-25 14:07:00']
        output.to_csv(RESULT_FILEPATH_PROCESSED, sep=',', encoding='utf-8', index = False)
        create_arff_file(RESULT_FILEPATH_PROCESSED,
                         output=devsets_path+RESULT_FILEPATH_PROCESSED.split(os.sep)[-1]) # export in ARFF

        # Now plot close price and volume overtime.
        df.set_index('datetime',drop=True).plot(y=["close_t-1"], figsize=(18,6))
        plt.show()

        print(f'Number of instances: {len(df)}')
        # Printing classes distributions
        print("Class distribution: ")
        label_zero = len(df[df['label'] == 0])
        label_one = len(df[df['label'] == 1])

        print("0 in "+str(float(label_zero)/(label_one+label_zero))+"%")
        print("1 in "+str(float(label_one)/(label_one+label_zero))+"%")

        print("===============================================================\n\n")


# Print a List of commands to be outputed by the terminal to replace the labels to categorical
files = list()
for file in os.listdir(devsets_path):
    #     if '.' not in file:
    seed_folder = os.sep.join([devsets_path, file])
    #         seed_files = os.listdir(seed_folder)
    #         for sf in seed_files:
    if '.arff' in file:  # sf:
        #                 full_path = os.sep.join([devsets_path, file, sf])
        full_path = os.sep.join([devsets_path, file]).replace('C:\\Users\\suare\\Workspace\\phd_cetrulin\\',
                                                              '/mnt/c/Users/suare/Workspace/phd_cetrulin/').replace('\\','/')
        #                 print(full_path)
        print("sed -i 's/^.*@attribute label numeric.*$/@attribute label {0, 1}/' " + full_path)