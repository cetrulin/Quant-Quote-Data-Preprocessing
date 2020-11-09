# -*- coding: utf-8 -*-

# Data preparation at one-second level for Ph.D thesis
# @author: Andres L. Suarez-Cetrulo

# Imports
import os
import matplotlib as mpl
import pandas as pd
import numpy as np
import datetime, timedelta
import logging
from pathlib import Path
import yaml

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

# Using TALib abstract API to create a dictionary of technical indicators to iterate later.
from talib import abstract

# Set logging in Jupyter notebooks
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# In[9]:
def define_indicators():
    # Creating a dictionary of technical indicators using TALib abstract API
    indicator = {}

    # Add as many indicators as necessary: see all indicators in https://mrjbq7.github.io/ta-lib/
    indicator['sma'] = abstract.Function('sma')  # Simple Moving Average
    indicator['ema'] = abstract.Function('ema')  # Exponential Moving Average
    indicator['wma'] = abstract.Function('wma')  # Weighted Moving Average
    indicator['mom'] = abstract.Function('mom')  # Momentum
    indicator['stoch'] = abstract.Function('stoch')  # Stochastic (returns K and D)
    indicator['macd'] = abstract.Function('macd')  # Moving Average Convergence/Divergence
    indicator['rsi'] = abstract.Function('rsi')  # Relative Strength Index
    indicator['willr'] = abstract.Function('willr')  # Williams' %R
    indicator['adosc'] = abstract.Function('adosc')  # Chaikin A/D Oscillator
    indicator['cci'] = abstract.Function('cci')  # Commodity Channel Index

    # other indicators
    use_extra_indicators = True
    if use_extra_indicators:
        indicator['adx'] = abstract.Function('adx')  # Average Directional Movement Index
        indicator['aroon'] = abstract.Function('aroon')  # Aroon
        indicator['bbands'] = abstract.Function('bbands')  # Bollinger Bands
        indicator['obv'] = abstract.Function('obv')  # On Balance Volume
        indicator['trima'] = abstract.Function('trima')  # Triangular Moving Average
        indicator['roc'] = abstract.Function('roc')  # Rate of change : ((price/prevPrice)-1)*100
        indicator['rocr'] = abstract.Function('rocr')  # Rate of change ratio: (price/prevPrice)
        indicator['stochf'] = abstract.Function('stochf')  # Stochastic fast (returns K and D)
        indicator['adosc'] = abstract.Function('adosc')  # Chaikin A/D Oscillator
        indicator['medprice'] = abstract.Function('medprice')  # Median Price
        indicator['typprice'] = abstract.Function('typprice')  # Typical Price
        indicator['wclprice'] = abstract.Function('wclprice')  # Weighted Close Price
        indicator['atr'] = abstract.Function('atr')  # Average True Range
        indicator['macdfix'] = abstract.Function('macdfix')  # #Moving Average Convergence/Divergence Fix 12/26
        indicator['mfi'] = abstract.Function('mfi')  # Money Flow Index
        indicator['sar'] = abstract.Function('sar')  # Parabolic SAR
        indicator['ppo'] = abstract.Function('ppo')  # Percentage Price Oscillator

    return indicator


# In[4]:
def uncompress_src(path):
    import subprocess
    try:
        byteoutput = subprocess.check_output(['tar', '-xzf', path+'data_2014-2018.tar.gz'])
        return byteoutput.decode('UTF-8').rstrip()
    except subprocess.CalledProcessError as e:
        print("Error in uncompressing src -a:\n", e.output)
        return None


# In[5]:
def move_file(file, destination):
    import subprocess
    try:
        byteoutput = subprocess.check_output(['mv', file, destination])
        return byteoutput.decode('UTF-8').rstrip()
    except subprocess.CalledProcessError as e:
        print("Error in uncompressing src -a:\n", e.output)
        return None


# In[6]:
def get_datediff(start, end):
    """
    Obtains amount of days between starting and end date.
    @param: start: start date
    @param: end: end date
    @return: amount of days between both input dates
    """
    d1 = datetime.datetime.strptime(start.replace('-', ':')+':00:00:00', '%Y:%m:%d:%H:%M:%S')
    d2 = datetime.datetime.strptime(end.replace('-', ':')+':23:59:59', '%Y:%m:%d:%H:%M:%S')
    return round((d2 - d1).total_seconds() / 3600 / 24, 0)


# In[7]:
def get_dates_month(year, month):
    """ Get daterange for a given month of the year."""
    year = int(year)
    month = int(month)
    d1 = datetime.datetime(year, month, 1).date()
    if month == 12:
        d2 = (datetime.datetime(year, month, 31)).date()
    else:
        d2 = (datetime.datetime(year, month + 1, 1) + datetime.timedelta(days=-1)).date()
    return str(d1), str(d2)


# In[8]:
def get_dates_quarter(year, quarter):
    """ Get daterange for a given quarter of the year."""
    year = int(year)
    quarter = int(quarter)
    d1 = datetime.datetime(year, 3 * quarter - 2, 1).date()
    if quarter == 4:
        d2 = (datetime.datetime(year, 3 * quarter, 31)).date()
    else:
        d2 = (datetime.datetime(year, 3 * quarter + 1, 1) + datetime.timedelta(days=-1)).date()
    return str(d1), str(d2)


# In[10]:
def run_tests_minutes(df, result_path, level, start_date, end_date):
    # Save tests
    # Testing the dataset (TO-
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file = open(result_path + 'Tests_EFT' + level + '-level_' + '[' + start_date + '-' + end_date + '].txt', "w")

    # First check to see if there are any gaps or duplicates.
    # The number of rows must be:
    # diff*60*24 in df[(df['timestamp'] > start_date) & (df['timestamp'] <= end_date+" 23:59:59")
    # approx
    number_market_minutes = 6.5 * (60 ** 2)  # 6.5 market hours
    # 4 bank holidays in this semester.get_datediff(start_date, end_date)/7*5 to exclude weekends.
    # TODO. a function that removes weekends and bank holidays from the number of days directly! 
    # So the condition of TEST1 is '== to' and not '< to'
    # df.date.dt.date.unique() instead of get_datediff(start_date, end_date)/7*5-4
    # with this we assume that we don't miss entire days. # TO-DO add extra test for days
    minutes_of_market = len(df[(df['datetime'] > start_date) & (
                df['datetime'] <= end_date + " 23:59:59")].datetime.dt.date.unique()) * number_market_minutes
    total_seconds = len(
        df[(df['datetime'] > start_date) & (df['datetime'] <= end_date + " 23:59:59")].between_time('09:30', '16:00'))
    test1 = minutes_of_market < total_seconds
    logging.warning('TEST 1: ' + str(test1) + ' - The amount of records is correct?' + '\n')
    file.write('TEST 1: ' + str(test1) + ' - The amount of records is correct?')
    # TODO. a function that removes weekends and bank holidays from the number of days.
    #  So the condition of TEST1 is '== to' and not '< to'
    logging.warning(
        'Approx mins of market should be: ' + str(minutes_of_market) + ' and actual number is: ' + str(total_seconds))
    file.write('Approx mins of market should be: ' + str(minutes_of_market) + ' and actual number is: ' + str(
        total_seconds) + '\n')
    logging.info('TO-DO: Improve this test to be more accurate')
    file.write('TO-DO: Improve this test to be more acccurate.' + '\n')

    # Second check to see if there are any gaps
    ohlc_dict = {'close': 'last', 'high': 'max', 'low': 'min', 'open': 'first', 'volume': 'sum'}
    test = df.resample(level).agg(
        ohlc_dict)  # This will create rows with NaN values if there are any minutes missing in the timerange
    test = test[(test.index > start_date) & (test.index <= end_date + " 23:59:59")]
    list_of_days = df.datetime.dt.date.unique()
    test['datetime'] = test.index.date
    test = test[test['datetime'].isin(list_of_days)].between_time('09:30',
                                                                  '16:00')  # only search for gaps in market hours
    consecutive_gaps = test.close.isnull().astype(int).groupby(
        test.close.notnull().astype(int).cumsum()).sum().sort_values(ascending=False)
    check = test[test['open'].isnull()]  # only displays gaps. if empty, happy days!
    # print(check)
    test2 = len(test[test['open'].isnull()]) == 0
    logging.warning('TEST 2: ' + str(test2) + ' - Is the dataset complete? (no gaps at minute level).  ' + str(
        len(test[test['open'].isnull()])) + ' gaps.')
    file.write('TEST 2: ' + str(test2) + ' - Is the dataset complete? (no gaps at minute level).  ' + str(
        len(test[test['open'].isnull()])) + ' gaps.' + '\n')
    # logging.info('TODO: print how many consecutive gaps (group by),
    #  how many gaps per day, what times are the gaps normally at.')
    # file.write('TODO: print how many consecutive gaps (group by),
    #  how many gaps per day, what times are the gaps normally at.'+'\n')
    logging.warning('The max number of consecutive gaps at this level and this period is: ' + str(
        consecutive_gaps.max()) + ' gaps.' + '\n')
    logging.warning('A sample of the distribution of these consecutive gaps, sorted by descending order is: ' + '\n')
    logging.warning(consecutive_gaps[:25])  # just the 25 greatest values
    # print('how many consecutive gaps (group by), how many gaps per day, what times are the gaps normally at.')
    file.write('The max number of consecutive gaps at this level and this period is: ' + str(
        consecutive_gaps.max()) + ' gaps.' + '\n')
    file.write('A sample of the distribution of these consecutive gaps, sorted by descending order is: ' + '\n')
    file.write(str(consecutive_gaps[:25]))  # just the top 25 values

    # third check
    original_plus_gaps = (len(
        df[(df['datetime'] > start_date) & (df['datetime'] <= end_date + " 23:59:59")].between_time('09:30',
                                                                                                    '16:00')) + len(
        check))
    resampled = len(test) * 60
    test3 = (original_plus_gaps == resampled)
    logging.warning('TEST 3: ' + str(
        test3) + ' - Are the resampled (at second level) and the original dataset of the same size? (so # rows is ok)')
    file.write('TEST 3: ' + str(
        test3) + ' - Are the resampled (at second level) and the original dataset of the same size? (so # rows is ok)' +
               '\n')
    logging.info('original_plus_gaps is:' + str(original_plus_gaps) + ' and resampled is: ' + str(
        resampled) +
                 '. If resampled is greater, it means gaps at second level (that may not be material at minute level).')
    file.write('original_plus_gaps is:' + str(original_plus_gaps) + ' and resampled is: ' + str(
        resampled) +
               '. If resampled is greater, it means gaps at second level (that may not be material at minute level).' +
               '\n')

    # fourth check
    test4 = df.groupby(['timestamp']).count()['high']
    test4 = (len(test4[test4 > 1]) == 0)
    logging.warning('TEST 4: ' + str(test4) + ' - Is the dataset free from any duplicates?')
    file.write('TEST 4: ' + str(test4) + ' - Is the dataset free from any duplicates?' + '\n')
    # If there are any dups (Test 4 == False), check the given row (change timestamp below)
    # example=df[(df['timestamp'] == '2018-07-01 20:33:00')]
    # example

    print(
        'TEST 5: Number of market days (it should be an average of 251 market days a year +/-1 normally). # days is:' + str(
            len(list_of_days)) + '\n')
    file.write(
        'TEST 5: Number of market days (it should be an average of 251 market days a year +/-1 normally). # days is:' + str(
            len(list_of_days)) + '\n')
    # print(list_of_days)
    # get_ipython().magic('notify -m "Look at tests"')

    # TEST 6: Sacar el porcentaje de gaps con relacion al dataset

    # example 2014: 365 (number of days in 2014) – 104 (number of weekend days in 2014) – 9 (public holidays) = 252 days
    # 2015 - 252, 206 and 2017 - 251
    logging.info('')
    logging.info('If any test fails, run quality checks.')
    file.write('If any test fails, run quality checks.\n')
    file.close() 


# In[11]:

def run_tests_seconds(df, result_path, level, start_date, end_date):
    # Testing the dataset (TO-DO: use a proper unit testing library)
    file = open(result_path + 'Tests_EFT' + level + '-level_' + '[' + start_date + '-' + end_date + '].txt', "w")

    # First check to see if there are any gaps or duplicates.
    # The number of rows must be:
    # diff*60*24 in df[(df['timestamp'] > start_date) & (df['timestamp'] <= end_date+" 23:59:59")    approx
    number_market_minutes = 6.5 * (60 ** 2)  # 6.5 market hours
    # 4 bank holidays in this semester.get_datediff(start_date, end_date)/7*5 to exclude weekends.
    # TODO. a function that removes weekends and bank holidays from the number of days directly! 
    # So the condition of TEST1 is '== to' and not '< to'
    # df.date.dt.date.unique() instead of get_datediff(start_date, end_date)/7*5-4
    # with this we assume that we don't miss entire days. # TO-DO add extra test for days
    minutes_of_market = len(df[(df['datetime'] > start_date) & (
                df['datetime'] <= end_date + " 23:59:59")].datetime.dt.date.unique()) * number_market_minutes
    total_seconds = len(
        df[(df['datetime'] > start_date) & (df['datetime'] <= end_date + " 23:59:59")].between_time('09:30', '16:00'))
    test1 = minutes_of_market < total_seconds
    logging.warning('TEST 1: ' + str(test1) + ' - The amount of records is correct?')
    logging.warning(
        'Approx mins of market should be: ' + str(minutes_of_market) + ' and actual number is: ' + str(total_seconds))
    logging.info('TO-DO: Improve this test to be more accurate')
    file.write('TEST 1: ' + str(test1) + ' - The amount of records is correct?')
    file.write('Approx mins of market should be: ' + str(minutes_of_market) + ' and actual number is: ' + str(
        total_seconds) + '\n')
    file.write('TO-DO: Improve this test to be more acccurate.' + '\n')

    # Second check to see if there are any gaps
    ohlc_dict = {'close': 'last', 'high': 'max', 'low': 'min', 'open': 'first', 'volume': 'sum'}
    test = df.resample('1s').agg(
        ohlc_dict)  # This will create rows with NaN values if there are any minutes missing in the timerange
    test = test[(test.index > start_date) & (test.index <= end_date + " 23:59:59")]
    list_of_days = df.datetime.dt.date.unique()
    test['datetime'] = test.index.date
    test = test[test['datetime'].isin(list_of_days)].between_time('09:30',
                                                                  '16:00')  # only search for gaps in market hours
    consecutive_gaps = test.close.isnull().astype(int).groupby(
        test.close.notnull().astype(int).cumsum()).sum().sort_values(ascending=False)
    check = test[test['open'].isnull()]  # only displays gaps. if empty, happy days!
    test2 = len(test[test['open'].isnull()]) == 0
    logging.warning('TEST 2: ' + str(test2) + ' - Is the dataset complete? (no gaps at minute level).  ' + str(
        len(test[test['open'].isnull()])) + ' gaps.')
    file.write('TEST 2: ' + str(test2) + ' - Is the dataset complete? (no gaps at minute level).  ' + str(
        len(test[test['open'].isnull()])) + ' gaps.' + '\n')
    logging.warning('The max number of consecutive gaps at this level and this period is: ' + str(
        consecutive_gaps.max()) + ' gaps.' + '\n')
    logging.warning('A sample of the distribution of these consecutive gaps, sorted by descending order is: ' + '\n')
    logging.warning(consecutive_gaps[:25])  # just the 25 greatest values
    file.write('The max number of consecutive gaps at this level and this period is: ' + str(
        consecutive_gaps.max()) + ' gaps.' + '\n')
    file.write('A sample of the distribution of these consecutive gaps, sorted by descending order is: ' + '\n')
    file.write(str(consecutive_gaps[:25]))  # just the top 25 values

    # third check
    original_plus_gaps = (len(
        df[(df['datetime'] > start_date) & (df['datetime'] <= end_date + " 23:59:59")].between_time('09:30',
                                                                                                    '16:00')) + len(
        check))
    resampled = len(test) * 60
    test3 = (original_plus_gaps == resampled)
    logging.warning('TEST 3: ' + str(
        test3) + ' - Are the resampled (at second level) and the original dataset of the same size? (so # rows is ok)')
    logging.warning('original_plus_gaps is:' + str(original_plus_gaps) + ' and resampled is: ' + str(
        resampled) +
                '. If resampled is greater, it means gaps at second level (that may not be material at minute level).')
    file.write('TEST 3: ' + str(
        test3) + ' - Are the resampled (at second level) and the original dataset of the same size? (so # rows is ok)' +
               '\n')
    file.write('original_plus_gaps is:' + str(original_plus_gaps) + ' and resampled is: ' + str(
        resampled) +
               '. If resampled is greater, it means gaps at second level (that may not be material at minute level).' +
               '\n')

    # fourth check
    test4 = df.groupby(['timestamp']).count()['high']
    test4 = (len(test4[test4 > 1]) == 0)
    logging.warning('TEST 4: ' + str(test4) + ' - Is the dataset free from any duplicates?')
    file.write('TEST 4: ' + str(test4) + ' - Is the dataset free from any duplicates?' + '\n')
    # If there are any dups (Test 4 == False), check the given row (change timestamp below)
    # example=df[(df['timestamp'] == '2018-07-01 20:33:00')]
    # example

    logging.warning(
        'TEST 5: Number of market days (it should be an average of 251 market days a year +/-1 normally). # days is:' + str(
            len(list_of_days)))
    file.write(
        'TEST 5: Number of market days (it should be an average of 251 market days a year +/-1 normally). # days is:' + str(
            len(list_of_days)) + '\n')

    # TEST 6: Sacar el porcentaje de gaps con relacion al dataset
    # example 2014: 365 (number of days in 2014) – 104 (number of weekend days in 2014) – 9 (public holidays) = 252 days
    # 2015 - 252, 206 and 2017 - 251
    logging.info('')
    logging.info('If any test fails, run quality checks.')
    file.write('If any test fails, run quality checks.\n')
    file.close() 


# In[12]:
def define_global_settings(output_path, quarter, selected, year, level, dev=False):

    # Declaring attributes: for '1s' and DEVSET mode, QUARTER indicates month of the year 1-12.
    # The split is done, monthly due to memory constraints
    if dev or level == '1s':
        start_date, end_date = get_dates_month(year=year, month=quarter)
        year = str(year)
        quarter = str(quarter)
    elif 'min' not in level:
        start_date, end_date = get_dates_quarter(year, quarter)
        year = str(year)
        quarter = str(quarter)
    else:
        # For minute-level data, get the full year
        year = str(year)
        quarter = str(quarter)
        start_date = year + '-01-01'
        end_date = year + '-12-31'
        start_date = year + '-03-06'
        end_date = year + '-03-19'  # final date -1 (20 was final for here)

    # EFTs to take into account
    indexes_and_symbols = {
        'S&P500': 'SPY',
        'NASDAQ': 'QQQ',
        'DOWJONES': 'DIA',
        'IBEX': 'EWP',
        'EMB': 'EMB'
    }

    # select EFT to be processed
    eft = list(indexes_and_symbols.keys())[selected]
    symbol = indexes_and_symbols[eft]
    id = '[' + '_'.join([symbol, level, year, quarter]) + ']'

    # Paths
    result_path = os.sep.join([output_path + 'analysis', 'quantquote', f'{level}-level', f'{eft}{os.sep}'])

    # List of dates for files to be loaded ( +1 in periods is extra day in next month so the labels are complete)
    dates = pd.date_range(start_date, periods=(get_datediff(start_date, end_date) + 1), freq='D')
    result_filepath_preprocessed = result_path + eft + '_[' + str(dates[0])[:10] + '_to_' + str(dates[len(dates) - 1])[
                                                                                            :10] + '].csv.gz'
    result_filepath_processed = result_path + eft + '_(' + start_date + '_to_' + end_date + ')_indicators.csv'

    return result_filepath_processed, result_filepath_preprocessed, result_path, \
        symbol, year, quarter, dates, end_date, start_date, id


# In[]:
def read_and_merge_collection(config, dates, symbol):
    # Create list of DFs
    dataframes = []

    # Iterating through files
    for date in dates:
        folder_aux = str(date)[:10].replace('-', '')  # exclude time (00:00:00)  and remove dashes
        file_path = folder_aux + '/table_' + symbol.lower() + config['input_file_extension']
        # file_path = folder_aux + '/' + symbol.lower() + config['input_file_extension']
        csv_path = config['data_path'] + config['src_subdirectory'] + file_path

        # Only if file exists (only market days considered)
        file_check = Path(csv_path)
        print(csv_path)
        if file_check.exists():
            print(date)
            new_df = pd.read_csv(csv_path, sep=',', parse_dates=True, infer_datetime_format=True, header=None)
            # new_df.columns = ['milliseconds', 'open', 'high', 'low', 'close', 'volume', 'suspicious']
            new_df.columns = ['date', 'milliseconds', 'open', 'high', 'low', 'close', 'volume', 'suspicious', 'Dividends', 'Extrapolation']
            # new_df['datetime'] = pd.Timestamp(date)  # Get date from foldername
            # new_df['datetime'] = new_df.date + ' T' + new_df.time
            print(new_df['milliseconds'].head(5))

            # parsing hardcoded integers hh:mm to milliseconds
            new_df['milliseconds'] = new_df['milliseconds'].apply(lambda x: (int(str(x)[:-2])*60 + int(str(x)[-2:]))*60 * 1000)

            new_df = new_df.astype('double', copy=False)  # All values as Double
            new_df['datetime'] = pd.Timestamp(date)
            new_df.drop(columns=['Dividends', 'Extrapolation', 'date'], axis=1, inplace=True)
            dataframes.append(new_df)

    # concat all dataframes in a single one
    df = pd.concat(dataframes)
    df = df.drop_duplicates(['milliseconds', 'open', 'high', 'low', 'close', 'volume', 'suspicious',
                             'datetime'])
    print(df)

    del dataframes, new_df  # force deletion of intermediate stuff from memory

    return df


# In[]:
def apply_corrections(df):
    # Parse datetime at 00:00 to timestamp and multiply by 1000 to have milliseconds.
    # Then add raw timestamp with milliseconds from 00:00
    df['timestamp'] = ((df.datetime.values.astype(np.int64) // 10 ** 9) * 1000) + df.milliseconds
    # Timestamp in milliseconds to readable datetime
    df['timestamp_r'] = pd.to_datetime(df.timestamp, unit='ms')
    # Index dataframe by its actual readable timestamp and sort it
    df.index = df['timestamp_r']
    df = df.sort_index()
    # df.drop also drops the selected column (if value=1) or rows (if =0) #df = df.drop(df.columns[[0]], 1)
    # df = df.drop('milliseconds', 1)
    # df = dfinal_columnsf.drop('datetime', 1)
    # Adjusting decimal point
    for column in ['high', 'low', 'open', 'close']:
        df[column] = df[column].apply(lambda x: x / 10000)
    # Cast volume to integer
    # df.volume = df.volume.astype(int) # this may crash with some versions of Ta-Lib
    # Only keep market hours. Decide where should the cut-off be. I get data from 4am to 8pm...
    # Accoring to Quantquote, they may give hours before and after market if they have them.
    # But they give market hours from 9:30H to 16:00H. This is translated to 13.30H to 20:00H in Dublin time.
    # by now I get prices from 9H to 20H as the granularity before 9H is not that good.
    # Ideally it should be only market hours, but I'm too afraid to loose valuable data.
    # 08:00h = 28800000 in ms | 09:00h corresponds to millisecond 32400000 | 09:30h = 34200000 in ms
    # 16:00h = 57600000 in ms | 17:00h = 61200000 in ms | 20:00h corresponds to millisecond 72000000
    # df = df.loc[(df['milliseconds'] >= 17917000.0) & (df['milliseconds'] <= 71988000.0)]  # not limiting times yet
    # df
    return df


# In[]:
def resampling_data(df, end_date, level, start_date):
    resample_from_close = True  # ASC 05 June 2020   # TODO: read from config.yaml
    fill_missing_values = 0   # HANDLING GAPS: 0 - do nothing. 1 - standard. 2 - ffill  # todo: read from config.yaml

    # This would be the logic for downsampling/go to minute level from seconds.
    # For seconds, when upsampling like in our quantquote data, the values will be NULL anyway
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    # Handling missing data
    # Resample to 1 min level
    if resample_from_close:
        aux = df.close.resample(level).ohlc()  # due to noise at the second level in quantquote
    else:
        aux = df.resample(level).agg(ohlc_dict)

    # print(aux[aux['open'].isnull()])  # only displays gaps. if empty, happy days!
    aux = aux[(aux.index > start_date) & (aux.index <= end_date + " 23:59:59")]
    list_of_days = df.datetime.dt.date.unique()
    aux['datetime'] = aux.index.date

    # so we have enough data to not delete anything when parsing indicators,
    # we keep some bandwith before the market starts
    start_time = end_time = ''
    if level == '1s':  # 20 mins difference to cover gaps at second level
        start_time, end_time = '09:10', '16:20'
    elif 'min' not in level:  # we keep an hour before the market starts at second level
        start_time, end_time = '08:30', '16:20'
    else:  # we don't limit the time at minutes level, to give enough space to the 30-min level indicators resampling
        start_time, end_time = '00:00', '23:59'
    aux = aux[aux['datetime'].isin(list_of_days)].between_time(start_time, end_time)

    df = aux
    # Volume at 0 for missing seconds and propagate last closing price to the 4 price columns.
    df['gap'] = df.close

    if fill_missing_values == 1:
        # 1 if the record was a gap
        df.gap.where(df.gap.isnull(), 0, inplace=True)
        df['gap'] = df['gap'].fillna(1)
        # ################
        df['volume'] = df['volume'].fillna(0)
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['datetime'] = df.index
        df['timestamp'] = df.datetime.values.astype(np.int64) // 10 ** 9

    # Propagate all columns
    elif fill_missing_values == 2:
        # 1 if the record was a gap
        df.gap.where(df.gap.isnull(), 0, inplace=True)
        df['gap'] = df['gap'].fillna(1)
        # ################
        df['volume'] = df['volume'].ffill()
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].ffill()
        df['low'] = df['low'].ffill()
        df['high'] = df['high'].ffill()
    elif fill_missing_values == 0:
        df.dropna(inplace=True)
    df['datetime'] = df.index
    df['timestamp'] = df.datetime.values.astype(np.int64) // 10 ** 9
    return df


# In[]:
def create_feature_set(df):

    # Add parameters to transform in TS
    timeseries = ['low', 'close', 'open', 'high', 'volume']

    # Length of the TS. How many values do we keep per serie.
    # e.g. 1 -> t / 2 -> t,t-1 / 3 -> t,t-1,t-2 / 4 -> t,t-1,t-2,t-3
    use_OHLCV_timeseries = True
    OHLCV_timeseries_lenght = 5

    # Add lagged times
    if (use_OHLCV_timeseries):
        for column in timeseries:
            # df[column+'_t']=df[column]
            for i in range(1, OHLCV_timeseries_lenght):
                df[column + '_t-' + str(i)] = df[column].shift(i)  # it could also be sorted and group by if needed
            # del drops the delected df column
            # del df[column]

    # All the numbers here and below assume a default time period for ta params of 10 mins averages
    #  change the other numbers if the mins number is changed.
    #  (Params of 5min and 20min and the removal of 20 first mins of the day below)
    # In that case change as well the time tag '_10' with the corresponding one.
    default_timerange = 10
    # Set extra timeranges for moving averages
    extra_timeranges = [default_timerange / 2, default_timerange, default_timerange * 2, default_timerange * 3]

    # ###########################################
    # Iterate and run list of indicators selected
    # All of them produced for 25 prior mins of data
    # ###########################################
    indicator = define_indicators()
    for key in list(indicator.keys()):
        # For indicators that only return one column
        # (this will need to be modified depending on the selection of indicators)
        if key in ['ema', 'sma', 'wma', 'trima']:  # ,'macdfix']:
            for timerange in extra_timeranges:
                df[key + '_' + str(timerange)] = indicator[key](df, timeperiod=timerange)
        elif key not in ['bbands', 'aroon', 'stoch', 'macd', 'macdfix', 'stochf']:
            df[key + '_' + str(default_timerange)] = indicator[key](df, timeperiod=default_timerange)
        # Otherwise check the list of columns and return all
        else:
            key_output = indicator[key](df, timeperiod=default_timerange)  # , price='close')
            for j in range(0, len(list(key_output.columns))):
                df[key + '_' + key_output.columns[j]] = key_output[key_output.columns[j]]


# In[]:
def label_dataset(df, end_date, start_date):

    def set_binary_label(row):
        if row['close_t+1'] > row['close']:
            return 1
        else:
            return 0

    def set_new_label(row):
        if row['close_t+1'] > row['close']:
            return 1
        elif row['close_t+1'] == row['close']:
            return 0
        elif row['close_t+1'] < row['close']:
            return -1

    # One minute ahead closing price
    df['close_t+1'] = df['close'].shift(-1)
    df['gap_t+1'] = df['gap'].shift(-1)

    # Creating label/y to be predicted / independent (predicted) feature 'y'
    df['binary_label'] = df.apply(set_binary_label, axis=1)  # binary_label
    df['label'] = df.apply(set_new_label, axis=1)

    # Filter out irrelevant dates once the dataset is complete.
    df = df[(df['datetime'] > start_date) & (df['datetime'] <= end_date + " 23:59:59")]
    print(len(df))
    df = df.between_time('09:30', '16:00', include_end=True)
    print(len(df))
    # 48875/45125 nasdaq
    # 48875/45125 spy
    return df


# In[ ]:
def save_result(result_filepath_processed, df):

    # Select columns for output
    columns_selected = ['datetime',
                        'rsi_10', 'willr_10', 'adosc_10', 'macd_macd', 'cci_10', 'mom_10',
                        'stoch_slowk', 'stoch_slowd',
                        'sma_5', 'sma_10', 'sma_20', 'sma_30',
                        'wma_5', 'wma_10', 'wma_20', 'wma_30',
                        'ema_5', 'ema_10', 'ema_20', 'ema_30',
                        'trima_5', 'trima_10', 'trima_20', 'trima_30',
                        'adx_10', 'bbands_upperband', 'bbands_middleband', 'bbands_lowerband', 'obv_10',
                        'roc_10', 'rocr_10', 'stochf_fastk', 'stochf_fastd',
                        'aroon_aroondown', 'aroon_aroonup', 'medprice_10', 'typprice_10', 'wclprice_10',
                        'atr_10', 'macdfix_macd', 'mfi_10', 'sar_10', 'ppo_10',
                        'volume', 'volume_t-1', 'volume_t-2', 'volume_t-3', 'volume_t-4',
                        'close', 'close_t-1', 'close_t-2', 'close_t-3', 'close_t-4',
                        'high', 'high_t-1', 'high_t-2', 'high_t-3', 'high_t-4',
                        'open', 'open_t-1', 'open_t-2', 'open_t-3', 'open_t-4',
                        'low', 'low_t-1', 'low_t-2', 'low_t-3', 'low_t-4',
                        'binary_label',
                        'label', 'gap_t+1', 'close_t+1']
    # Export processed data
    output = pd.DataFrame(df, columns=columns_selected)
    output.to_csv(result_filepath_processed + ".gz", sep=';', index=False, compression='gzip')


# In[ ]:
def print_class_distribution(result_path, symbol, df, end_date, level, start_date):
    # Printing classes distributions
    ############
    label_zero = len(df[df['binary_label'] == 0])
    label_one = len(df[df['binary_label'] == 1])
    ############
    label_z0 = len(df[(df['label'] == 0) & (df['gap_t+1'] == 0)])
    label_o0 = len(df[(df['label'] == 1) & (df['gap_t+1'] == 0)])
    label_t0 = len(df[(df['label'] == 2) & (df['gap_t+1'] == 0)])
    label_z1 = len(df[(df['label'] == 0) & (df['gap_t+1'] == 1)])
    label_o1 = len(df[(df['label'] == 1) & (df['gap_t+1'] == 1)])
    label_t1 = len(df[(df['label'] == 2) & (df['gap_t+1'] == 1)])
    ############
    # Save class distribution
    file = open(
        result_path + 'Class_distribution_EFT' + symbol + '_' +
        level + '-level_' + '[' + start_date + '-' + end_date + '].txt', "w")
    file.write("Binary Label: 0=remains or decreases / 1=increases")
    file.write("0 in " + str(100 * float(label_zero) / (label_one + label_zero)) + "%")
    file.write("1 in " + str(100 * float(label_one) / (label_one + label_zero)) + "%")
    file.write("---")
    # file.write("NON-GAP in T+1: Percentage of records that don't occur before a gap splitted by label")
    # file.write("0 in " + str(
    #     100 * (float(label_z0) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    # file.write("1 in " + str(
    #     100 * (float(label_o0) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    # file.write("2 in " + str(
    #     100 * (float(label_t0) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    # file.write(' ')
    # file.write("GAP in T+1: Percentage of records that occur before a gap splitted by label")
    # file.write("0 in " + str(
    #     100 * (float(label_z1) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    # file.write("1 in " + str(
    #     100 * (float(label_o1) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    # file.write("2 in " + str(
    #     100 * (float(label_t1) / (label_o0 + label_z0 + label_t0 + label_o1 + label_z1 + label_t1))) + "%")
    file.write('\n')
    file.close()


# in[ ]:
def plot_prices_and_volumes(quarter, result_path, symbol, year, df, level):
    # now plot close price and volume overtime.
    df.plot(x="datetime", y=["open", "close", "sma_10", "sma_30"])
    # plt.show()
    plt.savefig(result_path + 'prices_' + symbol + '_lvl' + level + '_' + str(year) + (
        ('_' + str(quarter)) if 'min' not in level else '') + '.pdf', bbox_inches='tight')
    df.plot(x="datetime", y=["volume"])
    # plt.show()
    plt.savefig(result_path + 'volume_' + symbol + '_lvl' + level + '_' + str(year) + (
        ('_' + str(quarter)) if 'min' not in level else '') + '.pdf', bbox_inches='tight')


# In[ ]:
def create_dataset(config, level, selected, year, quarter=None, dev=False):
    """
    This is the main function of the script. For each period, it collects files from raw csvs and creates the dataset.
    """

    # Define paths and parameter values
    result_filepath_processed, result_filepath_preprocessed, result_path, \
        symbol, year, quarter, dates, \
        end_date, start_date, id = define_global_settings(config['output_path'], quarter, selected, year, level, dev)

    logging.info(id + '  1/8 - Putting together and cleaning dataset')

    # @ asuarez: Comment about the design choice:
    # It would be interesting to compute indicators and labels for every single day and then merge the subsets.
    # Although this wouldn't work in terms of corrections in the dataset or duplicates,
    # in quantquote the data is clean already.
    # Futhermore, a a real online incremental trading system wouldn't be able to apply the corrections and
    # it would have to deal with the noise as it is.
    # At the end I left it as it is because I can always delete the rows afterwards.
    df = read_and_merge_collection(config, dates, symbol)
    df = apply_corrections(df)

    logging.info(id + '  2/8 - Running tests')
    if 'min' in level:
        run_tests_minutes(df, result_path, level, start_date, end_date)
    else:
        run_tests_seconds(df, result_path, level, start_date, end_date)

    logging.info(id + '  3/8 - Sampling data at the required level and saving intermediate results')
    df = resampling_data(df, end_date, level, start_date)
    df.to_csv(result_filepath_preprocessed.replace(']', ']_no_missing_data'), sep=';', index=False, compression='gzip')
    #
    # logging.info(id + '  4/8 Creating feature set: computing technical indicators and time series')
    # create_feature_set(df)
    #
    # logging.info(id + '  5/8 Labeling dataset')
    # df = label_dataset(df, end_date, start_date)
    #
    # logging.info(id + '  6/8 - Saving result in CSV format')
    # save_result(result_filepath_processed, df)

    # logging.info(id + '  7/8 - Printing class distribution')
    # print_class_distribution(result_path, symbol, df, end_date, level, start_date)

    # logging.info(id + '  8/8 - Plotting prices and volumes')
    # plot_prices_and_volumes(quarter, result_path, symbol, year, df, level)


# In[ ]:
def iterate_through_periods(eft, lvl, years, n_periods, config):
    """
    This contains a loop to iterative through the period specified
    :param eft: eft to be processed
    :param lvl: level of sampling for the dataset to create
    :param n_periods: number of periods to iterate through. If monthly, 12. IF QUARTERLY, 4. OTHERWISE 1.
    :param years
    :param config
    """
    for yr in years:
        logging.info('-----------')
        logging.info('Year: ' + str(yr))
        for p in range(n_periods):
            logging.info('Starting ' + ('month' if lvl == '1s' else 'quarter' if 'min' not in lvl else 'iteration') +
                         ':' + str((p + 1)))
            create_dataset(config=config, level=lvl, selected=eft, year=yr, quarter=None if 'min' in lvl else p + 1)


# In[ ]:
def compute():

    # Load global parameters as paths, symbols and periods to iterate through
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())

    # Creating a global dictionary of functions for every technical indicator
    define_indicators()

    # If data needs to be unzipped, do it
    if config['uncompress']:
        print(uncompress_src(config['zipfile_path']))  # TODO. remove folders at the end

    for eft in config['efts']:

        logging.info('--------------------------')
        logging.info('--------------------------')
        logging.info('')
        logging.info('PROCESSING EFT: '+str(eft))
        logging.info('')

        for lvl in config['levels']:
            logging.info('###############')
            logging.info('###############')
            logging.info('Level: '+lvl)

            if config['dev_period']:
                logging.info('Processing devset:')
                create_dataset(config=config, level=lvl, selected=eft, year=2014, quarter=12, dev=True)
            else:
                # Train & test
                # For second level, iterate for every quarter. Monthly for 1s level due to memory related issues
                n_periods = 12 if lvl == '1s' else 4 if 'min' not in lvl else 1
                iterate_through_periods(eft, lvl, config['years'], n_periods, config)


if __name__ == '__main__':
    compute()
