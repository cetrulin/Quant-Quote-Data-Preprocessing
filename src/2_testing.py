# -*- coding: utf-8 -*-

# Data preparation at one-second level for Ph.D thesis
# @author: Andres L. Suarez-Cetrulo

import glob
import time
import logging
import yaml
import subprocess
import os
import pandas as pd
import numpy as np
import datetime

# Global attributes
SLASH = os.path.sep
EQUIVALENCE = {
    's': 1,
    'min': 60
}

# logging.getLogger().addHandler(logging.StreamHandler())  # for debugging while coding / comment out otherwise
LOGLEVEL = logging.DEBUG  # logging.WARNING  # logging.DEBUG


def get_datetime_format(date):
    return time.strptime(date, "%Y-%m-%d")


def show_missing_values(dates, df_name, df):

    logging.debug('Printing list of dates with missing values in ' + df_name + ' level (e.g. NA and nulls).')
    logging.debug(str(np.unique(dates.index.time)))
    logging.debug('Showing full DF for only rows with missing values in ' + df_name + ' level')
    logging.debug(df[df.isnull().any(axis=1)].to_string())

    logging.debug('Returning the list of columns which have missing values in ' + df_name + ' level')
    nulls = list()
    for col, bool_var in df.isna().any().items():
        if bool_var:
            nulls.append(col)
    logging.debug(df.isna().any().keys())


def set_logging(config):
    # Set logging level
    logname = str(config['efts']).replace('\'','') + ' Tests across files starting with ' + config['levels'][0] + \
              ' level and period (' + config['start'] + '-' + config['end'] + ').log'
    logging.basicConfig(filename=config['path'] + logname)
    logger = logging.getLogger()
    logger.setLevel(LOGLEVEL)
    logging.info('')


def log_new_execution(config):
    logging.info('')
    logging.info('')
    logging.info('')
    logging.info('')
    logging.info('')
    logging.info('#########################')
    logging.info('')
    logging.info('')
    logging.info('NEW EXECUTION AT: '+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('')
    logging.info('')
    logging.info('#########################')
    logging.info('Logging config settings:')
    logging.info(config)


def log_iteration(config):
    logging.info(' --------------------------------------------------------------  ')
    logging.info(' START OF TEST FOR ' + str(config['eft']).replace('\'','') +
                 ' ' + config['level'] + ' level on period: ' +
                 config['start'] + ' to ' + config['end'])
    logging.info(' --------------------------------------------------------------  ')


def load_config():
    # Load global parameters as paths, symbols and periods to iterate through
    with open(os.path.sep.join(['.', 'config.yaml']), 'r') as f:
        config = yaml.safe_load(f.read())['merging']

    # Path where every individual folder is located
    config['path'] = config['src_path']  # + config['level'] + '-level' + SLASH + config['eft'] + SLASH
    config['filename_pattern'] = config['eft'] + '_*_indicators.csv.gz'
    config['start_date'] = get_datetime_format(config['start'])
    config['end_date'] = get_datetime_format(config['end'])
    config['end_date'] = get_datetime_format(config['end'])
    config['name'] = '_'.join([config['eft'], config['output_subname'], config['level']])+'-level'
    config['output'] = config['path'] + config['name']

    # The feature set in 15min and 30min levels differ to avoid considering dependencies across days
    config['featureset'] = config['columns'+nconf['level']] if config['level'] in ['15min', '30min'] \
        else config['columns']

    return config


def new_config(nconf):
    """ Only use this for testing and not creating of .arff, as we add the field gapt+1 to the featureset"""
    # Path where every individual folder is located
    nconf['path'] = nconf['src_path'] + nconf['level'] + '-level' + SLASH + nconf['eft'] + SLASH
    nconf['filename_pattern'] = nconf['eft'] + '_*_indicators.csv.gz'
    nconf['start_date'] = get_datetime_format(nconf['start'])
    nconf['end_date'] = get_datetime_format(nconf['end'])
    nconf['name'] = '_'.join([nconf['eft'], nconf['output_subname'], nconf['level']])+'-level'
    nconf['output'] = nconf['path'] + nconf['name']

    # The feature set in 15min and 30min levels differ to avoid considering dependencies across days
    nconf['featureset'] = nconf['columns'+nconf['level']] if nconf['level'] in ['15min', '30min'] \
        else nconf['columns']

    if 'gap_t+1' not in nconf['featureset']:
        nconf['featureset'].append('gap_t+1')

    return nconf


def load_data(conf, min_time=''):
    # Iterate through collection of processed data with indicators
    df_full = pd.DataFrame()

    for filename in sorted(glob.glob(conf['path'] + conf['filename_pattern'])):
        # Extract dates from filename
        aux_dates = filename.replace(conf['path'] + conf['eft']+'_(', '')
        aux_dates = aux_dates.replace(')_indicators.csv.gz', '')
        dates = aux_dates.split('_to_')

        # If file in date range, add it to DF
        if get_datetime_format(dates[0]) >= conf['start_date'] and \
                get_datetime_format(dates[1]) <= conf['end_date']:
            logging.info('Importing data period ' + str(dates[0]) + ' to ' + str(dates[1]) + '.')
            new_df = pd.read_csv(filename, encoding="utf-8", index_col=0, sep=";")
            new_df['datetime'] = new_df.index  # save string readable timestamp as a column
            new_df.index = pd.to_datetime(new_df.index)
            print('Preview of 5 fist rows of file being load.')
            print(new_df.head(5))

            # If required, remove times if there are dependencies from previous days,
            #  or missing values in certain dates that we should avoid
            if min_time == '':
                print('Rows removed at this level: '+str(conf['rows_per_day_to_remove'][str(conf['level'])]))
                print(conf['rows_per_day_to_remove'][str(conf['level'])])
                min_time = '09:30' if int(conf['rows_per_day_to_remove'][str(conf['level'])]) == 0 \
                                   else get_min_time(conf, new_df)
            print('Current minimum time being considered: '+str(min_time))
            new_df = new_df.between_time(min_time, '16:00')
            df_full = df_full.append(new_df)

    # logging.debug('Printing a sample of DF at level: '+conf['level'])
    # logging.debug(df)
    logging.info('Data read.')
    logging.warning('WARNING: Now make sure that the count of rows makes sense for the number of market days:')
    logging.warning(df_full.groupby([df_full.index.year, df_full.index.month, df_full.index.day]).agg({'count'}).to_string())

    return df_full[conf['featureset']], min_time


def get_min_time(conf, df):

    # Return the time of the minimum datetime
    return min(df[int(conf['rows_per_day_to_remove'][str(conf['level'])]):].index.tolist()).time()


def split_level(txt, seps):  # this function would make sense if we had  tick, hourly or daily level also.
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    level = ''
    for sep in seps[1:]:
        aux_len = len(txt)
        txt = txt.replace(sep, default_sep)
        level = sep if len(txt) != aux_len else level  # if it has changed, then we keep the level string

    level_split = [i.strip() for i in txt.split(default_sep)]
    level_split[1] = default_sep if level == '' else level
    return level_split


def plot_stats(cl_df, ol_df, scl, sol):
    logging.info('Iterated level DF has a length of: '+str(len(ol_df))+' versus a length of: '+str(len(cl_df)) +
                 '. The equivalence should be of: cdf = '+str(float(scl)/float(sol))+'*odf')
    logging.info('Current DF cdf stats: '+cl_df.describe().to_string())
    logging.info('Compared DF odf stats: '+ol_df.describe().to_string())


def test_subsets(df1, df2):

    # checking that datetime for the current level exists in inferior levels
    key_diff = set(df1.index).difference(df2.index)
    where_diff = df1.index.isin(key_diff)
    logging.info('Logging difference:')
    logging.info(df1[where_diff])

    assert len(df1[where_diff]) == 0
    logging.debug('Test 2 PASSED: No missing subsets (rows based on datetime) when looking at lower levels.')


def test_number_of_rows(cl_df, ol_df, comparison_type, scl, sol):

    if comparison_type == 'eft':
        cl_df.index = pd.to_datetime(cl_df.index)
        logging.debug(len(np.unique(cl_df.index.date)))

        len_cdf = len(cl_df)
        len_odf = len(ol_df)

        logging.debug('Size of c_eft: '+str(len_cdf))
        logging.debug(cl_df.head())
        logging.debug('Size of o_eft: '+str(len_odf))
        logging.debug(ol_df.head())

        cl_df['datetime'] = cl_df.index
        print(cl_df[cl_df['datetime'] == '2015-04-01 09:41:15'])

        test = len_cdf == len_odf
        logging.info('')
        logging.info('Test 0: Length of both EFTs have the same length for the same periods and level: ' +
                     ' PASSED' if test else ' NOT PASSED')
        logging.info('')
        test_subsets(ol_df, cl_df)

        assert test
    else:

        # Test not ready. It only works for single dates.
        from collections import Counter

        cl_df['stridx'] = cl_df.index
        ol_df['stridx'] = ol_df.index

        logging.info(pd.Series(cl_df['stridx'].str.split(' ')[0].map(Counter).sum()))
        logging.info(len(cl_df[cl_df['stridx'].str.contains('2015-01-02')]))
        logging.info(len(ol_df[ol_df['stridx'].str.contains('2015-01-02')]))

        logging.infologging.info(len(cl_df.loc['2015-01-02' in cl_df.index]))
        logging.info(len(ol_df.loc['2015-01-02' in ol_df.index]))

        logging.info('---')
        logging.info(len_cdf * (float(scl)/float(sol)) * len(np.unique(cl_df.index.date)) - (float(scl)/float(sol) * len(np.unique(cl_df.index.date))) + len(np.unique(cl_df.index.date)))
        ## for a single day
        logging.infologging.info(len_cdf * float(scl)/float(sol) - float(scl)/float(sol) + 1)
        logging.info(len_odf * sol)

        # for a single day
        #assert len_cdf * float(scl)/float(sol) - float(scl)/float(sol) + 1 == len_odf * sol


def max_and_min_dates(cl_df, ol_df):
    cl_df['datetime'] = pd.to_datetime(cl_df.index)
    ol_df['datetime'] = pd.to_datetime(ol_df.index)

    logging.info('Max and min dates in current DF are: ' +
                 str(cl_df['datetime'].min())+' - '+str(cl_df['datetime'].max()))
    logging.info('Max and min dates in compared DF are: ' +
                 str(ol_df['datetime'].min())+' - '+str(ol_df['datetime'].max()))


def any_nulls(cl_df, ol_df):

    # Exclude last row from comparisons as it may be null for lack of values after 4pm
    cl_df.drop(cl_df.tail(1).index, inplace=True)  # drop last n rows
    ol_df.drop(ol_df.tail(1).index, inplace=True)  # drop last n rows

    logging.info('Confirming that both currend compared models don\'t have missing values:')
    passed = (len(cl_df[cl_df.isnull().any(axis=1)]) == 0 and len(ol_df[ol_df.isnull().any(axis=1)]) == 0)
    logging.info(('PASSED' if passed else 'NOT PASSED'))

    if ~passed:
        aux_df = cl_df[cl_df.isnull().any(axis=1)]
        aux_df.index = pd.to_datetime(aux_df.index)
        show_missing_values(dates=aux_df, df_name='current', df=cl_df)
        aux_df = ol_df[ol_df.isnull().any(axis=1)]
        aux_df.index = pd.to_datetime(aux_df.index)
        show_missing_values(dates=aux_df, df_name='compared', df=ol_df)

    assert passed


def get_percentage_of_gaps(cl_df, ol_df):

    logging.debug('--------------------------------------------')
    logging.debug('Printing percentage of gaps in current level')
    logging.info(len(cl_df[cl_df['gap_t+1'] == 1])/len(cl_df))

    logging.debug('Printing percentage of gaps in compared level')
    logging.info(len(ol_df[ol_df['gap_t+1'] == 1])/len(ol_df))
    logging.debug('')
    logging.warning('Are the two values above similar enough when compared to the previous testing logs? Manual check.')
    logging.debug('--------------------------------------------')


def compare_levels(cl_df, ol_df, c_level, o_level, featureset):

    logging.info('')
    logging.info('#################################')
    logging.info('')
    logging.info('o_level: '+str(o_level))
    logging.info('c_level: '+str(c_level))
    logging.info('')

    # second equivalence of current level and the other level to compare against in the current iteration
    scl = int(c_level[0]) * EQUIVALENCE[c_level[1]]
    sol = int(o_level[0]) * EQUIVALENCE[o_level[1]]

    # First unit test
    logging.info('Test 1: Stats and counts - Judge manually below to determine if it PASSED. Different number of ' +
                 'columns across levels it\'s ok, as it is a design-made choice.')
    logging.info('Please, check that the difference in amount of rows for the same date period makes sense.')
    logging.info('For a single date, this should be truth: ' +
                 'len_current_df * float(scl)/float(sol) - float(scl)/float(sol) + 1 == len_compared_df * sol')
    logging.info('scl and sol are the amount of seconds of the current and compared level respectively. '+
                 'If current_df level =30 min level, then scl=30*60.')
    plot_stats(cl_df, ol_df, scl, sol)

    # if the other level to compare against is lower than the current one, trigger the following tests
    # added extra checks just in case
    if sol < scl and int(c_level[0]) % int(o_level[0]) == 0 and c_level[1] == o_level[1]:
        logging.info('Test 2: Check if lower levels miss any row that they shouldn\'t')
        test_subsets(cl_df, ol_df)

    logging.info('Test 3: Max and Min dates:')
    max_and_min_dates(cl_df, ol_df)

    logging.info('Test 4: Any null values?')
    any_nulls(cl_df[featureset], cl_df[featureset])

    logging.info('Test 5: Percentage of gaps. ' +
                 'It should map the percentages that we already have in the same files ' +
                 'regarding to the class distribution.')
    get_percentage_of_gaps(cl_df, ol_df)


def compare_efts(ce_df, oe_df, c_eft, o_eft):

    logging.info('')
    logging.info('#################################')
    logging.info('')
    logging.info('o_eft: '+str(o_eft))
    logging.info('c_eft: '+str(c_eft))
    logging.info('')

    logging.info('Test 0: Stats and counts - Judge manually below to determined if it PASSED. ' +
                 'IBEX may be the most different one due to bank holidays in Spain.')
    test_number_of_rows(ce_df, oe_df, 'eft', 1, 1)


def run_tests_for_config(config):


    # Load all the data for the current level and period in a Dataframe
    cl_df, min_time = load_data(config)

    logging.info('First part in testing, comparing to efts at same level:')
    for other_eft in config['compare_efts']:
        logging.info(' ///// ')
        logging.info('Comparing to efts: '+other_eft)

        # Load DF to compare against
        oe_config = config.copy()
        oe_config['eft'] = other_eft
        oe_df, _ = load_data(new_config(oe_config))  # only getting DF, as min_time doesn't need to update
        compare_efts(cl_df, oe_df, config['eft'], oe_config['eft'])
        logging.info(' ///// ')

    logging.info('Second part in testing, comparing different levels of the same EFT and period:')
    c_level = split_level(config['level'], list(EQUIVALENCE.keys()))
    for other_level in config['compare_against']:
        logging.info(' ///// ')
        logging.info('Comparing to level: '+other_level)

        # Load DF to compare against
        ol_config = config.copy()
        ol_config['level'] = other_level
        # new_config updates some config settings that are dependant on the values changed
        ol_df, _ = load_data(new_config(ol_config), min_time)  # only getting DF, as min_time doesn't need to update

        # Tests
        o_level = split_level(other_level, list(EQUIVALENCE.keys()))

        compare_levels(cl_df, ol_df, c_level, o_level, config['featureset'])
        logging.info(' ///// ')


def run_tests():

    # Load settings and set logging config
    config = load_config()
    config['featureset'].append('gap_t+1')  # only for testing (not merging)
    set_logging(config)
    log_new_execution(config)

    for eft in config['efts']:
        print()
        print('EFT: '+str(eft))
        print('------')

        for period in config['periods'].values():
            print('')
            print('Period: '+str(period))
            print('')

            for level in config['levels']:
                print('Level: '+str(level))
                config['eft'] = eft
                config['start'] = period[0]
                config['end'] = period[1]
                config['level'] = level
                log_iteration(config)
                run_tests_for_config(new_config(config))


if __name__ == '__main__':
    run_tests()
