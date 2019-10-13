# -*- coding: utf-8 -*-

# Data preparation at one-second level for Ph.D thesis
# @author: Andres L. Suarez-Cetrulo

import pandas as pd
import glob
import time
import logging
import yaml
import subprocess
import os


# Global attributes
SLASH = os.path.sep


def get_datetime_format(date):
    return time.strptime(date, "%Y-%m-%d")


def load_config():
    # Load global parameters as paths, symbols and periods to iterate through
    with open(os.path.sep.join(['.','config.yaml']), 'r') as f:
        config = yaml.safe_load(f.read())['merging']

    # Path where every individual folder is located
    config['path'] = config['src_path'] + config['level'] + '-level' + SLASH + config['eft'] + SLASH
    config['filename_pattern'] = config['eft'] + '_*_indicators.csv.gz'
    config['start_date'] = get_datetime_format(config['start'])
    config['end_date'] = get_datetime_format(config['end'])
    config['name'] = '_'.join([config['eft'], config['output_subname'], config['level']])+'-level'
    config['output'] = config['path'] + config['name']

    # The feature set in 15min and 30min levels differ to avoid considering dependencies across days
    config['featureset'] = config['columns'+config['level']] if config['level'] in ['15min', '30min'] \
        else config['columns']

    return config


def save_dataset(config, df):
    logging.info('Saving csv file...')
    if config['export_csv']:
        df.to_csv(config['path'] + config['name'] + '.csv.gz', sep=',', index=False, compression='gzip')

    if config['export_arff']:
        df.to_csv(config['path'] + config['name'] + '.csv', sep=',', index=False)  # excluding index -> datetime
        logging.info('Creating arff file...')
        create_arff_file(config)
        delete_csv(config)


def create_arff_file(config):
    command = ['java', config['java_mem'], '-classpath', config['wekadev_libpath'],
               'weka.core.converters.CSVLoader', config['output'] + '.csv', '>', config['output'] + '.arff']
    log_and_print('--')
    log_and_print('If the arff is not generated, run the next command in the terminal: '+'\n')
    log_and_print(str(' '.join(command)))
    log_and_print('--')
    f = open(config['output'] + '.arff', "w")
    subprocess.call(command, stdout=f)


def log_and_print(string):
    print(string)
    logging.info(string)


def delete_csv(config):
    run_bash(['rm', '-rvf', (config['output'] + '.csv')])


def run_bash(command):
    try:
        # print(command)
        byteoutput = subprocess.check_output(command)
        return byteoutput.decode('UTF-8').rstrip()
    except subprocess.CalledProcessError as e:
        print("Error running command.", e.output)
        return None


def get_min_time(conf, df):

    # Return the time of the minimum datetime
    return min(df[int(conf['rows_per_day_to_remove'][str(conf['level'])]):].index.tolist()).time()


def merge():
    # Set logging level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Load settings and declare DF
    df = pd.DataFrame()
    config = load_config()

    # Iterate through collection of processed data with indicators
    for filename in sorted(glob.glob(config['path'] + config['filename_pattern'])):
        # Extract dates from filename
        aux_dates = filename.replace(config['path'] + config['eft']+'_(', '')
        aux_dates = aux_dates.replace(')_indicators.csv.gz', '')
        dates = aux_dates.split('_to_')

        # If file in date range, add it to DF
        if get_datetime_format(dates[0]) >= config['start_date'] and \
                get_datetime_format(dates[1]) <= config['end_date']:
            logging.info('Importing data period ' + str(dates[0]) + ' to ' + str(dates[1]) + '.')
            new_df = pd.read_csv(filename, encoding="utf-8", index_col=0, sep=";")
            new_df['datetime'] = new_df.index  # save string readable timestamp as a column
            new_df.index = pd.to_datetime(new_df.index)

            # If required, remove times if there are dependencies from previous days,
            #  or missing values in certain dates that we should avoid.
            min_time = '09:30' if int(config['rows_per_day_to_remove'][str(config['level'])]) == 0 \
                else get_min_time(config, new_df)
            df = df.append(new_df.between_time(min_time, '16:00')[config['featureset']])

    save_dataset(config, df)
    logging.info('Done!')


if __name__ == '__main__':
    merge()
