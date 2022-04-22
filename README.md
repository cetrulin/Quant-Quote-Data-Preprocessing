# Quant-Quote-Data-Preprocessing



This repository contains the code for data preprocessing used in the PhD thesis ''*Adaptive Algorithms for Classification on High-Frequency Data Streams: Application to Finance*''.

- The script *[1_rawdata_processing.py](https://github.com/cetrulin/Quant-Quote-Data-Preprocessing/blob/master/src/1_rawdata_processing.py)* prepares prices from www.quantquote.com in the desired frequency level. To run it, it's required Python 2.7. This script was also used for the paper "*[Incremental Market Behavior Classification in Presence of Recurring Concepts](https://doi.org/10.3390/e21010025)*".

  To run this script, the frequency levels, periods to be processed, stock or ETF symbols, and input and output paths must be specified in the file *config.yaml*.

- The script [*3_merge_periods_and_create_arff.py*](https://github.com/cetrulin/Quant-Quote-Data-Preprocessing/blob/master/src/3_merge_periods_and_create_arff.py) merges CSVs processed from different periods, selects the desired columns (indicators), and generates ARFF files for WEKA / MOA.

The other scripts contained in *src* perform tests, create technical indicators (TI), and automate the selection of the mahalanobis sets.

