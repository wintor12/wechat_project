
"""
Prepare data
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/', help='output file for the prepared data')
parser.add_argument('--train_data', default='training.xlsx', help='file name of training data')
parser.add_argument('--test_data', default='test.xlsx', help='file name of test data')
opt = parser.parse_args()
print(opt)

# Save dataframe to .txt file
def saveToTxt(df, file_name):
    df.to_csv(file_name, header=None, index=None, sep=' ')

# Get xs and ys (columns '发布位置','标题', '正文', '阅读数','发布时间')
def getData(df):
    position, title, body, y, time = df['发布位置'], df['标题'], df['正文'], df['阅读数'], df['发布时间']
    return position, title, body, y, time

print('Start to prepare data...')
# Read excel file to dataframe
df_train = pd.read_excel(opt.data + opt.train_data, sheetname='list')
df_test = pd.read_excel(opt.data + opt.test_data, sheetname='list')
print('Before cleaning: train | test')
print(len(df_train.index), len(df_test.index))
# Clean data
df_train.dropna(axis=0, how='any', subset=['正文','标题'], inplace=True)
df_test.dropna(axis=0, how='any', subset=['正文','标题'], inplace=True)
print('After cleaning: train | test')
print(len(df_train.index), len(df_test.index))

position_train, title_train, body_train, y_train, time_train = getData(df_train)
position_test, title_test, body_test, y_test, time_test = getData(df_test)

# Save data sets to .txt files
saveToTxt(position_train, opt.data + 'position_train.txt')
saveToTxt(title_train, opt.data + 'title_train.txt')
saveToTxt(body_train, opt.data + 'body_train.txt')
saveToTxt(time_train, opt.data + 'time_train.txt')
saveToTxt(y_train, opt.data + 'y_train.txt')
saveToTxt(position_test, opt.data + 'position_test.txt')
saveToTxt(title_test, opt.data + 'title_test.txt')
saveToTxt(body_test, opt.data + 'body_test.txt')
saveToTxt(time_test, opt.data + 'time_test.txt')
saveToTxt(y_test, opt.data + 'y_test.txt')

print('Data preparation done.')