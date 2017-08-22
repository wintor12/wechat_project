"""
Prepare data
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/', help='output file for the prepared data')
parser.add_argument('--raw_data', default='北美留学生.xlsx', help='file name of raw data')
parser.add_argument('--train_percent', default=0.8, help='percentage of training set')
parser.add_argument('--valid_percent', default=0.1, help='percentage of validation set')
parser.add_argument('--test_percent', default=0.1, help='percentage of test set')
parser.add_argument('--lda', default='True', help='preparing data for LDA model')
opt = parser.parse_args()
print(opt)


# Save dataframe to .txt file
def saveToTxt(df, file_name):
    df.to_csv(file_name, header=None, index=None, sep=' ')


# Get xs and ys (columns '标题', '正文', '点赞数')
def getData(df):
    title, body, y, v = df['标题'], df['正文'], df['点赞数'], df['阅读数']
    return title, body, y, v


print('Start to prepare data...')
# Read excel file to dataframe
df = pd.read_excel(opt.data + opt.raw_data, sheetname='list')
# Clean data
df.dropna(axis=0, how='any', subset=['正文','标题'], inplace=True)

# Prepare data for LDA
if opt.lda:
    print('Prepare data for LDA...')

    # Extract year, month
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['年'] = df['发布时间'].dt.year
    df['月'] = df['发布时间'].dt.month
    for i in range(np.min(df['年']), np.max(df['年'])+1):
        for j in range(np.min(df['月']), np.max(df['月'])+1):
            df_body = df[['标题','正文']][(df['年']==i) & (df['月']==j)]
            if not (df_body.empty):
                saveToTxt(df_body, './lda/' + str(i)+'_'+str(j)+'.txt')

# Split data into training, validation, test data sets, psedudo randomized
train, validate, test = np.split(df.sample(frac=1, random_state=123), 
                        [int(opt.train_percent*len(df)), \
                         int((opt.train_percent + opt.valid_percent)*len(df))])


title_train, body_train, y_train, v_train = getData(train)
title_validate, body_validate, y_validate, v_validate = getData(validate)
title_test, body_test, y_test, v_test = getData(test)

# Standardize y ('点赞数')
# train_mean = np.mean(y_train, 0)
# train_std = np.std(y_train, 0)
# y_train = (y_train - train_mean) / train_std
# y_validate = (y_validate - train_mean) / train_std
# y_test = (y_test - train_mean) / train_std

# Save data sets to .txt files
saveToTxt(title_train, opt.data + 'title_train.txt')
saveToTxt(body_train, opt.data + 'body_train.txt')
saveToTxt(y_train, opt.data + 'y_train.txt')
saveToTxt(v_train, opt.data + 'v_train.txt')
saveToTxt(title_validate, opt.data + 'title_validate.txt')
saveToTxt(body_validate, opt.data + 'body_validate.txt')
saveToTxt(y_validate, opt.data + 'y_validate.txt')
saveToTxt(v_validate, opt.data + 'v_validate.txt')
saveToTxt(title_test, opt.data + 'title_test.txt')
saveToTxt(body_test, opt.data + 'body_test.txt')
saveToTxt(y_test, opt.data + 'y_test.txt')
saveToTxt(v_test, opt.data + 'v_test.txt')

print('Data preparation done.')

