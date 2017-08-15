"""
Prepare data
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt

file_path = './北美留学生.xlsx'
train_set = 0.8 # Percentage of training set
valid_set = 0.1 # Percentage of validation set
test_set = 0.1 # Percentage of test set
lda = True # LDA model

# Save dataframe to .txt file
def saveToTxt(df, file_name):
    df.to_csv(file_name, header=None, index=None, sep=' ')


# Get xs and ys (columns '标题', '正文', '点赞数')
def getData(df):
    title, body, y = df['标题'], df['正文'], df['点赞数']
    return title, body, y


print('Start to prepare data...')
# Read excel file to dataframe
df = pd.read_excel(file_path, sheetname='list')
# Clean data
df.dropna(axis=0, how='any', subset=['正文','标题'], inplace=True)

# Prepare data for LDA
if lda:
    print('Prepare data for LDA...')

    # Extract year, month
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['年'] = df['发布时间'].dt.year
    df['月'] = df['发布时间'].dt.month
    for i in range(np.min(df['年']), np.max(df['年'])+1):
        for j in range(np.min(df['月']), np.max(df['月'])+1):
            df_body = df['正文'][(df['年']==i) & (df['月']==j)]
            if not (df_body.empty):
                saveToTxt(df_body, str(i)+'_'+str(j)+'.txt')

# Split data into training, validation, test data sets, psedudo randomized
train, validate, test = np.split(df.sample(frac=1, random_state=123), 
                                 [int(train_set*len(df)), int((train_set + valid_set)*len(df))])

title_train, body_train, y_train = getData(train)
title_validate, body_validate, y_validate = getData(validate)
title_test, body_test, y_test = getData(test)

# Standardize y ('点赞数')
train_mean = np.mean(y_train, 0)
train_std = np.std(y_train, 0)
y_train = (y_train - train_mean) / train_std
y_validate = (y_validate - train_mean) / train_std
y_test = (y_test - train_mean) / train_std

# Save data sets to .txt files
saveToTxt(title_train, 'title_train.txt')
saveToTxt(body_train, 'body_train.txt')
saveToTxt(y_train, 'y_train.txt')
saveToTxt(title_validate, 'title_validate.txt')
saveToTxt(body_validate, 'body_validate.txt')
saveToTxt(y_validate, 'y_validate.txt')
saveToTxt(title_test, 'title_test.txt')
saveToTxt(body_test, 'body_test.txt')
saveToTxt(y_test, 'y_test.txt')

print('Data preparation done.')

