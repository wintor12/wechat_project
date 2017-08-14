"""
Prepare data
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

file_path = '/home/indoornav/Hefei/INSIGHT_CHINA/北美留学生.xlsx'
train_set = 0.8
valid_set = 0.1
test_set = 0.1

# Save dataframe to .txt file
def saveToTxt(dataframe, file_name):
    dataframe.to_csv(file_name, header=None, index=None, sep=' ')


df = pd.read_excel(file_path, sheetname='list')

# Get training, validation, test data sets
# This will split randomizly
#train, validate, test = np.split(df.sample(frac=1), [int(train_set*len(df)), int((train_set + valid_set)*len(df))])
train, validate, test = np.split(df, [int(train_set*len(df)), int((train_set + valid_set)*len(df))])

# Prepare training set
title_train = train['标题']
body_train = train['正文']
y_train = train['点赞数']

# Prepare validation set
title_validate = validate['标题']
body_validate = validate['正文']
y_validate = validate['点赞数']

# Prepare test set
title_test = test['标题']
body_test = test['正文']
y_test = test['点赞数']

# Save data sets to .txt files
saveToTxt(title_train, 'title_train.txt')
saveToTxt(body_train, 'body_train.txt')
saveToTxt(y_train, 'y_train')
saveToTxt(title_validate, 'title_validate.txt')
saveToTxt(body_validate, 'body_validate.txt')
saveToTxt(y_validate, 'y_validate.txt')
saveToTxt(title_test, 'title_test.txt')
saveToTxt(body_test, 'body_test')
saveToTxt(y_test, 'y_test')

