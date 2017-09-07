from sklearn.feature_extraction.text import CountVectorizer
import utils
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.sparse
import copy
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='if true, y = log(y + 1)')
parser.add_argument('--label', default='upvote',
                    help='upvote | view | position')
parser.add_argument('--crit', default='mse', type=str, help='mse | mae | acc')
parser.add_argument('--model', default='lasso', type=str, help='lasso | rf | svm | random | random_small')
parser.add_argument('--classification', action='store_true',
                    help='if true, use classfication model')



opt = parser.parse_args()
print(opt)


def filter_text(X, T, Y, P, TIME):
    f_X, f_t, f_y, f_p, f_time = [], [], [], [], []
    for x, t, y, p, time in zip(X, T, Y, P, TIME):
        if len(x.split()) > 10:
            f_X.append(x)
            f_t.append(t)
            f_y.append(y)
            f_p.append(p)
            f_time.append(time)
    return f_X, f_t, f_y, f_p, f_time


def criterion(crit, pred, y):
    loss = 0
    if crit == 'mse':
        loss = np.mean(np.power(pred - y, 2))
    elif crit == 'mae':
        loss = np.mean(np.absolute(pred - y))
    elif crit == 'acc':
        loss = accuracy_score(y, pred)
    return loss
    

def get_posts_by_day(tf_test, time_test):
    test = copy.deepcopy(tf_test)
    test = np.array(test.todense())    
    posts_by_day = []
    start_index, end_index = 0, 0
    length = len(test)
    print('Start to get posts by day in test data...')
    #print('test shape: ', test.shape)
    #print('time_test length: ', len(time_test))
    
    for i in range(length):
        #if(i<10):
        #    print(time_test[i])
        #    print(time_test[i+1])
        if i+1 < length and \
           time_test[i+1] != time_test[i]: # compare time string
            end_index = i
            #print(start_index)
            #print(end_index)
            posts_by_day.append(test[start_index:end_index+1, 0:len(test[0])])
            start_index = i+1            
    
    posts_by_day.append(test[start_index:, 0:len(test[0])])
    print('Total days in test data: ', len(posts_by_day))
    #print("each day's posts' shape: ")
    #for p in posts_by_day:
    #    print(p.shape)
    return posts_by_day
   

def predict_position(model, posts_by_day):
    prediction = []
    for daily_posts in posts_by_day:
        # number of posts on some day
        num = len(daily_posts)
        max_indecis = [-sys.maxsize-1] * num
        for i in range(num):
            # set position all to be i
            daily_posts[:,len(daily_posts[0])-2] = i
            max_view = -sys.maxsize-1
            for j in range(num):
                
                if max_indecis[j] != -sys.maxsize-1:
                    continue
                predict_view = model.predict(daily_posts[j].reshape(1,-1))
                if predict_view > max_view:
                    max_view = predict_view
                    max_index = j
            max_indecis[max_index] = i
        prediction.append(max_indecis)
    prediction = [val for sublist in prediction for val in sublist]
    return np.array(prediction)
            
def regression(model, tf_train, tf_test, y_train, y_test, time_test):
    if model == 'random':
        #np.random.seed(1234)
        #val_pred = (np.max(y_train) - np.min(y_train)) * \
        #           np.random.random_sample(y_val.shape)
        np.random.seed(1234)
        test_pred = (np.max(y_train) - np.min(y_train)) * \
                    np.random.random_sample(y_test.shape)
    elif model == 'random_small':
        #np.random.seed(1234)
        #val_pred = np.random.random_sample(y_val.shape)
        np.random.seed(1234)
        test_pred = np.random.random_sample(y_test.shape)
        if not opt.log:
            #val_pred = 10 * val_pred
            test_pred = 10 * test_pred
    else:
        model.fit(tf_train, y_train)
        posts_by_day = get_posts_by_day(tf_test, time_test)
        test_pred = predict_position(model, posts_by_day)
    return test_pred


def classify(model, tf_train, tf_test, y_train, y_test, time_test):
    if model == 'random':
        np.random.seed(1234)
        test_pred = np.random.randint(np.min(y_test), np.max(y_test) + 1, y_test.shape)
    elif model == 'random_small':
        test_pred = np.zeros(y_test.shape)
    else:
        model.fit(tf_train, y_train)
        #test_pred = model.predict(tf_test)
        posts_by_day = get_posts_by_day(tf_test, time_test)
        test_pred = predict_position(model, posts_by_day)
    return test_pred



def main():
    X_train, X_test, t_train, t_test, \
        y_train, y_test, p_train, p_test, time_train, time_test = utils.loadTrainingTestData()

    print('Original dataset: train | test')
    print(len(X_train), len(X_test))
    assert len(X_train) == len(t_train) == len(y_train) == len(p_train) == len(time_train), \
        'train size of texts, titles, labels, positions, time not match'
    assert len(X_test) == len(t_test) == len(y_test) == len(p_test) == len(time_test), \
        'test size of texts, titles, labels, positions, time not match'

    X_train, t_train, y_train, p_train, time_train = filter_text(X_train, t_train, y_train, p_train, time_train)
    X_test, t_test, y_test, p_test, time_test = filter_text(X_test, t_test, y_test, p_test, time_test)
    print('filtered dataset: train | test')
    print(len(X_train), len(X_test))
    assert len(X_train) == len(t_train) == len(y_train) == len(p_train) == len(time_train), \
        'train size of texts, titles, labels, positions, time not match'
    assert len(X_test) == len(t_test) == len(y_test) == len(p_test) == len(time_test), \
        'test size of texts, titles, labels, positions, time not match'

    # store filtered y_test
    thefile = open('y_test.txt', 'w')
    for item in y_test:
        thefile.write("%s\n" % item)
    
    # print(sorted(X_train, key=lambda x: len(x))[:3])

    y_train, y_test = np.array(y_train), np.array(y_test)
    y = np.concatenate((y_train, y_test), axis = 0)
    print('total data: ', y.shape)
    
    if opt.classification:
        y[y < 8000] = 0
        y[np.logical_and(y >= 8000, y <= 13000)] = 1
        y[y > 13000] = 2

        print('data distribution in whole dataset y=0 | y=1 | y=2')
        print(y[y == 0].shape, y[y == 1].shape, y[y == 2].shape)
        y_train, y_test = y[:y_train.shape[0]], \
                          y[y_train.shape[0]:]

        print('data distribution in training set y=0 | y=1 | y=2')
        print(y_train[y_train == 0].shape, y_train[y_train == 1].shape,
              y_train[y_train == 2].shape)
        print('data distribution in test set y=0 | y=1 | y=2')
        print(y_test[y_test == 0].shape, y_test[y_test == 1].shape,
              y_test[y_test == 2].shape)
        opt.crit = 'acc'

    if opt.log:
        y_train, y_test = np.log(y_train + 1), np.log(y_test + 1)
    print(max(y_train), min(y_train))

    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    # max_features=n_features,
                                    min_df=2)
    tf_train = tf_vectorizer.fit_transform(X_train)
    # add position feature to training set
    p_train = np.array(p_train).reshape(-1,1)
    tf_train = scipy.sparse.csr_matrix(np.append(tf_train.todense(), p_train, 1))
    print('training data: ', tf_train.shape)
    
    tf_test = tf_vectorizer.transform(X_test)
    # add position feature to test set
    p_test = np.array(p_test).reshape(-1,1)
    tf_test = scipy.sparse.csr_matrix(np.append(tf_test.todense(), p_test, 1))
    print('test data: ', tf_test.shape)

    if opt.model == 'rf':
        model = RandomForestRegressor(n_estimators = 100) \
                if not opt.classification else RandomForestClassifier(n_estimators = 100)
    elif opt.model == 'lasso':
        model = linear_model.Lasso(alpha = 0.01) if not opt.classification else \
                linear_model.LogisticRegression(penalty='l1')
    elif opt.model == 'svm':
        model = svm.SVR() if not opt.classification else \
                svm.SVC()
    elif opt.model == 'random':
        model = 'random'
    elif opt.model == 'random_small':
        model = 'random_small'
    else:
        raise Exception('Model can only be rf | lasso | svm | random')
    print(model)

    if opt.classification:
        test_pred = classify(model, tf_train, tf_test,
                                       y_train, y_test, time_test)
    else:
        test_pred = regression(model, tf_train, tf_test,
                                         y_train, y_test, time_test)

    # save view predictions to txt file
    thefile = open('test_predictions.txt', 'w')
    for item in test_pred:
        thefile.write("%s\n" % item)
        
    crit = opt.crit
    print('criterion', crit)
    test_loss = criterion(crit, test_pred, p_test)
    print('test_loss', test_loss)
    print(test_pred.shape, p_test.shape)
    print('confusion matrix')
    print(confusion_matrix(p_test, test_pred))

    
if __name__ == "__main__":
    main()
