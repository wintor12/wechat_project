from sklearn.feature_extraction.text import CountVectorizer
import utils
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import svm
from sklearn import linear_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='if true, y = log(y + 1)')
parser.add_argument('--crit', default='mse', type=str, help='mse | mae')
parser.add_argument('--model', default='lasso', type=str, help='lasso | rf | svm | random | random_small')

opt = parser.parse_args()


def filter_text(X, T, Y):
    f_X, f_t, f_y = [], [], []
    for x, t, y in zip(X, T, Y):
        if len(x.split()) > 10:
            f_X.append(x)
            f_t.append(t)
            f_y.append(y)
    return f_X, f_t, f_y


def criterion(crit, pred, y):
    loss = 0
    if crit == 'mse':
        loss = np.mean(np.power(pred - y, 2))
    elif crit == 'mae':
        loss = np.mean(np.absolute(pred - y))
    return loss

def regression(model, tf_train, tf_val, tf_test, y_train, y_val, y_test):
    if model == 'random':
        np.random.seed(1234)
        val_pred = (np.max(y_train) - np.min(y_train)) * \
                   np.random.random_sample(y_val.shape)
        np.random.seed(1234)
        test_pred = (np.max(y_train) - np.min(y_train)) * \
                    np.random.random_sample(y_test.shape)
    elif model == 'random_small':
        np.random.seed(1234)
        val_pred = np.random.random_sample(y_val.shape)
        np.random.seed(1234)
        test_pred = np.random.random_sample(y_test.shape)
        if not opt.log:
            val_pred = 10 * val_pred
            test_pred = 10 * test_pred
    else:
        model.fit(tf_train, y_train)
        val_pred = model.predict(tf_val)
        test_pred = model.predict(tf_test)
    return val_pred, test_pred
    

def main():
    X_train, X_val, X_test, t_train, t_val, \
        t_test, y_train, y_val, y_test = utils.loadData()
    print(len(X_train), len(X_val), len(X_test))
    print(len(t_train), len(t_val), len(t_test))
    print(len(y_train), len(y_val), len(y_test))
    X_train, t_train, y_train = filter_text(X_train, t_train, y_train)
    X_val, t_val, y_val = filter_text(X_val, t_val, y_val)
    X_test, t_test, y_test = filter_text(X_test, t_test, y_test)
    print(len(X_train), len(X_val), len(X_test))
    print(len(t_train), len(t_val), len(t_test))
    print(len(y_train), len(y_val), len(y_test))
    # print(sorted(X_train, key=lambda x: len(x))[:3])

    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    if opt.log:
        y_train, y_val, y_test = np.log(y_train + 1), \
                                 np.log(y_val + 1), np.log(y_test + 1)
    print(max(y_train), min(y_train))

    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    # max_features=n_features,
                                    min_df=2)
    tf_train = tf_vectorizer.fit_transform(X_train)
    tf_val = tf_vectorizer.transform(X_val)
    tf_test = tf_vectorizer.transform(X_test)

    if opt.model == 'rf':
        model = RandomForestRegressor(n_estimators = 100)
    elif opt.model == 'lasso':
        model = linear_model.Lasso(alpha = 0.01)
    elif opt.model == 'svm':
        model = svm.SVR()
    elif opt.model == 'random':
        model = 'random'
    elif opt.model == 'random_small':
        model = 'random_small'
    else:
        raise Exception('Model can only be rf | lasso | svm | random')
    print(model)
    
    val_pred, test_pred = regression(model, tf_train, tf_val, tf_test,
                                     y_train, y_val, y_test)
    crit = opt.crit
    val_loss = criterion(crit, val_pred, y_val)
    print(val_loss)
    test_loss = criterion(crit, test_pred, y_test)
    print(test_loss)

    
if __name__ == "__main__":
    main()
