from sklearn.feature_extraction.text import CountVectorizer
import utils
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='if true, y = log(y + 1)')
parser.add_argument('--label', default='upvote',
                    help='upvote | view')
parser.add_argument('--crit', default='mse', type=str, help='mse | mae')
parser.add_argument('--model', default='lasso', type=str, help='lasso | rf | svm | random | random_small')
parser.add_argument('--classification', action='store_true',
                    help='if true, use classfication model')



opt = parser.parse_args()
print(opt)


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
    elif crit == 'acc':
        loss = accuracy_score(y, pred)
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


def classify(model, tf_train, tf_val, tf_test, y_train, y_val, y_test):
    if model == 'random':
        np.random.seed(1234)
        val_pred = np.random.randint(np.min(y_val), np.max(y_val) + 1, y_val.shape)
        np.random.seed(1234)
        test_pred = np.random.randint(np.min(y_test), np.max(y_test) + 1, y_test.shape)
    elif model == 'random_small':
        val_pred = np.zeros(y_val.shape)
        test_pred = np.zeros(y_test.shape)
    else:
        model.fit(tf_train, y_train)
        val_pred = model.predict(tf_val)
        test_pred = model.predict(tf_test)
    return val_pred, test_pred



def main():
    X_train, X_val, X_test, t_train, t_val, t_test, \
        y_train, y_val, y_test, v_train, v_val, v_test = utils.loadData()

    # Use view as the label
    if opt.label == 'view':
        y_train, y_val, y_test = v_train, v_val, v_test

    print('Original dataset: train | val | test')
    print(len(X_train), len(X_val), len(X_test))
    assert len(X_train) == len(t_train) == len(y_train), \
        'train size of texts, titles, labels not match'
    assert len(X_val) == len(t_val) == len(y_val), \
        'val size of texts, titles, labels not match'
    assert len(X_test) == len(t_test) == len(y_test), \
        'test size of texts, titles, labels not match'

    X_train, t_train, y_train = filter_text(X_train, t_train, y_train)
    X_val, t_val, y_val = filter_text(X_val, t_val, y_val)
    X_test, t_test, y_test = filter_text(X_test, t_test, y_test)
    print('filtered dataset: train | val | test')
    print(len(X_train), len(X_val), len(X_test))
    assert len(X_train) == len(t_train) == len(y_train), \
        'train size of texts, titles, labels not match'
    assert len(X_val) == len(t_val) == len(y_val), \
        'val size of texts, titles, labels not match'
    assert len(X_test) == len(t_test) == len(y_test), \
        'test size of texts, titles, labels not match'

    # print(sorted(X_train, key=lambda x: len(x))[:3])

    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    y = np.concatenate((y_train, y_val, y_test), axis = 0)
    print('total data: ', y.shape)
    
    if opt.classification:
        if opt.label == 'view':
            y[y < 8000] = 0
            y[np.logical_and(y >= 8000, y <= 13000)] = 1
            y[y > 13000] = 2
        else:
            y[y <= 10] = 0
            y[np.logical_and(y > 10, y <= 20)] = 1
            y[y > 20] = 2

        print('data distribution in whole dataset y=0 | y=1 | y=2')
        print(y[y == 0].shape, y[y == 1].shape, y[y == 2].shape)
        y_train, y_val, y_test = y[:y_train.shape[0]], \
                                 y[y_train.shape[0]: y_train.shape[0] + y_val.shape[0]], \
                                 y[y_train.shape[0] + y_val.shape[0]:]

        print('data distribution in training set y=0 | y=1 | y=2')
        print(y_train[y_train == 0].shape, y_train[y_train == 1].shape,
              y_train[y_train == 2].shape)
        print('data distribution in test set y=0 | y=1 | y=2')
        print(y_test[y_test == 0].shape, y_test[y_test == 1].shape,
              y_test[y_test == 2].shape)
        opt.crit = 'acc'

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
        val_pred, test_pred = classify(model, tf_train, tf_val, tf_test,
                                       y_train, y_val, y_test)
    else:
        val_pred, test_pred = regression(model, tf_train, tf_val, tf_test,
                                         y_train, y_val, y_test)

    crit = opt.crit
    print('criterion', crit)
    val_loss = criterion(crit, val_pred, y_val)
    print('val_loss', val_loss)
    test_loss = criterion(crit, test_pred, y_test)
    print('test_loss', test_loss)
    print(test_pred.shape, y_test.shape)
    print('confusion matrix')
    print(confusion_matrix(y_test, test_pred))

    
if __name__ == "__main__":
    main()
