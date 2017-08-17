from sklearn.feature_extraction.text import CountVectorizer
import utils
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import svm
from sklearn import linear_model


def filter_text(X, T, Y):
    f_X, f_t, f_y = [], [], []
    for x, t, y in zip(X, T, Y):
        if len(x.split()) > 10:
            f_X.append(x)
            f_t.append(t)
            f_y.append(y)
    return f_X, f_t, f_y


def regression(model, tf_train, tf_val, tf_test, y_train, y_val, y_test):
    model.fit(tf_train, y_train)
    val_res = model.predict(tf_val)
    val_mse = np.mean(np.power(val_res - y_val, 2))
    print(val_mse)
    res = model.predict(tf_test)
    mse = np.mean(np.power(res - y_test, 2))
    print(mse)


def random_predict(y_train, y_val, y_test):
    np.random.seed(1234)
    val_res = np.random.random_sample(y_val.shape)
    val_mse = np.mean(np.power(val_res - y_val, 2))
    print(val_mse)
    np.random.seed(123)
    res = np.random.random_sample(y_val.shape)
    mse = np.mean(np.power(res - y_test, 2))
    print(mse)
    


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
    print(max(y_train), min(y_train))

    y_train, y_val, y_test = np.log(np.array(y_train) + 1), \
                             np.log(np.array(y_val) + 1), np.log(np.array(y_test) + 1)
    print(max(y_train), min(y_train))

    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    # max_features=n_features,
                                    min_df=2)
    tf_train = tf_vectorizer.fit_transform(X_train)
    tf_val = tf_vectorizer.transform(X_val)
    tf_test = tf_vectorizer.transform(X_test)

    
    # model = RandomForestRegressor(n_estimators = 100)
    # model = linear_model.Lasso(alpha = 0.01)
    model = svm.SVR()
    regression(model, tf_train, tf_val, tf_test, y_train, y_val, y_test)
    random_predict(y_train, y_val, y_test)
    
if __name__ == "__main__":
    main()
