import os
import codecs


def loadText(path):
    with codecs.open(path, 'r', 'utf-8') as p:
        texts = [x.strip() for x in p.readlines()]
    return texts


def loadUpvotes(path):
    with codecs.open(path, 'r', 'utf-8') as p:
        upvotes = [float(x.strip()) for x in p.readlines()]
    return upvotes


def loadData():
    path = os.path.dirname('train/')
    X_train, X_val, X_test = (loadText(os.path.join(path, 'body_train.txt')),
                              loadText(os.path.join(path, 'body_validate.txt')),
                              loadText(os.path.join(path, 'body_test.txt')))
    title_train, title_val, title_test = (loadText(os.path.join(path, 'title_train.txt')),
                                          loadText(os.path.join(path, 'title_validate.txt')),
                                          loadText(os.path.join(path, 'title_test.txt')))

    y_train, y_val, y_test = (loadUpvotes(os.path.join(path, 'y_train.txt')),
                              loadUpvotes(os.path.join(path, 'y_validate.txt')),
                              loadUpvotes(os.path.join(path, 'y_test.txt')))
    v_train, v_val, v_test = (loadUpvotes(os.path.join(path, 'v_train.txt')),
                              loadUpvotes(os.path.join(path, 'v_validate.txt')),
                              loadUpvotes(os.path.join(path, 'v_test.txt')))
    return X_train, X_val, X_test, title_train, title_val, title_test, \
        y_train, y_val, y_test, v_train, v_val, v_test


def loadTrainingTestData():
    path = os.path.dirname('train/')
    X_train, X_test = (loadText(os.path.join(path, 'body_train.txt')),
                              loadText(os.path.join(path, 'body_test.txt')))
    title_train, title_test = (loadText(os.path.join(path, 'title_train.txt')),
                                          loadText(os.path.join(path, 'title_test.txt')))
    y_train, y_test = (loadUpvotes(os.path.join(path, 'y_train.txt')),
                              loadUpvotes(os.path.join(path, 'y_test.txt')))
    p_train, p_test = (loadUpvotes(os.path.join(path, 'position_train.txt')),
                              loadUpvotes(os.path.join(path, 'position_test.txt')))
    time_train, time_test = (loadText(os.path.join(path, 'time_train.txt')),
                              loadText(os.path.join(path, 'time_test.txt')))
    return X_train, X_test, title_train, title_test, y_train, y_test, p_train, p_test, time_train, time_test
