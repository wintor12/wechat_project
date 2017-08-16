from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import codecs
import os


k = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    message = ''
    for topic_idx, topic in enumerate(model.components_):
        message += "Topic #%d: " % (topic_idx + 1)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        message += '\n'
    return message


def runLDA(samples):
    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    # max_features=n_features,
                                    min_df=2)
    tf = tf_vectorizer.fit_transform(samples)
    lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return print_top_words(lda, tf_feature_names, n_top_words)
    

def main():
    res = ''
    path = './lda'
    fnames = [x for x in os.listdir(path) if x.startswith('token')]
    for fname in fnames:
        print(fname[6:])
        res += fname[6:] + '\n'
        with codecs.open(os.path.join(path, fname), 'r', 'utf-8') as f:
            samples = f.readlines()
            print('total articles: ' + str(len(samples)))
            message = runLDA(samples)
            res += message + '\n'
    with codecs.open('topics.txt', 'w', 'utf-8') as p:
        p.write(res)


if __name__ == "__main__":
    main()
