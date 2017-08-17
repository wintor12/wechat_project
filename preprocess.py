import os
import codecs
import jieba
from nltk.corpus import stopwords
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str, help='train | lda')

opt = parser.parse_args()


def preprocessSrc(path, stopwords, remove_sentences):
    '''
    return list of text, each text is a string of tokens separated by ' ' 
    '''
    res = []
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            empty = False
            line = line.strip()
            for r in remove_sentences:
                if r in line:
                    line = line.replace(r, '')
            if line:
                tokens = tokenize(line)
                filtered_tokens = [word for word in tokens
                                   if not word.isdigit() and word not in stopwords]
                if filtered_tokens:
                    res.append(' '.join(filtered_tokens))
                else:  
                    empty = True
            else:  # line is empty
                empty = True
            if empty:
                res.append('')
    return res

def tokenize(src):
    '''
    src (str)
    '''
    return list(jieba.cut(src, cut_all=False))


def getStopwords():
    stopwds = stopwords.words("english")
    with codecs.open('chinese_stop_words', 'r', 'utf-8') as p:
        stopwds += [word.strip() for word in p.readlines()]
    with codecs.open('remove_words', 'r', 'utf-8') as p:
        stopwds += [word.strip() for word in p.readlines()]
    return stopwds


def preprocessLDAFiles(stopwds, remove_sentences):
    path = './lda/'
    filenames = [x for x in os.listdir(path) if not x.startswith('token')]
    for fname in filenames:
        tokens = preprocessSrc(os.path.join(path, fname), stopwds, remove_sentences)
        with codecs.open(os.path.join(path, 'token_' + fname),
                         'w', 'utf-8') as p:
            p.write('\n'.join(tokens))

def preprocessTrainFiles(stopwds, remove_sentences):
    path = os.path.dirname('./data/')
    dst = os.path.dirname('./train/')
    if not os.path.exists(dst):
            os.makedirs(dst)
    filenames = [x for x in os.listdir(path) if x.startswith('body')]
    titlenames = [x for x in os.listdir(path) if x.startswith('title')]
    for fname, tname in zip(filenames, titlenames):
        titles = preprocessSrc(os.path.join(path, tname), stopwds, remove_sentences)
        texts = preprocessSrc(os.path.join(path, fname), stopwds, remove_sentences)
        with codecs.open(os.path.join(dst, fname), 'w', 'utf-8') as p:
            p.write('\n'.join(texts))
        with codecs.open(os.path.join(dst, tname), 'w', 'utf-8') as p:
            p.write('\n'.join(titles))



def main():
    with codecs.open('remove_sentences', 'r', 'utf-8') as p:
        remove_sentences = [word.strip() for word in p.readlines()]

    stopwds = getStopwords()
    if opt.mode == 'lda':
        preprocessLDAFiles(stopwds, remove_sentences)
    if opt.mode == 'train':
        preprocessTrainFiles(stopwds, remove_sentences)

if __name__ == "__main__":
    main()
