import os
import codecs
import jieba
from nltk.corpus import stopwords


def preprocessSrc(path, stopwords, remove_sentences):
    res = []
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
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
    return res

def tokenize(src):
    '''
    src (str)
    '''
    return list(jieba.cut(src, cut_all=False))


def main():
    with codecs.open('remove_sentences', 'r', 'utf-8') as p:
        remove_sentences = [word.strip() for word in p.readlines()]

    stopwds = stopwords.words("english")
    with codecs.open('chinese_stop_words', 'r', 'utf-8') as p:
        stopwds += [word.strip() for word in p.readlines()]
    with codecs.open('remove_words', 'r', 'utf-8') as p:
        stopwds += [word.strip() for word in p.readlines()]

    path = './lda/'
    filenames = [x for x in os.listdir(path) if not x.startswith('token')]
    for fname in filenames:
        tokens = preprocessSrc(os.path.join(path, fname), stopwds, remove_sentences)
        with codecs.open(os.path.join(path, 'token_' + fname),
                         'w', 'utf-8') as p:
            p.write('\n'.join(tokens))

if __name__ == "__main__":
    main()
