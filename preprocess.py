import os
import codecs
import jieba

remove = ['后台回复「福利」，',
          '出国前免费领取一张美国电话卡 ',
          '▲请大家扫描二维码进入话题或点击「阅读原文」进入首页移步至社区讨论',
          '说明：现在【有事】需要帮助的同学可以进入北美留学生社区进行投稿。投稿时请在文首标注【有事】，在这里你将遇到更多热心肠的小伙伴，更有社区君背后的智慧团为你提供更加专业的回答。',
          '▲请大家扫描二维码或点击「阅读原文」移步至社区讨论',
          '▲扫描二维码或点击「阅读原文」进入首页一起参与讨论吧']

def preprocessSrc(path):
    res = []
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            for r in remove:
                if r in line:
                    line = line.replace(r, '')
            if line:
                res.append(tokenize(line))
    return res

def tokenize(src):
    '''
    src (str)
    '''
    return ' '.join(list(jieba.cut(src, cut_all=False)))


def main():
    path = './lda/'
    filenames = [x for x in os.listdir(path) if not x.startswith('token')]
    for fname in filenames:
        tokens = preprocessSrc(os.path.join(path, fname))
        with codecs.open(os.path.join(path, 'token_' + fname),
                         'w', 'utf-8') as p:
            p.write('\n'.join(tokens))

if __name__ == "__main__":
    main()
