import json
from final.model import model
import os
import numpy as np
import tensorflow as tf
import jieba
from tqdm import tqdm
import ahocorasick
from gensim.models import Word2Vec

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path = os.getcwd()
graph = tf.get_default_graph()

train_data = json.load(open(path + '/data/train.json'))
id2predicate, predicate2id = json.load(open(path + '/data/schemas.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open(path + '/data/vocab.json'))
num_classes = len(id2predicate)

# 词向量
word2vec = Word2Vec.load(path + '/word2vec.model')
id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0
word2vec = np.concatenate([np.zeros((1, word2vec.shape[1])), word2vec])

max_s = 14
max_len = 140

train_model, subject_model, object_model = model(len(char2id), max_len, len(predicate2id))
subject_model.load_weights(path + '/out2/subject_model.weights')
object_model.load_weights(path + '/out2/object_model.weights')


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class SopSearch:
    def __init__(self):
        self.ac_s = ahocorasick.Automaton()
        self.ac_o = ahocorasick.Automaton()

        self.sop_dic = {}
        self.sop_total = {}
        for i, d in enumerate(tqdm(train_data, desc='build SOP search')):
            for s, p, o in d['spo_list']:
                self.ac_s.add_word(s, s)
                self.ac_o.add_word(o, o)
                if (s, o) not in self.sop_dic:
                    self.sop_dic[(s, o)] = set()
                if (s, p, o) not in self.sop_total:
                    self.sop_total[(s, p, o)] = set()
                self.sop_dic[(s, o)].add(p)
                self.sop_total[(s, p, o)].add(i)

        self.ac_s.make_automaton()
        self.ac_o.make_automaton()

    def find(self, text, i=None):
        spo = set()
        for s in self.ac_s.iter(text):
            for o in self.ac_o.iter(text):
                if (s[1], o[1]) in self.sop_dic.keys():
                    for p in self.sop_dic.get((s[1], o[1])):
                        if i is None:
                            spo.add((s[1], p, o[1]))
                        elif self.sop_total[(s[1], p, o[1])] - {i}:
                            spo.add((s[1], p, o[1]))
        return list(spo)


spo_search = SopSearch()


def sent2vec(S):
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


def extract_items(text):
    R = []
    text = text[:max_len]
    char_index = [char2id.get(c, 1) for c in text]
    pre_po = np.zeros((len(text), num_classes, 2))
    pre_s = np.zeros((len(text), 2))
    for s, p, o in spo_search.find(text):
        pre_s[text.find(s), 0] = 1
        pre_s[text.find(s) + len(s) - 1, 1] = 1
        pre_po[text.find(o), predicate2id[p], 0] = 1
        pre_po[text.find(o) + len(o) - 1, predicate2id[p], 1] = 1

    word = sent2vec([list(jieba.cut(text))])
    pre_s = np.expand_dims(pre_s, 0)
    char_index = np.array([char_index])
    s_star, s_end = subject_model.predict([char_index, word, pre_s])

    s_star, s_end = s_star[0, :, 0], s_end[0, :, 0]
    # index
    s_star_out, s_end_out = np.where(s_star > 0.5)[0], np.where(s_end > 0.5)[0]
    # one-hot
    s_star_in, s_end_in = np.where(s_star > 0.5, 1, 0), np.where(s_end > 0.5, 1, 0)
    s_star, s_end = s_star_out, s_end_out
    subjects = []
    for i in s_star:
        j = s_end[s_end >= i]
        if len(j) > 0:
            j = j[0]
            subject = text[i: j + 1]
            subjects.append((subject, i, j))

    # subjects.append(('阿斯达', 1, 4))
    # subjects.append(('得到的', 2, 5))
    if subjects:

        s_index = []
        for subject in subjects:
            s_index.append([char2id.get(c, 1) for c in subject[0]])
        # s_index = [char2id.get(c, 1) for c in subjects[0][0]]
        # s_index = np.array([s_index])
        s_index = seq_padding(s_index)

        s_star_in = np.array([s_star_in])
        s_end_in = np.array([s_end_in])
        pre_po = pre_po.reshape(pre_po.shape[0], -1)
        pre_po = np.expand_dims(pre_po, 0)

        char_index = np.repeat(char_index, len(subjects), 0)
        word = np.repeat(word, len(subjects), 0)
        s_star_in = np.repeat(s_star_in, len(subjects), 0)
        s_end_in = np.repeat(s_end_in, len(subjects), 0)
        pre_s = np.repeat(pre_s, len(subjects), 0)
        pre_po = np.repeat(pre_po, len(subjects), 0)

        o1, o2 = object_model.predict([char_index, word, s_index, s_star_in, s_end_in, pre_s, pre_po])

        for i, subject in enumerate(subjects):
            _oo1, _oo2 = np.where(o1[i] > 0.5), np.where(o2[i] > 0.5)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text[_ooo1: _ooo2 + 1]
                        _predicate = id2predicate[_c1]
                        R.append((subject[0], _predicate, _object))
                        break
        zhuanji, gequ = [], []
        for s, p, o in R[:]:
            if p == u'妻子':
                R.append((o, u'丈夫', s))
            elif p == u'丈夫':
                R.append((o, u'妻子', s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s, p, o in R:
            if p in [u'歌手', u'作词', u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []


def predict(text):
    global graph
    with graph.as_default():
        result = extract_items(text)
        return result


if __name__ == '__main__':
    while 1:
        text = input('text:')
        r = predict(text)
        print(r)
