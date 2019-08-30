import json
from model import model
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

id2predicate, predicate2id = json.load(open('data/schemas.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('data/vocab.json'))
num_classes = len(id2predicate)

max_s = 14
max_len = 140

train_model, subject_model, object_model = model(len(char2id), max_len, len(predicate2id))
subject_model.load_weights('out/subject_model.weights')
object_model.load_weights('out/object_model.weights')


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def extract_items(text):
    R = []
    text = text[:max_len]
    char_index = [char2id.get(c, 1) for c in text]
    s_star, s_end = subject_model.predict(np.array([char_index]))
    s_star, s_end = s_star[0, :, 0], s_end[0, :, 0]
    # index
    s_star_out, s_end_out = np.where(s_star > 0.5)[0], np.where(s_end > 0.4)[0]
    # one-hot
    s_star_in, s_end_in = np.where(s_star > 0.5, 1, 0), np.where(s_end > 0.4, 1, 0)
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
        s_index = seq_padding(s_index)

        char_index = np.array([char_index])
        char_index = np.repeat(char_index, len(subjects), 0)

        s_star_in = np.array([s_star_in])
        s_star_in = np.repeat(s_star_in, len(subjects), 0)
        s_end_in = np.array([s_end_in])
        s_end_in = np.repeat(s_end_in, len(subjects), 0)

        o1, o2 = object_model.predict([char_index, s_index, s_star_in, s_end_in])

        for i, subject in enumerate(subjects):
            _oo1, _oo2 = np.where(o1[i] > 0.5), np.where(o2[i] > 0.4)
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
    result = extract_items(text)
    return result


if __name__ == '__main__':
    while 1:
        text = input('text:')
        r = predict(text)
        print(r)
