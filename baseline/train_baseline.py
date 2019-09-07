import json
from model_baseline import model
from keras.callbacks import Callback
import keras.backend as K
from tqdm import tqdm
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_data = json.load(open('data/train.json'))
dev_data = json.load(open('data/dev.json'))[:500]
id2predicate, predicate2id = json.load(open('data/schemas.json'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('data/vocab.json'))
num_classes = len(id2predicate)

# ss = []
# for dd in train_data:
#     s = dd['spo_list']
#     for d in s:
#         ss.append(d)
# sss = [len(s[0]) for s in ss]
# aaa = np.percentile(np.array(sss), 1)
# print(aaa)

# s = []
# for d in train_data:
#     s.append(d['text'])
# l = [len(i) for i in s]
# print(np.percentile(np.array(l), 98))

max_s = 14
max_len = 140

train_model, subject_model, object_model = model(len(char2id), max_len, len(predicate2id))


# train_model.load_weights('best_model.weights')
# subject_model.load_weights('best_model.weights')
# object_model.load_weights('object_model.weights')

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class DataGenerator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            char_index, s_indexs, s_stars, s_ends, po_stars, po_ends = [], [], [], [], [], []
            for d in self.data:
                text = d['text'][:max_len]
                s_star_v, s_end_v = np.zeros(len(text)), np.zeros(len(text))
                po_star_v, po_end_v = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                for sop in d['spo_list']:
                    s_index = [char2id.get(c, 1) for c in sop[0]]
                    s_index = s_index[:max_s]
                    s_star = text.find(sop[0])
                    po_star = text.find(sop[2])

                    if s_star != -1 and po_star != -1:
                        s_end = s_star + len(sop[0]) - 1
                        po_end = po_star + len(sop[2]) - 1
                        p_index = predicate2id[sop[1]]

                        s_star_v[s_star] = 1
                        s_end_v[s_end] = 1

                        po_star_v[po_star][p_index] = 1
                        po_end_v[po_end][p_index] = 1

                s_indexs.append(s_index)
                char_index.append([char2id.get(c, 1) for c in text])
                s_stars.append(s_star_v)
                s_ends.append(s_end_v)
                po_stars.append(po_star_v)
                po_ends.append(po_end_v)

                if len(char_index) == self.batch_size or d == self.data[-1]:
                    s_indexs = seq_padding(s_indexs)
                    char_index = seq_padding(char_index)
                    s_stars = seq_padding(s_stars)
                    s_ends = seq_padding(s_ends)
                    po_stars = seq_padding(po_stars, np.zeros(num_classes))
                    po_ends = seq_padding(po_ends, np.zeros(num_classes))

                    yield [char_index, s_indexs, s_stars, s_ends, po_stars, po_ends], None
                    char_index, s_indexs, s_stars, s_ends, po_stars, po_ends = [], [], [], [], [], []


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


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.F1 = []
        self.best = 0
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            lr = (self.passed + 1) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 >= self.best:
            self.best = f1
            train_model.save_weights('out/train_model.weights')
            subject_model.save_weights('out/subject_model.weights')
            object_model.save_weights('out/object_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        if epoch + 1 == 50 or (
                self.stage == 0 and epoch > 10 and
                (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('out/train_model.weights')
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10

        for d in tqdm(dev_data):
            text = extract_items(d['text'])
            R = set(text)
            spo = []
            for z in d['spo_list']:
                spo.append(tuple(z))
            T = set(spo)
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C


def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10

    for d in dev_data:
        text = extract_items(d['text'])
        R = set(text)
        spo = []
        for z in d['spo_list']:
            spo.append(tuple(z))
        print('pred:', text, ' ori:', str(spo))
        T = set(spo)
        A += len(R & T)
        B += len(R)
        C += len(T)
    return 2 * A / (B + C), A / B, A / C


if __name__ == '__main__':
    # evaluate()
    train_model.load_weights('out/train_model.weights')
    train_ge = DataGenerator(train_data, batch_size=512)
    callback = Evaluate()
    train_model.fit_generator(train_ge.__iter__(),
                              steps_per_epoch=len(train_ge),
                              epochs=200,
                              callbacks=[callback])
