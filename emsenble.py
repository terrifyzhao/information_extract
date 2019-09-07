from baseline.predict_baseline import predict as predict_base
from final.predict import predict
from tqdm import tqdm
import json

dev_data = json.load(open('data/dev.json'))[:500]


def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10

    for d in tqdm(dev_data):
        R1 = predict(d['text'])
        R2 = predict_base(d['text'])
        R = set(R1) | set(R2)

        spo = []
        for z in d['spo_list']:
            spo.append(tuple(z))
        T = set(spo)
        A += len(R & T)
        B += len(R)
        C += len(T)
    return 2 * A / (B + C), A / B, A / C


def evaluate1():
    A, B, C = 1e-10, 1e-10, 1e-10

    for d in tqdm(dev_data):
        R = predict_base(d['text'])
        R = set(R)

        spo = []
        for z in d['spo_list']:
            spo.append(tuple(z))
        T = set(spo)
        A += len(R & T)
        B += len(R)
        C += len(T)
    return 2 * A / (B + C), A / B, A / C


def evaluate2():
    A, B, C = 1e-10, 1e-10, 1e-10

    for d in tqdm(dev_data):
        R = predict(d['text'])
        R = set(R)

        spo = []
        for z in d['spo_list']:
            spo.append(tuple(z))
        T = set(spo)
        A += len(R & T)
        B += len(R)
        C += len(T)
    return 2 * A / (B + C), A / B, A / C


if __name__ == '__main__':
    f1, precision, recall = evaluate()
    print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1, precision, recall))
    f1, precision, recall = evaluate1()
    print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1, precision, recall))
    f1, precision, recall = evaluate2()
    print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1, precision, recall))
