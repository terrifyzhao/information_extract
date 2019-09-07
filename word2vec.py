from gensim.models import Word2Vec
from tqdm import tqdm
import jieba
import json

texts = []

train_data = json.load(open('data/train.json'))
dev_data = json.load(open('data/dev.json'))
for d in tqdm(train_data + dev_data):
    texts.append(list(jieba.cut(d['text'])))

model = Word2Vec(texts, size=300, window=5, min_count=0, workers=4)

model.save("word2vec.model")
