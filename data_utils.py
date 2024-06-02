import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def read_data(split='train'):
    data = pd.read_csv(f"dataset/{split}.csv", header=None)

    labels, texts = np.array(data.iloc[:, 0]), np.array(data.iloc[:, 1])
    texts = list(texts)

    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    texts = [text.translate(str.maketrans('', '', punctuation)) for text in texts]

    return texts, labels

def create_vocab(texts, n=1000):
    vocab = {}
    for text in tqdm(texts):
        for word in text.split():
            word = word.lower()
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab = {word[0]: i for i, word in enumerate(vocab[:n])}
    with open('vocab.json','w') as f:
        json.dump(vocab,f,ensure_ascii=False,indent=4)
    return vocab

def text_to_feature_vector_bow(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word.lower() in vocab:
            vector[vocab[word.lower()]] += 1
    return vector

def text_to_feature_vector_glove(text):
    glove_path = 'dataset/glove.6B.100d.txt'
    with open(glove_path, 'r') as f:
        glove = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove[word] = vector
    
    vector = np.zeros(100)
    for word in text.split():
        if word.lower() in glove:
            vector += glove[word.lower()]
    vector /= len(text.split())
    return vector

if __name__ == '__main__':
    texts, labels = read_data('test')
    vocab = create_vocab(texts)
    X_bow = text_to_feature_vector_bow(texts[0], vocab)
    X_glove = text_to_feature_vector_glove(texts[0])
    print('bow shape:', X_bow.shape)
    print('glove shape:', X_glove.shape)
