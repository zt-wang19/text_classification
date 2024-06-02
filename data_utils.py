import time
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd

def read_data(split='train'):
    data = pd.read_csv(f"dataset/{split}.csv", header=None)

    labels, texts = np.array(data.iloc[:, 0]), np.array(data.iloc[:, 1])
    texts = list(texts)

    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    texts = [text.translate(str.maketrans('', '', punctuation)) for text in texts]

    return texts, labels

def create_vocab(texts):
    vocab = set()
    for text in texts:
        for word in text.split():
            vocab.add(word.lower())
    vocab = sorted(list(vocab))
    vocab = {word: i for i, word in enumerate(vocab)}
    return vocab

def text_to_feature_vector_bow(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word.lower() in vocab:
            vector[vocab[word.lower()]] += 1
    return vector

glove_path = 'dataset/glove.6B.100d.txt'
with open(glove_path, 'r') as f:
    glove = {}
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove[word] = vector

def text_to_feature_vector_glove(text):
    vector = np.zeros(100)
    words = text.split()
    for word in words:
        if word.lower() in glove:
            vector += glove[word.lower()]
    if len(words) > 0:
        vector /= len(words)
    return vector

def mp_text_to_feature_vector(texts, method='bow', vocab=None):
    
    start = time.time()
    
    pool = Pool(8)
    if method == 'bow':
        text_to_feature_vector_bow_partial = partial(text_to_feature_vector_bow, vocab=vocab)
        feature_vectors = pool.map(text_to_feature_vector_bow_partial, texts)
    elif method == 'glove':
        feature_vectors = pool.map(text_to_feature_vector_glove, texts)
    pool.close()
    pool.join()
    print(time.time() - start)
    return np.array(feature_vectors)

if __name__ == '__main__':
    texts, labels = read_data('train')
    vocab = create_vocab(texts)
    X_bow = text_to_feature_vector_bow(texts[0], vocab)
    X_glove = text_to_feature_vector_glove(texts[0])

    # for text in tqdm(texts):
    #     X_glove = text_to_feature_vector_glove(text)
    X_bow = mp_text_to_feature_vector(texts, method='bow', vocab=vocab)
    X_glove = mp_text_to_feature_vector(texts, method='glove')
    print('bow shape:', X_bow.shape)
    print('glove shape:', X_glove.shape)
