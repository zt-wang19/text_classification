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
    return sorted(list(vocab))

def text_to_feature_vector_bow(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word.lower() in vocab:
            vector[vocab.index(word.lower())] += 1
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
