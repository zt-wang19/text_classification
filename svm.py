from data_utils import read_data,create_vocab,mp_text_to_feature_vector
import numpy as np
from tqdm import tqdm
# 将文本转换为词袋模型

# method = 'bow'
method = 'glove'
texts, labels = read_data('train')
vocab = create_vocab(texts,n=1000)
X_train = mp_text_to_feature_vector(texts, method=method, vocab=vocab)
y_train = np.array(labels)

texts, labels = read_data('test')
X_test = mp_text_to_feature_vector(texts,method=method, vocab=vocab)
y_test = np.array(labels)

def eval(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 2)
    print(y_test)
    print(y_pred)
    acc =  y_test[y_test == y_pred].shape[0] / y_test.shape[0]
    # 二分类计算precision,recall,f1
    precision = np.sum(y_test[y_test == y_pred] == 2) / np.sum(y_pred == 2)
    recall = np.sum(y_test[y_test == y_pred] == 2) / np.sum(y_test == 2)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'accuracy: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    return acc
    


EVAL_STEPS = 1
# 定义支持向量机模型
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.where(y == 1, -1, 1)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for i in tqdm(range(self.n_iters),desc='training',disable=True):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]
            if i % EVAL_STEPS == 0:
                acc = eval(self,X_test,y_test)
                print(f'iteration {i} completed, accuracy: {acc}')

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 训练模型
model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=100)
model.fit(X_train, y_train)

final_acc = eval(model,X_test,y_test)
# print(f'final accuracy: {final_acc}')
