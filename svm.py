from data_utils import read_data,create_vocab,text_to_feature_vector_bow,text_to_feature_vector_glove
import numpy as np
from tqdm import tqdm
# 将文本转换为词袋模型
texts, labels = read_data('train')
vocab = create_vocab(texts)

mode = 'glove'
if mode == 'bow':
    text_to_feature_vector = text_to_feature_vector_bow
elif mode == 'glove':
    text_to_feature_vector = text_to_feature_vector_glove
else:
    raise ValueError("Invalid mode")

X_train = np.array([text_to_feature_vector(text, vocab) for text in tqdm(texts)])
y_train = np.array(labels)

texts, labels = read_data('test')
X_test = np.array([text_to_feature_vector(text, vocab) for text in tqdm(texts)])
y_test = np.array(labels)


import numpy as np

# 定义支持向量机模型
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in tqdm(range(self.n_iters),desc='training'):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 训练模型
model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
model.fit(X_train, y_train)
model.save('svm.model')

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 将预测结果转换回原始标签
y_pred = np.where(y_pred == -1, 1, 2)

print("Accuracy:", y_test[y_test == y_pred].shape[0] / y_test.shape[0])
