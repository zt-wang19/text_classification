from data_utils import read_data,create_vocab,text_to_feature_vector_bow
import numpy as np
from tqdm import tqdm
# 将文本转换为词袋模型
texts, labels = read_data('train')
vocab = create_vocab(texts)
print(len(vocab))
X_train = np.array([text_to_feature_vector_bow(text, vocab) for text in tqdm(texts)])
y_train = np.array(labels)

texts, labels = read_data('test')
X_test = np.array([text_to_feature_vector_bow(text, vocab) for text in tqdm(texts)])
y_test = np.array(labels)


import numpy as np

# 定义高斯核函数
def rbf_kernel(X1, X2, gamma=0.1):
    if len(X1.shape) == 1:
        X1 = X1.reshape(1, -1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(1, -1)
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

# 定义支持向量机模型
class SVM:
    def __init__(self, kernel=rbf_kernel, learning_rate=0.001, lambda_param=0.01, n_iters=1000, gamma=0.1):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.gamma = gamma
        self.alpha = None
        self.b = 0

    def fit(self, X, y):
        y = np.where(y == 1, -1, 1)
        n_samples, n_features = X.shape
        
        self.alpha = np.zeros(n_samples)
        self.b = 0

        K = self.kernel(X, X, self.gamma)

        for _ in tqdm(range(self.n_iters),desc='training'):
            for i in range(n_samples):
                condition = y[i] * (np.sum(self.alpha * y * K[:, i]) - self.b) >= 1
                if condition:
                    self.alpha[i] -= self.learning_rate * (2 * self.lambda_param * self.alpha[i])
                else:
                    self.alpha[i] -= self.learning_rate * (2 * self.lambda_param * self.alpha[i] - np.sum(y * K[:, i]))
                    self.b -= self.learning_rate * y[i]

    def predict(self, X):
        K = self.kernel(X, self.X_fit, self.gamma)
        return np.sign(np.dot(K, self.alpha * self.y_fit) - self.b)

model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000, gamma=0.1)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 将预测结果转换回原始标签
y_pred = np.where(y_pred == -1, 1, 2)

print("Accuracy:", y_test[y_test == y_pred].shape[0] / y_test.shape[0])
