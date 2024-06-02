# 示例数据集
texts = [
    'I love programming in Python',
    'Python is a great language',
    'I hate bugs in the code',
    'Debugging is fun',
    'I enjoy learning new programming languages',
    'Machine learning with Python is awesome'
]
labels = [1, 1, 0, 1, 1, 1]  # 1表示正面评论，0表示负面评论

import numpy as np

# 将文本转换为词袋模型
def create_vocab(texts):
    vocab = set()
    for text in texts:
        for word in text.split():
            vocab.add(word.lower())
    return sorted(list(vocab))

def text_to_feature_vector(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word.lower() in vocab:
            vector[vocab.index(word.lower())] += 1
    return vector

vocab = create_vocab(texts)
X = np.array([text_to_feature_vector(text, vocab) for text in texts])
y = np.array(labels)

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

        for _ in range(self.n_iters):
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
model.fit(X, y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 将预测结果转换回原始标签
y_pred = np.where(y_pred == -1, 0, 1)

# 输出分类报告和准确率
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
