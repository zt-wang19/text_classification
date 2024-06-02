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

# 定义最大熵模型
class MaxEnt:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _loss(self, X, y):
        m = len(y)
        logits = np.dot(X, self.weights)
        probs = self._softmax(logits)
        log_likelihood = -np.sum(np.log(probs[np.arange(m), y]))
        return log_likelihood / m

    def _gradient(self, X, y):
        m = len(y)
        logits = np.dot(X, self.weights)
        probs = self._softmax(logits)
        probs[np.arange(m), y] -= 1
        gradient = np.dot(X.T, probs) / m
        return gradient

    def fit(self, X, y, num_classes):
        num_features = X.shape[1]
        self.weights = np.zeros((num_features, num_classes))

        for i in range(self.max_iter):
            gradient = self._gradient(X, y)
            self.weights -= self.learning_rate * gradient

            if i % 100 == 0:
                loss = self._loss(X, y)
                print(f"Iteration {i} - Loss: {loss}")

    def predict(self, X):
        logits = np.dot(X, self.weights)
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

# 训练模型
num_classes = len(set(labels))
model = MaxEnt(learning_rate=0.01, max_iter=1000)
model.fit(X, y, num_classes)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model.fit(X_train, y_train, num_classes)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告和准确率
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
