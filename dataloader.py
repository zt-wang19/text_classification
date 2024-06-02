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

# 定义多层感知机模型
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # 初始化权重
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = y.shape[0]

        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y] = 1

        d_z2 = output - y_one_hot
        d_W2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

    def fit(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            loss = self.cross_entropy_loss(y, output)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                print(f'Epoch {epoch} - Loss: {loss}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 训练模型
input_size = X.shape[1]
hidden_size = 5
output_size = len(set(labels))
model = MLP(input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000)
model.fit(X, y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告和准确率
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
