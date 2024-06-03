import numpy as np
from data_utils import read_data, create_vocab, mp_text_to_feature_vector, train_val_split

# 最大熵模型
class MaximumEntropyModel:
    def __init__(self, num_classes, learning_rate=0.02, max_iter=5000, tol=1e-4, patience=20, decay_rate=0.98):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def compute_loss_and_gradients(self, X, y):
        num_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = self._softmax(scores)
        
        # Loss: Negative Log-Likelihood
        correct_logprobs = -np.log(probs[range(num_samples), y])
        loss = np.sum(correct_logprobs) / num_samples
        
        # Gradients
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples
        
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0)
        
        return loss, dW, db

    def fit(self, X_train, y_train, X_val, y_val):
        num_features = X_train.shape[1]
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros(self.num_classes)

        best_loss = np.inf
        patience_counter = 0

        for i in range(self.max_iter):
            loss, dW, db = self.compute_loss_and_gradients(X_train, y_train)
            # Update weights
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            # Validation loss
            val_loss, _, _ = self.compute_loss_and_gradients(X_val, y_val)

            if i % 100 == 0:
                self.learning_rate *= self.decay_rate  # Learning rate decay
                print(f'Iteration {i}/{self.max_iter}, Loss: {loss} | Validation Loss: {val_loss}')
            
            # Early stopping
            if val_loss < best_loss - self.tol:
                best_loss = val_loss
                patience_counter = 0
            else: 
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'=====Early stopping at iteration {i}=====')
                    break


    def predict(self, X):
        logits = np.dot(X, self.weights)
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)


if __name__ == '__main__':
    # Params
    method = 'bow'  # 'bow' or 'glove'
    lr = 0.02  # 0.2
    max_iter = 5000
    tol = 1e-4
    patience = 20
    decay_rate = 0.98  # 0.99

    # Data & Embedding 
    texts, labels = read_data('train')
    vocab = create_vocab(texts)
    X = mp_text_to_feature_vector(texts, method=method, vocab=vocab)
    y = np.array(labels) - 1
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    # Train
    model = MaximumEntropyModel(num_classes=2, learning_rate=lr, max_iter=max_iter, tol=tol, patience=patience, decay_rate=decay_rate)
    model.fit(X_train, y_train, X_val, y_val)

    # Report Accuracy
    train_accuracy = np.mean(model.predict(X_train) == y_train)
    val_accuracy = np.mean(model.predict(X_val) == y_val)
    print(f'Train Accuracy: {train_accuracy} | Validation Accuracy: {val_accuracy}')

    # Test
    test_texts, test_labels = read_data('test')
    X_test = mp_text_to_feature_vector(test_texts, method=method, vocab=vocab)
    y_test = np.array(test_labels) - 1
    test_accuracy = np.mean(model.predict(X_test) == y_test)
    print('Test Accuracy:', test_accuracy)
    