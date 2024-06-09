import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_utils import read_data, create_vocab, mp_text_to_feature_vector, train_val_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        validate_model(model, valid_loader, criterion)

def validate_model(model, valid_loader, criterion):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            valid_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    valid_loss = valid_loss / len(valid_loader)
    valid_accuracy = 100 * correct / total
    print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.2f}%")


def test_model(model, test_loader):
    model.eval()
    # test with precision recall f1
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            tp += ((predicted == y_batch) & (y_batch == 1)).sum().item()
            fp += ((predicted != y_batch) & (y_batch == 0)).sum().item()
            tn += ((predicted == y_batch) & (y_batch == 0)).sum().item()
            fn += ((predicted != y_batch) & (y_batch == 1)).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Test Accuracy: {(tp + tn) / (tp + fp + tn + fn):.4f}")

if __name__ == '__main__':
    method = 'glove'
    learning_rate = 0.001
    num_epochs = 5

    train_texts, train_labels = read_data('train')
    vocab = create_vocab(train_texts)
    X_train = mp_text_to_feature_vector(train_texts, method=method, vocab=vocab)
    y_train = np.array(train_labels)
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_size=0.1)

    test_texts, test_labels = read_data('test')
    X_test = mp_text_to_feature_vector(test_texts, method=method, vocab=vocab)
    y_test = np.array(test_labels)

    y_train -= 1
    y_val -= 1
    y_test -= 1

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train))
    valid_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(set(train_labels))
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs)
    test_model(model, test_loader)
