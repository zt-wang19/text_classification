import os
import json

import torch

from mlp import MLP
from data_utils import read_data, create_vocab, text_to_feature_vector_bow

if __name__ == '__main__':
    with open('dataset/vocab.json', 'r') as vocab_f:
        vocab = json.load(vocab_f)
    
    input_dim = len(vocab)
    hidden_dim = 128
    output_dim = 2

    model = MLP(input_dim, hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load('ckpt/mlp.pt')
    model.load_state_dict(sd)
    model = model.to(device)

    while True:
        text = input("Please input the text >> ")
        if text == "exit":
            break
        vector = text_to_feature_vector_bow(text, vocab)
        vector = torch.tensor(vector).float().to(device)
        output = model(vector)
        _, predicted = torch.max(output.data, 0)
        flag = predicted.item()
        if flag == 0:
            keyword = "negative"
        else:
            keyword = "positive"
        print(f"This passage is {keyword}!")
