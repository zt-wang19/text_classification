## Text Classification Project

### dataloader utils
put yelp and glove files under directory `datasets/`

```python
train_texts = read_data(split='train')
test_texts = read_data(split='test')

# for bow
vocab = create_vocab(train_texts)
X_bow = text_to_feature_vector_bow(text, vocab)

# for glove
X_glove = text_to_feature_vector_glove(text)

# mp
X_bow = mp_text_to_feature_vector(texts, method='bow', vocab=vocab)
X_glove = mp_text_to_feature_vector(texts, method='glove')
```
