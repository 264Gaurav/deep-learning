# Embeddings with Keras & TensorFlow

This README explains what embeddings are, why we use them, different types of word embeddings, and how to implement embeddings in TensorFlow/Keras (both learning embeddings during training and using pre-trained embeddings such as GloVe). The examples are Python / `tf.keras` based and assume TensorFlow 2.x.

---

## Table of contents

1. Overview
2. Why use embeddings?
3. Types of embeddings
4. The Keras `Embedding` layer (how it works)
5. Example: Learning an embedding from scratch (classification)
6. Example: Using pre-trained GloVe embeddings
7. Tips, best practices and debugging
8. References

---

## 1. Overview

An *embedding* is a dense vector representation of discrete items (typically words) where semantically similar items map to nearby vectors in a continuous vector space. Embeddings compress sparse, high-dimensional representations (e.g., one-hot vectors) into compact, dense vectors that capture useful properties and relationships.

---

## 2. Why use embeddings?

* **Dimensionality reduction**: convert very large sparse vectors (vocabulary-sized) to low-dimensional dense vectors.
* **Semantic generalization**: similar words get similar vectors, enabling models to generalize across related inputs.
* **Efficient learning**: fewer parameters and better numerical properties than sparse encodings.
* **Transfer learning**: pre-trained embeddings (e.g., Word2Vec, GloVe) allow reuse of semantic knowledge learned from large corpora.

---

## 3. Types of embeddings

* **One-hot encoding** — sparse, high-dimensional, no notion of similarity.
* **Learned embeddings (task-specific)** — trained as part of a neural network using an `Embedding` layer.
* **Predictive embeddings (Word2Vec)** — trained by predicting surrounding words; comes in CBOW and Skip-Gram variants.
* **Count-based embeddings (GloVe)** — trained on global word co-occurrence statistics.
* **Subword-aware embeddings (FastText)** — uses character n-grams so rare words share subword representations.
* **Contextual embeddings (BERT, GPT, etc.)** — produce token vectors that depend on context (sentence-level); obtained from transformer models.

---

## 4. The Keras `Embedding` layer (how it works)

Key arguments:

* `input_dim` — size of vocabulary (integer IDs range 0..input\_dim-1).
* `output_dim` — embedding vector size (dimensionality of embeddings).
* `input_length` — length of input sequences (number of tokens per sample).

Behavior:

* The layer is essentially a lookup table (matrix) of shape `(input_dim, output_dim)`.
* Input integer sequences are mapped to sequences of vectors by indexing into this matrix.
* The embedding matrix is trainable by default (weights updated during backprop).

Example API (tf.keras):

```python
from tensorflow.keras.layers import Embedding

emb = Embedding(input_dim=200, output_dim=32, input_length=50)
```

Output shape for a batch of shape `(batch_size, input_length)` is `(batch_size, input_length, output_dim)`.

---

## 5. Example: Learning an embedding from scratch

This example shows a tiny sentiment classification model that learns embeddings during training.

```python
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# sample data
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']
labels = np.array([1,1,1,1,1,0,0,0,0,0])

# integer encode (hashing via one_hot)
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]

# pad sequences
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# fit
model.fit(padded_docs, labels, epochs=50, verbose=2)
```

After training, the `Embedding` layer weights (`model.layers[0].get_weights()`) contain the learned embedding matrix.

---

## 6. Example: Using pre-trained GloVe embeddings

Steps:

1. Download pre-trained GloVe vectors (e.g., `glove.6B.100d.txt`).
2. Build a tokenizer on your dataset and create `word_index`.
3. Create an embedding matrix where each row index (word id) contains the pre-trained vector for that word (or random vector if missing).
4. Create an `Embedding` layer and set `weights=[embedding_matrix]` and `trainable=False` (or True to fine-tune).

Example code (outline):

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# assume texts and labels are prepared
max_words = 10000
maxlen = 100
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=maxlen)
word_index = tokenizer.word_index

# load GloVe
emb_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        emb_index[word] = coefs

# prepare embedding matrix
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = emb_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# model
model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=maxlen))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, labels, epochs=10, batch_size=32)
```

---

## 7. Tips, best practices and debugging

* **Embedding size**: typical sizes are 50, 100, 200, 300; choose by dataset size and task complexity. More dimensions = more capacity but risk overfitting.
* **Vocabulary size**: limit `num_words` to most frequent words; map rare words to an `<OOV>` id.
* **Padding/truncation**: be consistent with `input_length` and padding strategy (`pre` vs `post`).
* **Freezing vs fine-tuning**: freeze pre-trained embeddings at first to avoid catastrophic forgetting; optionally unfreeze later for fine-tuning.
* **Saving embeddings**: save the `Embedding` layer weights or the entire model (`model.save()`).
* **Inspect embeddings**: use dimensionality reduction (t-SNE, PCA) to visualize vectors and sanity-check clusters.
* **Contextual embeddings**: if you need stronger contextual representations, use transformer-based models (BERT/DistilBERT) via Hugging Face and extract token embeddings.

---

## 8. References

* Tutorial inspiration and examples adapted from *Machine Learning Mastery: How to Use Word Embedding Layers for Deep Learning with Keras*.
* TensorFlow / Keras API documentation for `Embedding` layer.

---
