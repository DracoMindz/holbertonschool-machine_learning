# Natural Language Processing - Word Embeddings
Specialization - Machine Learning Supervised Learning


##  Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### **General**
```
    What is natural language processing?
    What is a word embedding?
    What is bag of words?
    What is TF-IDF?
    What is CBOW?
    What is a skip-gram?
    What is an n-gram?
    What is negative sampling?
    What is word2vec, GloVe, fastText, ELMo?
```

For this project will need to dowload/install Gensim 3.8.x

## Tasks

**0. Bag Of Words**

Write a function def bag_of_words(sentences, vocab=None):
that creates a bag of words embedding matrix.
---
**1. TF-IDF**

Write a function def tf_idf(sentences, vocab=None):
that creates a TF-IDF embedding.
---
**2. Train Word2Vec**

Write a function def word2vec_model(sentences, size=100, min_count=5,
window=5, negative=5, cbow=True, iterations=5, seed=0):
that creates and trains a genism word2vec model.
---
**3. Extract Word2Vec**

Write a function def gensim_to_keras(model): that
gets the converts the gensim word2vec model to a keras layer.
---
**4. FastText**

Write a function def fasttext(sentences, size=100, min_count=5,
negative=5, window=5, cbow=True, iterations=5, seed=0):
that creates and trains a genism fastText model.
---

**5. ELMo**
When training an ELMo embedding model, what are you training.

---


