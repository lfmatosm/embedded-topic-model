# Embedded Topic Model
[![PyPI version](https://badge.fury.io/py/embedded-topic-model.svg)](https://badge.fury.io/py/embedded-topic-model)
[![Actions Status](https://github.com/lffloyd/embedded-topic-model/workflows/Python%20package/badge.svg)](https://github.com/lffloyd/embedded-topic-model/actions)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/lffloyd/embedded-topic-model/blob/main/LICENSE)

This package was made to easily run embedded topic modelling on a given corpus.

ETM is a topic model that marries the probabilistic topic modelling of Latent Dirichlet Allocation with the
contextual information brought by word embeddings-most specifically, word2vec. ETM models topics as points
in the word embedding space, arranging together topics and words with similar context.
As such, ETM can either learn word embeddings alongside topics, or be given pretrained embeddings to discover
the topic patterns on the corpus.

ETM was originally published by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei on a article titled ["Topic Modeling in Embedding Spaces"](https://arxiv.org/abs/1907.04907) in 2019. This code is an adaptation of the [original](https://github.com/adjidieng/ETM) provided with the article and is not affiliated in any manner with the original authors. Most of the original code was kept here, with some changes here and there, mostly for ease of usage. This package was created to facilitate research purposes. If you want a more stable and feature-rich package to train ETM and other models, take a look at [OCTIS](https://github.com/MIND-Lab/OCTIS).

With the tools provided here, you can run ETM on your dataset using simple steps.

# Index

* [:beer: Installation](#beer-installation)
* [:wrench: Usage](#wrench-usage)
    * [:microscope: Examples](#microscope-examples)
* [:books: Citation](#books-citation)
* [:heart: Contributing](#heart-contributing)
* [:v: Acknowledgements](#v-acknowledgements)
* [:pushpin: License](#pushpin-license)

# :beer: Installation
You can install the package using ```pip``` by running: ```pip install -U embedded_topic_model```

# :wrench: Usage
To use ETM on your corpus, you must first preprocess the documents into a format understandable by the model.
This package has a quick-use preprocessing script. The only requirement is that the corpus must be composed
by a list of strings, where each string corresponds to a document in the corpus.

You can preprocess your corpus as follows:

```python
from embedded_topic_model.utils import preprocessing
import json

# Loading a dataset in JSON format. As said, documents must be composed by string sentences
corpus_file = 'datasets/example_dataset.json'
documents_raw = json.load(open(corpus_file, 'r'))
documents = [document['body'] for document in documents_raw]

# Preprocessing the dataset
vocabulary, train_dataset, _, = preprocessing.create_etm_datasets(
    documents, 
    min_df=0.01, 
    max_df=0.75, 
    train_size=0.85, 
)
```

Then, you can train word2vec embeddings to use with the ETM model. This is optional, and if you're not interested
on training your embeddings, you can either pass a pretrained word2vec embeddings file for ETM or learn the embeddings
using ETM itself. If you want ETM to learn its word embeddings, just pass ```train_embeddings=True``` as an instance parameter.

To pretrain the embeddings, you can do the following:

```python
from embedded_topic_model.utils import embedding

# Training word2vec embeddings
embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents)
```

To create and fit the model using the training data, execute:

```python
from embedded_topic_model.models.etm import ETM

# Training an ETM instance
etm_instance = ETM(
    vocabulary,
    embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                   # a KeyedVectors instance
    num_topics=8,
    epochs=100,
    debug_mode=True,
    train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                            # topic embeddings. By default, is False. If 'embeddings' argument
                            # is being passed, this argument must not be True
)

etm_instance.fit(train_dataset)
```

You can get the topic words with this method. Note that you can select how many word per topic you're interest in:
```python
t_w_mtx = etm_instance.get_topics(top_n_words=20)
```

You can get the topic word matrix with this method. Note that it will return all word for each topic:
```python
t_w_mtx = etm_instance.get_topic_word_matrix()
```

You can get the topic word distribution matrix and the document topic distribution matrix with the following methods, both return a normalized distribution matrix:
```python
t_w_dist_mtx = etm_instance.get_topic_word_dist()
d_t_dist_mtx = etm_instance.get_document_topic_dist()
```

Also, to obtain topic coherence or topic diversity of the model, you can do as follows:

```python
topics = etm_instance.get_topics(20)
topic_coherence = etm_instance.get_topic_coherence()
topic_diversity = etm_instance.get_topic_diversity()
```

You can also predict topics for unseen documents with the following.

```python
from embedded_topic_model.utils import preprocessing
from embedded_topic_model.models.etm import ETM

corpus_file = 'datasets/example_dataset.json'
documents_raw = json.load(open(corpus_file, 'r'))
documents = [document['body'] for document in documents_raw]

# Splits into train/test datasets
train = documents[:len(documents)-100]
test = documents[len(documents)-100:]

# Model fitting
# ...

# The vocabulary must be the same one created during preprocessing of the training dataset (see above)
preprocessed_test = preprocessing.create_bow_dataset(test, vocabulary)
# Transforms test dataset and returns normalized document topic distribution
test_d_t_dist = etm_instance.transform(preprocessed_test)
print(f'test_d_t_dist: {test_d_t_dist}')
```

For further details, see [examples](#microscope-examples).

## :microscope: Examples

| title                                       | link |
| :-------------:                             | :--: |
| ETM example - Reddit (r/depression) dataset | [Jupyter Notebook](./2023-09-01%20-%20reddit%20-%20depression%20dataset%20-%20etm%20-%20example.ipynb) |

# :books: Citation
To cite ETM, use the original article's citation:

```
@article{dieng2019topic,
    title = {Topic modeling in embedding spaces},
    author = {Dieng, Adji B and Ruiz, Francisco J R and Blei, David M},
    journal = {arXiv preprint arXiv: 1907.04907},
    year = {2019}
}
```

# :heart: Contributing
Contributions are always welcomed :heart:! You can take a look at []() to see some guidelines. Feel free to contact through issues, to elaborate on desired enhancements and to check if work is already being done on the matter.

# :v: Acknowledgements
Credits given to Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei for the original work.

# :pushpin: License
Licensed under [MIT](LICENSE) license.
