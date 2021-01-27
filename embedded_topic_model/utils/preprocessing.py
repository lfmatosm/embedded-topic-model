from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse


def _remove_empty_documents(documents):
    return [doc for doc in documents if doc != []]


def _create_list_words(documents):
    return [word for document in documents for word in document]


def _create_document_indices(documents):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(documents)]
    return [int(x) for y in aux for x in y]


def _create_bow(document_indices, words, num_docs, vocab_size):
    return sparse.coo_matrix(
        ([1] *
         len(document_indices),
         (document_indices,
          words)),
        shape=(
            num_docs,
            vocab_size)).tocsr()


def _split_bow(bow_in, num_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(num_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(num_docs)]
    return indices, counts


def _create_dictionaries(vocabulary):
    word2id = dict([(w, j) for j, w in enumerate(vocabulary)])
    id2word = dict([(j, w) for j, w in enumerate(vocabulary)])

    return word2id, id2word


def _to_numpy_array(documents):
    return np.array([[np.array(doc) for doc in documents]],
                    dtype=object).squeeze()


def create_etm_datasets(
        dataset,
        train_size=1.0,
        min_df=1,
        max_df=100.0,
        debug_mode=False):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorized_documents = vectorizer.fit_transform(dataset)

    documents_without_stop_words = [
        [word for word in document.split()
            if word not in vectorizer.stop_words_]
        for document in dataset]

    signed_documents = vectorized_documents.sign()

    if debug_mode:
        print('Building vocabulary...')

    sum_counts = signed_documents.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
    word2id = dict([(w, vectorizer.vocabulary_.get(w))
                    for w in vectorizer.vocabulary_])
    id2word = dict([(vectorizer.vocabulary_.get(w), w)
                    for w in vectorizer.vocabulary_])

    if debug_mode:
        print('Initial vocabulary size: {}'.format(v_size))

    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)
    # Creates vocabulary
    vocabulary = [id2word[idx_sort[cc]] for cc in range(v_size)]

    # Split in train/test
    if debug_mode:
        print('Tokenizing documents and splitting into train/test...')

    num_docs = signed_documents.shape[0]
    train_dataset_size = int(np.floor(train_size * num_docs))
    test_dataset_size = int(num_docs - train_dataset_size)
    idx_permute = np.random.permutation(num_docs).astype(int)

    # Remove words not in train_data
    vocabulary = list(set([w for idx_d in range(train_dataset_size)
                           for w in documents_without_stop_words[idx_permute[idx_d]] if w in word2id]))

    # Create dictionary and inverse dictionary
    word2id, id2word = _create_dictionaries(vocabulary)

    if debug_mode:
        print(
            'vocabulary after removing words not in train: {}'.format(
                len(vocabulary)))

    docs_train = [[word2id[w] for w in documents_without_stop_words[idx_permute[idx_d]]
                   if w in word2id] for idx_d in range(train_dataset_size)]
    docs_test = [
        [word2id[w] for w in \
            documents_without_stop_words[idx_permute[idx_d + train_dataset_size]] \
                if w in word2id] for idx_d in range(test_dataset_size)]

    if debug_mode:
        print(
            'Number of documents (train_dataset): {} [this should be equal to {}]'.format(
                len(docs_train),
                train_dataset_size))
        print(
            'Number of documents (test_dataset): {} [this should be equal to {}]'.format(
                len(docs_test),
                test_dataset_size))

    if debug_mode:
        print('Removing empty documents...')

    docs_train = _remove_empty_documents(docs_train)
    docs_test = _remove_empty_documents(docs_test)

    # Remove test documents with length=1
    docs_test = [doc for doc in docs_test if len(doc) > 1]

    # Obtains the training and test datasets as word lists
    words_train = [[id2word[w] for w in doc] for doc in docs_train]
    words_test = [[id2word[w] for w in doc] for doc in docs_test]

    docs_test_h1 = [[w for i, w in enumerate(
        doc) if i <= len(doc) / 2.0 - 1] for doc in docs_test]
    docs_test_h2 = [[w for i, w in enumerate(
        doc) if i > len(doc) / 2.0 - 1] for doc in docs_test]

    words_train = _create_list_words(docs_train)
    words_test = _create_list_words(docs_test)
    words_ts_h1 = _create_list_words(docs_test_h1)
    words_ts_h2 = _create_list_words(docs_test_h2)

    if debug_mode:
        print('  len(words_train): ', len(words_train))
        print('  len(words_test): ', len(words_test))
        print('  len(words_ts_h1): ', len(words_ts_h1))
        print('  len(words_ts_h2): ', len(words_ts_h2))

    doc_indices_train = _create_document_indices(docs_train)
    doc_indices_test = _create_document_indices(docs_test)
    doc_indices_test_h1 = _create_document_indices(docs_test_h1)
    doc_indices_test_h2 = _create_document_indices(docs_test_h2)

    if debug_mode:
        print('  len(np.unique(doc_indices_train)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_train)), len(docs_train)))
        print('  len(np.unique(doc_indices_test)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test)), len(docs_test)))
        print('  len(np.unique(doc_indices_test_h1)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test_h1)), len(docs_test_h1)))
        print('  len(np.unique(doc_indices_test_h2)): {} [this should be {}]'.format(
            len(np.unique(doc_indices_test_h2)), len(docs_test_h2)))

    # Number of documents in each set
    n_docs_train = len(docs_train)
    n_docs_test = len(docs_test)
    n_docs_test_h1 = len(docs_test_h1)
    n_docs_test_h2 = len(docs_test_h2)

    bow_train = _create_bow(
        doc_indices_train,
        words_train,
        n_docs_train,
        len(vocabulary))
    bow_test = _create_bow(
        doc_indices_test,
        words_test,
        n_docs_test,
        len(vocabulary))
    bow_test_h1 = _create_bow(
        doc_indices_test_h1,
        words_ts_h1,
        n_docs_test_h1,
        len(vocabulary))
    bow_test_h2 = _create_bow(
        doc_indices_test_h2,
        words_ts_h2,
        n_docs_test_h2,
        len(vocabulary))

    bow_train_tokens, bow_train_counts = _split_bow(bow_train, n_docs_train)
    bow_test_tokens, bow_test_counts = _split_bow(bow_test, n_docs_test)
    bow_test_h1_tokens, bow_test_h1_counts = _split_bow(
        bow_test_h1, n_docs_test_h1)
    bow_test_h2_tokens, bow_test_h2_counts = _split_bow(
        bow_test_h2, n_docs_test_h2)

    train_dataset = {
        'tokens': _to_numpy_array(bow_train_tokens),
        'counts': _to_numpy_array(bow_train_counts),
    }

    test_dataset = {
        'test': {
            'tokens': _to_numpy_array(bow_test_tokens),
            'counts': _to_numpy_array(bow_test_counts),
        },
        'test1': {
            'tokens': _to_numpy_array(bow_test_h1_tokens),
            'counts': _to_numpy_array(bow_test_h1_counts),
        },
        'test2': {
            'tokens': _to_numpy_array(bow_test_h2_tokens),
            'counts': _to_numpy_array(bow_test_h2_counts),
        }
    }

    return vocabulary, train_dataset, test_dataset
