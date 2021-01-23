from gensim.models import CoherenceModel
import pandas as pd, functools
from collections import Counter
from decimal import Decimal
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer


def get_coherence_score(topics, documents, dictionary, coherence, no_of_words=20):
    """Calculates topic coherence using gensim's coherence pipeline.

    Parameters:

    topics (list of str list): topic words for each topic
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    coherence (str): coherence type. Can be 'c_v', 'u_mass', 'c_uci' or 'c_npmi'

    Returns:

    float: coherence score
    """
    coherence_model = CoherenceModel(
                topics=topics, 
                texts=documents, 
                dictionary=dictionary, 
                coherence=coherence,
                processes=0,
                topn=no_of_words
    )

    return coherence_model.get_coherence()


def get_topic_diversity(topics, no_of_words=20):
    """Calculates topic diversity for given topics. Topic diversity is defined as 
    the frequency in which words are associated with a single topic

    Parameters:

    topics (list of str list): topic words for each topic

    Returns:

    float: diversity score
    """
    word_frequencies_per_topic = [Counter(topic[:no_of_words]) for topic in topics]
    word_frequencies = functools.reduce(lambda dict1, dict2 : {x: Decimal(dict1.get(x, 0)) + Decimal(dict2.get(x, 0)) \
        for x in set(dict1).union(dict2)}, word_frequencies_per_topic)
    
    topic_diversity_scores = []
    for topic in topics:
        topic_diversity = 0
        for i in range(0, no_of_words):
            if word_frequencies[topic[i]] == Decimal(1):
                topic_diversity = topic_diversity + 1
        
        topic_diversity_scores.append(Decimal(topic_diversity) / Decimal(no_of_words))
    
    return float(Decimal(sum(topic_diversity_scores)) / Decimal(len(topic_diversity_scores)))


def create_dictionary(documents):
    """Creates word dictionary for given corpus.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset
    """
    dictionary = Dictionary(documents)
    dictionary.compactify()

    return dictionary


def filter_tokens_by_frequencies(documents, min_df=1, max_df=1.0):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit_transform(documents)
    
    return [[word for word in document if word not in vectorizer.stop_words_] for document in documents]


def get_topic_word_matrix(beta, k_values, vocab):
    topics = []

    for k in range(k_values):
        words = list(beta[k].cpu().numpy())
        topic_words = [vocab[a] for a, _ in enumerate(words)]
        topics.append(topic_words)

    return topics