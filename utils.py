import torch 
import numpy as np
from gensim.models import CoherenceModel

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))
    return TD

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab, top_n = 10):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        beta_top_n = list(beta[k].argsort()[-top_n:][::-1])
        top_words = [vocab[a] for a in beta_top_n]
        TC_k = 0
        counter = 0
        for i, word in enumerate(beta_top_n):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(beta_top_n) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, beta_top_n[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))
    return TC


def get_gensim_coherence(topics, documents, dictionary, coherence):
    """Calculates topic coherence using gensim's coherence pipeline.

    Parameters:

    topics (list of str list): topic words for each topic
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    coherence (str): coherence type. Can be 'c_v', 'u_mass', 'c_uci' or 'c_npmi'

    Returns:

    float: coherence score
    """
    return CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence=coherence
    ).get_coherence()


def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
