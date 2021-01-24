import gensim
import numpy as np


# Class for a memory-friendly iterator over the dataset
class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def create_word2vec_embedding_from_dataset(
    dataset, dim_rho=300, min_count=1, sg=1,
    workers=25, negative_samples=10, window_size=4, iters=50,
    embedding_file_path=None):
    """
    Creates a Word2Vec embedding from dataset file or a list of sentences. 
    If a file path is given, the file must be composed
    by a sequence of sentences separated by \\n.

    If the dataset is big, prefer using its file path.

    Parameters:
        dataset (str or list of str): txt file containing the dataset or a list of sentences
        dim_rho (int): dimensionality of the word embeddings
        min_count (int): minimum term frequency (to define the vocabulary)
        sg (int): whether to use skip-gram
        workers (int): number of CPU cores
        negative_samples (int): number of negative samples
        window_size (int): window size to determine context
        iters (int): number of iterations
        embedding_file_path (str): optional. File to save the word embeddings
    
    Returns:
        dict: dictionary containing the mapping between words and their vector representations. 
        Example:
            { 'water': nd.array([0.024187922, 0.053684134, 0.034520667, ... ]) }
    """
    assert isinstance(dataset, str) or isinstance(dataset, list), \
        'dataset must be file path or list of sentences'

    if isinstance(dataset, str):
        assert isinstance(embedding_file_path, str), \
            'if dataset is a file path, an output embeddings file path must be given'

    sentences = MemoryFriendlyFileIterator(dataset) if isinstance(dataset, str) else [document.split() for document in dataset]
    model = gensim.models.Word2Vec(sentences, min_count=min_count, sg=sg, size=dim_rho, 
        iter=iters, workers=workers, negative=negative_samples, window=window_size)
    
    embeddings = {}
    for v in list(model.wv.vocab):
        vec = list(model.wv.__getitem__(v))
        embeddings[v] = np.array(vec).astype(np.float)
    
    # Write the embeddings to a file
    if embedding_file_path is not None:
        with open(embedding_file_path, 'w') as f:
            for word, vector in embeddings.items():
                vector_str = ['%.9f' % val for val in list(vector)]
                f.write(f'{word} {" ".join(vector_str)}\n')
    
    return embeddings
