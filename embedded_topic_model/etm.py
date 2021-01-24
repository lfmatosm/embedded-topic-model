#/usr/bin/python

from __future__ import print_function

import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import joblib
import scipy.io

from torch import nn, optim
from torch.nn import functional as F

from embedded_topic_model import data
from embedded_topic_model.model import Model
from embedded_topic_model.utils import metrics


class ETM(object):
    """
    Creates an embedded topic model instance. The model hyperparameters are:

        vocabulary (list of str): training dataset vocabulary
        embeddings (str or dict): directory containing word embeddings or dictionary containing word embeddings mapping
        model_path (str): path to save trained model
        batch_size (int): input batch size for training
        num_topics (int): number of topics
        rho_size (int): dimension of rho
        emb_size (int): dimension of embeddings
        t_hidden_size (int): dimension of hidden space of q(theta)
        theta_act (str): tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)
        train_embeddings (int): whether to fix rho or train it
        lr (float): learning rate
        lr_factor (float): divide learning rate by this...
        epochs (int): number of epochs to train...150 for 20ng 100 for others
        optimizer_type (str): choice of optimizer
        seed (int): random seed (default: 1)
        enc_drop (float): dropout rate on encoder
        clip (float): gradient clipping
        nonmono (int): number of bad hits allowed
        wdecay (float): some l2 regularization
        anneal_lr (bool): whether to anneal the learning rate or not
        bow_norm (bool): normalize the bows or not
        num_words (int): number of words for topic viz
        log_interval (int): when to log training
        visualize_every (int): when to visualize results
        eval_batch_size (int): input batch size for evaluation
        load_from (str): the name of the ckpt to eval from
        tc (bool): whether to compute topic coherence or not
        td (bool): whether to compute topic diversity or not
        eval_perplexity (bool): whether to compute perplexity on document completion task
        debug_mode (bool): wheter or not should log model operations
    """
    def __init__(self, vocabulary, embeddings=None, model_path=None,
        batch_size=1000, num_topics=50, rho_size=300, emb_size=300, t_hidden_size=800,
        theta_act='relu', train_embeddings=False, lr=0.005, lr_factor=4.0, epochs=20,
        optimizer_type='adam', seed=2019, enc_drop=0.0, clip=0.0,
        nonmono=10, wdecay=1.2e-6, anneal_lr=False, bow_norm=True, num_words=10,
        log_interval=2, visualize_every=10, eval_batch_size=1000, load_from='', tc=False,
        td=False, eval_perplexity=False, debug_mode=True,
    ):
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.model_path = model_path if model_path is not None else f'./etm_{num_topics}.etm'
        self.batch_size = batch_size
        self.num_topics = num_topics
        self.rho_size = rho_size
        self.emb_size = emb_size
        self.t_hidden_size = t_hidden_size
        self.theta_act = theta_act
        self.train_embeddings = train_embeddings
        self.lr = lr
        self.lr_factor = lr_factor
        self.epochs = epochs
        self.seed = seed
        self.enc_drop = enc_drop
        self.clip = clip
        self.nonmono = nonmono
        self.wdecay = wdecay
        self.anneal_lr = anneal_lr
        self.bow_norm = bow_norm
        self.num_words = num_words
        self.log_interval = log_interval
        self.visualize_every = visualize_every
        self.eval_batch_size = eval_batch_size
        self.load_from = load_from
        self.tc = tc
        self.td = td
        self.eval_perplexity = eval_perplexity
        self.debug_mode = debug_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.embeddings = None if self.train_embeddings else self._initialize_embeddings(embeddings)

        ## define model and optimizer
        self.model = Model(self.device, self.num_topics, self.vocabulary_size, self.t_hidden_size, self.rho_size, self.emb_size, 
                        self.theta_act, self.embeddings, self.train_embeddings, self.enc_drop).to(self.device)
        self.optimizer = self._get_optimizer(optimizer_type)
    

    def __str__(self):
        return f'{self.model}'
    

    def _initialize_embeddings(self, embeddings):
        vectors = embeddings if isinstance(embeddings, dict) else {}

        if isinstance(embeddings, str):
            with open(embeddings, 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    if word in self.vocabulary:
                        vect = np.array(line[1:]).astype(np.float)
                        vectors[word] = vect
        
        model_embeddings = np.zeros((self.vocabulary_size, self.emb_size))

        words_found = 0
        for i, word in enumerate(self.vocabulary):
            try: 
                model_embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                model_embeddings[i] = np.random.normal(scale=0.6, size=(self.emb_size, ))
        return torch.from_numpy(model_embeddings).to(self.device)


    def _get_optimizer(self, optimizer_type):
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        elif optimizer_type == 'adadelta':
            return optim.Adadelta(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        elif optimizer_type == 'asgd':
            return optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0., weight_decay=self.wdecay)
        else:
            if self.debug_mode:
                print('Defaulting to vanilla SGD')
            return optim.SGD(self.model.parameters(), lr=self.lr)


    def _set_training_data(self, train_data):
        self.train_tokens = train_data['tokens']
        self.train_counts = train_data['counts']
        self.num_docs_train = len(self.train_tokens)
    

    def _set_test_data(self, test_data):
        self.test_tokens = test_data['test']['tokens']
        self.test_counts = test_data['test']['counts']
        self.num_docs_test = len(self.test_tokens)
        self.test_1_tokens = test_data['test1']['tokens']
        self.test_1_counts = test_data['test1']['counts']
        self.num_docs_test_1 = len(self.test_1_tokens)
        self.test_2_tokens = test_data['test2']['tokens']
        self.test_2_counts = test_data['test2']['counts']
        self.num_docs_test_2 = len(self.test_2_tokens)


    def _train(self, epoch):
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.num_docs_train)
        indices = torch.split(indices, self.batch_size)
        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()

            data_batch = data.get_batch(self.train_tokens, self.train_counts, ind, self.vocabulary_size, self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self.model(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()

            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2) 
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                if self.debug_mode:
                    print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                        epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        
        cur_loss = round(acc_loss / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)

        if self.debug_mode:
            print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))


    def get_perplexity_for_document_completion(self, test_data):
        """Compute perplexity on document completion.
        """
        self._set_test_data(test_data)

        self.model.eval()
        with torch.no_grad():
            ## get \beta here
            beta = self.model.get_beta()

            ### do dc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(self.num_docs_test_1)), self.eval_batch_size)
            for idx, ind in enumerate(indices_1):
                ## get theta from first half of docs
                data_batch_1 = data.get_batch(self.test_1_tokens, self.test_1_counts, ind, self.vocabulary_size, self.device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if self.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.model.get_theta(normalized_data_batch_1)

                ## get prediction loss using second half
                data_batch_2 = data.get_batch(self.test_2_tokens, self.test_2_counts, ind, self.vocabulary_size, self.device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)
                
                loss = recon_loss / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)

            if self.debug_mode:
                print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            
            return ppl_dc
    

    def get_topics(self, top_n_words = 10):
        ## get topics using monte carlo
        with torch.no_grad():
            topics = []
            gammas = self.model.get_beta()

            for k in range(self.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-top_n_words:][::-1])
                topic_words = [self.vocabulary[a] for a in top_words]
                topics.append(topic_words)
    
            return topics
    

    def visualize_word_embeddings(self, queries):
        self.model.eval()

        ## visualize word embeddings by using V to get nearest neighbors
        with torch.no_grad():
            try:
                self.embeddings = self.model.rho.weight  # Vocab_size x E
            except:
                self.embeddings = self.model.rho         # Vocab_size x E
            
            neighbors = {}
            for word in queries:
                neighbors[word] = metrics.nearest_neighbors(word, self.embeddings, self.vocabulary)
            
            return neighbors


    def fit(self, train_data, test_data = None):
        self._set_training_data(train_data)

        ## train model on data 
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []

        if self.debug_mode:
            print(f'Topics before training: {self.get_topics()}')

        for epoch in range(1, self.epochs):
            self._train(epoch)

            if self.eval_perplexity:
                val_ppl = self.get_perplexity_for_document_completion(test_data)
                if val_ppl < best_val_ppl:
                    self.save_model(self.model_path)
                    best_epoch = epoch
                    best_val_ppl = val_ppl
                else:
                    ## check whether to anneal lr
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.anneal_lr and (len(all_val_ppls) > self.nonmono and val_ppl > min(all_val_ppls[:-self.nonmono]) and lr > 1e-5):
                        self.optimizer.param_groups[0]['lr'] /= self.lr_factor
            
                all_val_ppls.append(val_ppl)
            
            if self.debug_mode and (epoch % self.visualize_every == 0):
                print(f'Topics: {self.get_topics()}')
        
        self.save_model(self.model_path)

        if self.eval_perplexity:
            self.load_model(self.model_path)
            val_ppl = self.get_perplexity_for_document_completion(train_data)


    def get_topic_word_matrix(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta()

            topics = []

            for i in range(self.num_topics):
                words = list(beta[i].cpu().numpy())
                topic_words = [self.vocabulary[a] for a, _ in enumerate(words)]
                topics.append(topic_words)

            return topics
    

    def get_topic_word_dist(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            return self.model.get_beta()
    

    def get_document_topic_dist(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            indices = torch.tensor(range(self.num_docs_train))
            indices = torch.split(indices, self.batch_size)

            thetas = []

            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(self.train_tokens, self.train_counts, ind, self.vocabulary_size, self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums if self.bow_norm else data_batch
                theta, _ = self.model.get_theta(normalized_data_batch)

                thetas.append(theta)

            return torch.cat(tuple(thetas), 0)


    def get_topic_coherence(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta().data.cpu().numpy()
            return metrics.get_topic_coherence(beta, self.train_tokens, self.vocabulary)
    

    def get_topic_diversity(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            beta = self.model.get_beta().data.cpu().numpy()
            return metrics.get_topic_diversity(beta)


    def save_model(self, model_path):
        assert self.model is not None, \
            'no model to save'

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as file:
            torch.save(self.model, file)
    

    def load_model(self, model_path):
        assert os.path.exists(model_path), \
            "model path doesn't exists"
        
        with open(model_path, 'rb') as file:
            self.model = torch.load(file)
            self.model = self.model.to(self.device)
