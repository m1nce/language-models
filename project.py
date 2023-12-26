# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):

    r = requests.get(url)
    texted = r.text.split("***")
    return texted[2].replace('\r\n', '\n')



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    paragraphs = re.split(r'\n\n+', book_string)
    paragraphs = filter(None, paragraphs)
    paragraphs_with_controls = map(lambda para: '\x02' + para + '\x03', paragraphs)
    modified_text = '\n'.join(paragraphs_with_controls)
    r = "\w+|[@#$%\^,.!?;:\'\"-]|\\x02|\\x03"
    return re.findall(r, modified_text)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        total = set(tokens)
        total_length = len(total)
        uniform_prob = 1/total_length
        return pd.Series(uniform_prob, index=total)
    
    def probability(self, words):
        for word in words:
            if word not in self.mdl:
                return 0
        return self.mdl.iloc[0] ** len(words)
        
    def sample(self, M):
        uniform = [1 / len(self.mdl)] * len(self.mdl)
        return ' '.join(np.random.choice(self.mdl.index, M, p=uniform))
            

        


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        token_dict = {}
        for token in tokens:
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1
        total = len(tokens)
        updated_prob = {key: value / total for key, value in token_dict.items()}
        return pd.Series(updated_prob)
    
    def probability(self, words):
        prob = 1
        for word in words:
            if word not in self.mdl:
                return 0
            else:
                prob *= self.mdl[word]
        return prob
        
    def sample(self, M):
        not_uniformed = []
        for i in np.arange(len(self.mdl.index)):
            not_uniformed.append(self.mdl[i])
        return ' '.join(np.random.choice(self.mdl.index, M, p=not_uniformed))
        


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        to_return = []
        for i in np.arange(len(tokens)-(self.N-1)):
            to_return.append(tuple(tokens[i:i+self.N]))
        return to_return
    def train(self, ngrams):
        nminus = set()
        unique = set(ngrams)
        for ngram in unique:
            nminus.add(ngram[:self.N-1])
        ngramdict = {}
        nminusdict = {}
        for ngram in ngrams:
            ngramdict[ngram] = ngramdict.get(ngram, 0) + 1
            n_1_gram = ngram[:self.N-1]
            nminusdict[n_1_gram] = nminusdict.get(n_1_gram, 0) + 1
        probs = []
        ngram_list = []
        n1gram_list = []
        for ngram in unique:
            n_1_gram = ngram[:self.N-1]
            ngram_list.append(ngram)
            n1gram_list.append(n_1_gram)
            prob = ngramdict[ngram] / nminusdict[n_1_gram]
            probs.append(prob)
        data = {
            'ngram': ngram_list,
            'n1gram': n1gram_list,
            'prob': probs
    }
        return pd.DataFrame(data)
    
    def probability(self, words):
        prob = 1

        for i in np.arange(len(words)):
            if i < self.N - 1:
                if self.N == 2:
                    unigram = words[i]
                    prob *= self.prev_mdl.mdl[unigram] if unigram in self.prev_mdl.mdl else 0
                else:
                    lower_order_ngram = tuple(words[max(0, i - self.N + 2):i + 1])
                    lower_order_model = self.prev_mdl
                    while len(lower_order_ngram) < lower_order_model.N:
                        lower_order_model = lower_order_model.prev_mdl
                    lower_order_prob = lower_order_model.mdl.loc[lower_order_model.mdl['n1gram'] == lower_order_ngram, 'prob']
                    prob *= lower_order_prob.iloc[0] if not lower_order_prob.empty else 0
            else:
                ngram = tuple(words[i - self.N + 1:i + 1])
                ngram_prob = self.mdl.loc[self.mdl['ngram'] == ngram, 'prob']
                prob *= ngram_prob.iloc[0] if not ngram_prob.empty else 0
        return prob


    

    def sample(self, M):
        finished = ['\x02']
        context = ['\x02']

        while len(finished) < M + 1:
            maybe = self.mdl[self.mdl['n1gram'] == tuple(context)]
            
            if maybe.empty:
                finished.append('\x03')
                break
            
            next_token = np.random.choice(maybe['ngram'].apply(lambda x: x[-1]), p=maybe['prob'])
            finished.append(next_token)

            if len(finished) == M + 1 or next_token == '\x03':
                break
        
            if self.N > 2:
                context = context[1:] + [next_token]
            else:
                context = [next_token]

        return ' '.join(finished[0:])
