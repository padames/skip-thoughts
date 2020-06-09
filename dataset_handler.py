# Dataset handler for binary classification tasks (MR, CR, SUBJ, MQPA)

import numpy as np
from numpy.random import RandomState
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from os.path import join, isfile
from os import listdir



def load_data(encoder, name, loc='./data/', seed=1234):
    """
    Load one of MR, CR, SUBJ or MPQA, ACLIMBD
    """
    z = {}
    if name == 'MR':
        pos, neg = load_rt(loc=loc)
    elif name == 'SUBJ':
        pos, neg = load_subj(loc=loc)
    elif name == 'CR':
        pos, neg = load_cr(loc=loc)
    elif name == 'MPQA':
        pos, neg = load_mpqa(loc=loc)
    elif name == 'ACLIMBD':
        pos, neg = load_aclimbd(loc=loc)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels
    print( 'Computing skip-thought vectors...')
    features = encoder.encode(text, verbose=False, use_eos=True)
    return z, features


def load_rt(loc='./data/'):
    """
    Load the MR dataset
    """
    pos, neg = [], []
    with open(join(loc, 'rt-polarity.pos'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(join(loc, 'rt-polarity.neg'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg

def load_aclimbd(loc='./data/'):
    """
    Load the acl imbd dataset
    """
    pos, neg = [], []
    pos_path = join(loc, 'pos')
    onlyfiles = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
    for f_name in onlyfiles:
        with open(join(pos_path,f_name), 'rb') as f:
            for line in f:
                l = line.decode('latin-1').strip()
                sentences = sent_tokenize(l)
                for s in sentences:
                    tokens = word_tokenize(s)
                    tokens = [w.lower() for w in tokens]
                    table = str.maketrans(',', ' ', '!?@#%&*"\'')
                    words = [w.translate(table) for w in tokens]
                    # remove remaining tokens that are not alphabetic
                    sentence = " ".join(words)
                    sent = sentence.split()
                    pos.append(" ".join(sent))
    
    neg_path = join(loc, 'neg')
    onlyfiles = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]    
    for f_name in onlyfiles:
        with open(join(neg_path, f_name), 'rb') as f:
            for line in f:
                l = line.decode('latin-1').strip()
                sentences = sent_tokenize(l)
                for s in sentences:
                    tokens = word_tokenize(s)
                    tokens = [w.lower() for w in tokens]
                    table = str.maketrans(',', ' ', '!?@#%&*"\'')
                    words = [w.translate(table) for w in tokens]
                    # remove remaining tokens that are not alphabetic
                    sentence = " ".join(words)
                    sent = sentence.split()
                    neg.append(" ".join(sent))                

    return pos, neg


def load_subj(loc='./data/'):
    """
    Load the SUBJ dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'plot.tok.gt9.5000'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'quote.tok.gt9.5000'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_cr(loc='./data/'):
    """
    Load the CR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'custrev.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'custrev.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def load_mpqa(loc='./data/'):
    """
    Load the MPQA dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'mpqa.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'mpqa.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
    """
    Shuffle the data
    """
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)    




