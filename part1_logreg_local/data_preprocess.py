#!/usr/bin/env python3

import re
from tqdm import tqdm
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import h5py
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = PorterStemmer()


def preprocess(mode, f):
    countDocuments = 0
    vocabularyCount = 0
    label = {}
    text = {}
        
    if mode == 'train':
        vocab = defaultdict(int)
        vocabularySet = set()

    for line in tqdm(f.readlines()[3:]):
        labels = line.split(' \t')[0]
        line = line.split(' \t')[1].split('"')[1]
        m = re.findall('".*"@en', line.lower())
        # data_str = re.sub('\\\\\w\d*', ' ', m[0])  # remove tags
        data_str = bytes(m[0], 'utf-8').decode("unicode_escape")
        data_str = re.sub('@en', '', data_str)  # remove end of sentence
        data_str = re.sub('[^a-zA-Z]', ' ', data_str)  # remove all punctuations, special-char and digits
        data_str = re.sub('\s+', ' ', data_str)  # replace multiple spaces
        data_str = data_str.strip()  # replace multiple spaces

        # tokenize, remove stopwords and stem

        # tokenize
        words = word_tokenize(data_str)

        text[countDocuments] = line
        
        if mode == 'train':
            for word in words: 
                # word = lemmatizer.stem(word)
                if word not in vocabularySet and word not in stop_words:
                    vocabularySet.add(word)
                    #vocabulary.append(word)
                    vocabularyCount = vocabularyCount + 1
                    vocab[word] = 1
                elif word in vocabularySet:
                    vocab[word] = vocab[word]+1
                        
        j = []
        for lab in labels.split(','):
            j.append(lab)
        label[countDocuments] = j
        countDocuments += 1
        
    if mode == 'train':
        return text, vocab, countDocuments, label
    else:
        return text, countDocuments, label


print('loading and preprocessing train data... \n')

f = open('/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_train.txt')
mode = 'train'
text_train, vocab, countDocuments, label = preprocess(mode, f)

threshold=100
vocab_mod = {k: v for k, v in vocab.items() if v >= threshold}
d = {k: ind for ind, k in enumerate(vocab_mod.keys())}

mlb = MultiLabelBinarizer()

print('loading and preprocessing test data ... \n')


f = open('/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_test.txt')
mode = 'test'
text_test, countDocuments_test, label_test = preprocess(mode,f)

print('creating train and test arrays ... \n')

train = np.zeros((countDocuments, len(vocab_mod)), dtype=np.float32)
test = np.zeros((countDocuments_test, len(vocab_mod)), dtype=np.float32)

train_lab = mlb.fit_transform(label.values()).astype(float)
test_lab = mlb.transform(label_test.values()).astype(float)

for k, line in text_train.items():
    words = line.split()
    for w in words:
        if w in vocab_mod.keys():
            train[k, d[w]] = 1
    train_lab[k, :] = train_lab[k, :]/np.sum(train_lab[k, :])

for k, line in text_test.items():
    words = line.split()
    for w in words:
        if w in vocab_mod.keys():
            test[k, d[w]] = 1
    test_lab[k, :] = test_lab[k, :]/np.sum(test_lab[k, :])
    
'''
saving the train and test arrays in HDFS
'''

np.save("train_l", train_lab)
np.save("test_l", test_lab)
h5f1 = h5py.File('train.h5', 'w')
h5f1.create_dataset('d1', data=train)
h5f2 = h5py.File('test.h5', 'w')
h5f2.create_dataset('d2', data=test)
h5f1.close()
h5f2.close()
