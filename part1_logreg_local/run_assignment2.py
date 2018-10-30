
# coding: utf-8

# In[1]:


import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from itertools import chain
from operator import itemgetter
import json
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dataset_size', default='verysmall',
                    help='Size of dataset (full, small , verysmall)')
parser.add_argument('--dataset_path', default='../../../ds222/assignment-1/DBPedia.',
                    help='Path to dataset folder without dataset size suffix')
parser.add_argument('--out_prefix', default='final_',
                    help='Prefix to output filename')
args = parser.parse_args()

datestring = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')  # to suffix output files for dates

# In[2]:
start_time = time.time()

# Read train data and label
#dataset_size = "full"  # full, small ,verysmall
#dataset_path = "../../ds222/assignment-1/DBPedia."  # Path to dataset folder
dataset_size = args.dataset_size  # full, small ,verysmall
dataset_path = args.dataset_path  # Path to dataset folder
out_prefix = args.out_prefix  # Prefix to output filename
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.verysmall/verysmall_train.txt") as f:
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.small/small_train.txt") as f:
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_train.txt") as f:
print(f'\nExtracting data from  "{dataset_path}{dataset_size}/{dataset_size}_train.txt" ...')
with open(dataset_path + dataset_size + "/" + dataset_size + "_train.txt") as f:
    raw_data = f.readlines()

trn_label_full = [x.strip().split(' \t')[0].replace('\s*', '').split(',') for x in raw_data[3:]]  # first 3 lines are headers
trn_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_train.txt") as f:
    raw_data = f.readlines()

trn_label_full = [x.strip().split(' \t')[0].replace('\s*','').split(',') for x in raw_data[3:]]  # first 3 lines are headers
trn_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''

# label_full
print(f'\nTotal number of training documents : {len(trn_label_full)}')


# In[3]:


# Create MultiLabelBinarizer object
one_hot = MultiLabelBinarizer()

# One-hot encode data
labels_mat = one_hot.fit_transform(trn_label_full)
print(f'\nShape of one-hot-encoded labels : {labels_mat.shape}')

# View classes
print(f'\nList of classes : {one_hot.classes_}')
print(f'\nTotal number of classes : {len(one_hot.classes_)}')


# In[4]:


# Mapping creation : Labels to indices
dict_label2idx = {}
dict_idx2label = {}
print(f'\nCreating Mappings for labels ...')
for idx,label in enumerate(tqdm(one_hot.classes_)):
    dict_label2idx[label] = idx
    dict_idx2label[idx] = label

dict_label2idx['American_drama_films']  # 'American_drama_films' : 1
dict_idx2label[1]  # 1 : 'American_drama_films'
dict_idx2label


# In[ ]:

print(f'\nCalculating prior probabilities ...')

# Classwise count
class_counts = labels_mat.sum(axis=0)
total_classes = class_counts.sum()

# Class densities 
log_prior_class_ = np.log(class_counts/total_classes)
print(f'\nLog prior probabilities of classes : {log_prior_class_}')

# print(f'\nClass counts for each class : \n{class_counts} \nTotal class count : {class_counts.sum()}')


# In[ ]:

print(f'\nPreprocessing (stemming, stopword removal, tokenization) train data ...')

# String preprocessing (stemming, stopword removal, tokenization)
stemmer = PorterStemmer()  # Porter Stemmer from NLTK
#stemmer = nltk.stem.SnowballStemmer('english')  # Snowball Stemmer from NLTK
#stemmer = nltk.stem.lancaster.LancasterStemmer()  # Lancaster Stemmer from NLTK
stop_words = set(stopwords.words('english'))  # remove stopwords in english

tokenised_data_full = []
# data preprocessing 
for data in tqdm(trn_data_full):
    m = re.findall('".*"@en', data.lower())
    #data_str = re.sub('\\\\\w\d*', ' ', m[0])  # remove tags
    data_str = bytes(m[0], 'utf-8').decode("unicode_escape")
    data_str = re.sub('@en', '', data_str)  # remove end of sentence
    data_str = re.sub('[^a-zA-Z]', ' ', data_str)  # remove all punctuations, special-char and digits
    data_str = re.sub('\s+', ' ', data_str)  # replace multiple spaces
    data_str = data_str.strip()  # replace multiple spaces

    # tokenize, remove stopwords and stem

    # tokenize
    words = word_tokenize(data_str)

    # remove stopwords
    filtered_words = [stemmer.stem(w) for w in words if not w in stop_words]
    #filtered_words = [w for w in words if not w in stop_words]

    tokenised_data_full.append(filtered_words)

# In[ ]:

print(f'\nCreating mappings for vocabulary ...')
# Create a vocabulary of complete training set
vocab_set = set(chain.from_iterable(tokenised_data_full))

# Mapping construction : Vocabulary to indices
dict_vocab2idx = {}
dict_idx2vocab = {}
for idx, word in enumerate(tqdm(vocab_set)):
    dict_vocab2idx[word] = idx
    dict_idx2vocab[idx] = word
    
print(f'\nLength of vocabulary : {len(dict_idx2vocab)}')


# In[ ]:


# Creating a big document for each class to get the counts for each class
# Refer https://web.stanford.edu/~jurafsky/slp3/4.pdf for NB Algo
big_doc_class = []
word_counters = []
total_words_class = []
word_counters_json = {}
total_num_params = 0
print(f'\nBig-doc creation for each class to find word counts ...')
for label_ix in tqdm(range(len(dict_label2idx))):
    doc_idx = np.where(labels_mat[:, label_ix])
    
    # get all documents belonging to current class
    docs_label_ix = itemgetter(*doc_idx[0])(tokenised_data_full)
    
    # create a single big document list from all the documents of this class
    big_doc_class.append(list(chain.from_iterable(docs_label_ix)))
    
    # Create word counter for each class
    word_counters.append(Counter(big_doc_class[label_ix]))
    total_num_params += word_counters[label_ix].__len__() + 1
    word_counters_json[dict_idx2label[label_ix]] = dict(word_counters[label_ix])
    total_words_class.append(np.sum(list(word_counters[label_ix].values())))

print(f'Number of model parameters for {dataset_size} dataset = {total_num_params}.')


ckpt1_time = time.time()
print(f'{(ckpt1_time-start_time)//60} minutes, {int((ckpt1_time-start_time)%60)} seconds for complete training of {dataset_size} dataset.')

# print(len(big_doc_class))
# print(len(word_counters))
# len(big_doc_class[3])
# word_counters[0]
# word_counters_json.keys()


# In[ ]:


# Save JSON file with counts for each class
#word_counters_json_enc = json.dumps(word_counters_json)
with open(out_prefix + dataset_size + "_wordclass_counts.json", "w") as f:
    f.write(json.dumps(word_counters_json))
    print(f'\nSaved "{out_prefix}{dataset_size}_wordclass_counts.json"')

# Store mappings as seperate JSON file
mappings = {}
mappings['dict_idx2label'] = dict_idx2label
mappings['dict_label2idx'] = dict_label2idx
mappings['dict_idx2vocab'] = dict_idx2vocab
mappings['dict_vocab2idx'] = dict_vocab2idx

#mappings_enc = json.dumps(mappings)
with open(out_prefix + dataset_size + "_mappings.json", "w") as f:
    f.write(json.dumps(mappings))
    print(f'\nSaved "{out_prefix}{dataset_size}_mappings.json"')

# In[ ]:


# String preprocessing (stemming, stopword removal, tokenization)

print(f'\nCalculating accuracy on training data ...')
trn_accuracy_count = 0
trn_predicted_label = []
# Training accuracy
for trn_idx, stemmed_words in enumerate(tqdm(tokenised_data_full)):
    # use stemmed words

    # get the counts of words in test data
    test_word_counts = Counter(stemmed_words)

    # reset posterior probabilities for the next test data
    log_post_prob = []
    # posterior probabilities for all classes
    for label_ix in range(len(dict_label2idx)):
        post_prob_class = log_prior_class_[label_ix]

        # sum for all the words
        for word in list(test_word_counts.keys()):
            post_prob_class += test_word_counts[word] * np.log(
                (word_counters[label_ix][word] + 1) / (total_words_class[label_ix] + len(vocab_set)))  # 
        log_post_prob.append(post_prob_class)

    trn_predicted_label.append(dict_idx2label[np.argmax(log_post_prob)])
    #print(f'\n{trn_idx}/{len(trn_data_full)} : Predicted Label : {dict_idx2label[np.argmax(log_post_prob)]} \nTrue Label : {trn_label_full[trn_idx]}')

    if trn_predicted_label[trn_idx] in trn_label_full[trn_idx]:
        trn_accuracy_count += 1

trn_accuracy = trn_accuracy_count / (trn_idx + 1)
print(f'\nTraining Accuracy on {dataset_size} dataset : {trn_accuracy}')

trn_predicted_label_df = pd.DataFrame(trn_predicted_label)
#print(trn_predicted_label_df)

trn_predicted_label_df.to_csv(out_prefix + dataset_size + "_dev_pred_labels.txt", ',', index=False)
print(f'\nSaved predicted labels for training data in "{out_prefix}{dataset_size}_trn_pred_labels.txt"')

ckpt2_time = time.time()
print(f'{(ckpt2_time-ckpt1_time)//60} minutes, {int((ckpt2_time-ckpt1_time)%60)} seconds for complete training inference of {dataset_size} dataset.')

# In[ ]:


print(f'\nExtracting data from development set ...')
# Read dev data and label
with open(dataset_path + dataset_size + "/" + dataset_size + "_devel.txt") as f:
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.small/small_devel.txt") as f:
    raw_data = f.readlines()

dev_label_full = [x.strip().split(' \t')[0].replace('\s*', '').split(',') for x in raw_data[3:]]  # first 3 lines are headers
dev_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_devel.txt") as f:
    raw_data = f.readlines()

dev_label_full = [x.strip().split(' \t')[0].replace('\s*','').split(',') for x in raw_data[3:]]  # first 3 lines are headers
dev_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''

# label_full
print(f'\nTotal number of development documents : {len(dev_label_full)}')


# In[ ]:


# String preprocessing (stemming, stopword removal, tokenization)
print(f'\nPreprocessing (stemming, stopword removal, tokenization) development data ...')
dev_accuracy_count = 0
dev_predicted_label = []
# data preprocessing 
for dev_idx, data in enumerate(tqdm(dev_data_full)):
    m = re.findall('".*"@en', data.lower())
    #data_str = re.sub('\\\\\w\d*', '', m[0])  # remove tags
    data_str = bytes(m[0], 'utf-8').decode("unicode_escape")
    data_str = re.sub('@en', '', data_str)  # remove end of sentence
    data_str = re.sub('[^a-zA-Z]', ' ', data_str)  # remove all punctuations, special-char and digits
    data_str = re.sub('\s+', ' ', data_str)  # replace multiple spaces
    data_str = data_str.strip()  # replace multiple spaces

    # tokenize, remove stopwords and stem

    # tokenize
    words = word_tokenize(data_str)

    # remove stopwords
    filtered_words = [stemmer.stem(w) for w in words if not w in stop_words]
    #filtered_words = [w for w in words if not w in stop_words]

    # get the counts of words in test data
    test_word_counts = Counter(filtered_words)

    # reset posterior probabilities for the next test data
    log_post_prob = []
    # posterior probabilities for all classes
    for label_ix in range(len(dict_label2idx)):
        post_prob_class = log_prior_class_[label_ix]

        # sum for all the words
        for word in list(test_word_counts.keys()):
            post_prob_class += test_word_counts[word] * np.log((word_counters[label_ix][word]+1)/(total_words_class[label_ix]+len(vocab_set)))  # 
        log_post_prob.append(post_prob_class)
        
    dev_predicted_label.append(dict_idx2label[np.argmax(log_post_prob)])
    #print(f'\n{dev_idx}/{len(dev_data_full)} : Predicted Label : {dict_idx2label[np.argmax(log_post_prob)]} \nTrue Label : {dev_label_full[dev_idx]}')
    
    if dev_predicted_label[dev_idx] in dev_label_full[dev_idx]:
        dev_accuracy_count += 1
        
dev_accuracy = dev_accuracy_count/(dev_idx+1)
print(f'\nDevelopment Accuracy on {dataset_size} dataset : {dev_accuracy}')


# In[ ]:

# Save predicted labels on development data
dev_predicted_label_df = pd.DataFrame(dev_predicted_label)
#print(dev_predicted_label_df)

dev_predicted_label_df.to_csv(out_prefix + dataset_size + "_dev_pred_labels.txt", ',', index=False)
print(f'\nSaved predicted labels for development data in "{out_prefix}{dataset_size}_dev_pred_labels.txt"')

ckpt3_time = time.time()
print(f'{(ckpt3_time-ckpt2_time)//60} minutes, {int((ckpt3_time-ckpt2_time)%60)} seconds for complete development inference of {dataset_size} dataset.')

# In[ ]:

print(f'\nExtracting data from test set ...')
# Read test data and label
with open(dataset_path + dataset_size + "/" + dataset_size + "_test.txt") as f:
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.small/small_test.txt") as f:
    raw_data = f.readlines()

tst_label_full = [x.strip().split(' \t')[0].replace('\s*', '').split(',') for x in raw_data[3:]]  # first 3 lines are headers
tst_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''
#with open("/home/sourabhbalgi/ds222/assignment-1/DBPedia.full/full_test.txt") as f:
    raw_data = f.readlines()

tst_label_full = [x.strip().split(' \t')[0].replace('\s*','').split(',') for x in raw_data[3:]]  # first 3 lines are headers
tst_data_full = [x.strip().split(' \t')[1] for x in raw_data[3:]]  # first 3 lines are headers

'''

# label_full
print(f'\nTotal number of testing documents : {len(tst_label_full)}')


# In[ ]:


# String preprocessing (stemming, stopword removal, tokenization)
print(f'\nPreprocessing (stemming, stopword removal, tokenization) test data ...')

tst_accuracy_count = 0
tst_predicted_label = []
# data preprocessing 
for tst_idx, data in enumerate(tqdm(tst_data_full)):
    m = re.findall('".*"@en', data.lower())
    #data_str = re.sub('\\\\\w\d*', '', m[0])  # remove tags
    data_str = bytes(m[0], 'utf-8').decode("unicode_escape")
    data_str = re.sub('@en', '', data_str)  # remove end of sentence
    data_str = re.sub('[^a-zA-Z]', ' ', data_str)  # remove all punctuations, special-char and digits
    data_str = re.sub('\s+', ' ', data_str)  # replace multiple spaces
    data_str = data_str.strip()  # replace multiple spaces

    # tokenize, remove stopwords and stem

    # tokenize
    words = word_tokenize(data_str)

    # remove stopwords
    filtered_words = [stemmer.stem(w) for w in words if not w in stop_words]
    #filtered_words = [w for w in words if not w in stop_words]

    # get the counts of words in test data
    test_word_counts = Counter(filtered_words)

    # reset posterior probabilities for the next test data
    log_post_prob = []
    # posterior probabilities for all classes
    for label_ix in range(len(dict_label2idx)):
        post_prob_class = log_prior_class_[label_ix]

        # sum for all the words
        for word in list(test_word_counts.keys()):
            post_prob_class += test_word_counts[word] * np.log((word_counters[label_ix][word]+1)/(total_words_class[label_ix]+len(vocab_set)))  # 
        log_post_prob.append(post_prob_class)
        
    tst_predicted_label.append(dict_idx2label[np.argmax(log_post_prob)])
    #print(f'\n{tst_idx}/{len(tst_data_full)} : Predicted Label : {dict_idx2label[np.argmax(log_post_prob)]} \nTrue Label : {tst_label_full[tst_idx]}')
    
    if tst_predicted_label[tst_idx] in tst_label_full[tst_idx]:
        tst_accuracy_count += 1
        
tst_accuracy = tst_accuracy_count/(tst_idx+1)
print(f'\nTest Accuracy on {dataset_size} dataset : {tst_accuracy}')

# In[ ]:

# Save predicted test labels
tst_predicted_label_df = pd.DataFrame(tst_predicted_label)
#print(tst_predicted_label_df)

tst_predicted_label_df.to_csv(out_prefix + dataset_size + "_tst_pred_labels.txt", sep=',', index=False)
print(f'\nSaved predicted labels for test data in "{out_prefix}{dataset_size}_tst_pred_labels.txt"')

ckpt4_time = time.time()
print(f'{(ckpt4_time-ckpt3_time)//60} minutes, {int((ckpt4_time-ckpt3_time)%60)} seconds for complete test inference of {dataset_size} dataset.')

# Save prediction accuracies
accuracy_df = pd.DataFrame({'Train': [trn_accuracy], 'Development': [dev_accuracy], 'Test': [tst_accuracy]})
print(accuracy_df.to_string(index=False))

accuracy_df.to_csv(out_prefix + dataset_size + "_" +  datestring + "_accuracy.log", sep=',', index=False)
print(f'\nSaved model accuracies for {dataset_size} in "{out_prefix}{dataset_size}_{datestring}_accuracy.log"')

end_time = time.time()
print(f'{(end_time-start_time)//60} minutes, {int((end_time-start_time)%60)} seconds for complete run of {dataset_size} dataset.')
