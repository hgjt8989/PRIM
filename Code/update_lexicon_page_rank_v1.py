## instruction of running program
# python update_lexicon_page_rank_v1.py num_words n_files n_comatrix file_name keyword
#   e.g.: python update_lexicon_page_rank_v1.py 100 2 1000 - apple


import json
import numpy as np
from pathlib import Path
import re  
import nltk
from nltk.corpus import stopwords
from collections import Counter
import datetime as datetimeLib
from datetime import datetime, timedelta
from email.utils import parsedate_tz
import os  
import bz2
import scipy.sparse as sps
from scipy.special import comb
import sys
import string


# In[38]:

def generate_files(n_files=7, fname='-'):
    '''
    Generate a list of the names of specified data files

    input:
        n_files: the number of files
        fname: the name of file, it can be a part of the file's name.
              
    output:
        refiles: a list of data files' names.     
    '''
   
    files = []
    for i in os.listdir():
        if '.bz2' in i:
            files.append(i)
    files = sorted(files)
    refiles = []
    if fname != '-':
        for i in files:
            if fname in i:
                refiles.append(i)
    else:
        if len(files) > n_files:
            i = len(files)
            while i > len(files)-n_files:
                refiles.append(files[i-1])
                i = i-1
        else:
            refiles = files
    return refiles

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s):
    tokens = tokenize(s)
    return tokens

def to_datetime(datestring):
    time_tuple = parsedate_tz(datestring.strip())
    dt = datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])

def read_tweets(inpath, keyword):
    '''
    Read tweets file.

    input:
        inpath: the path of data file.

    output:
        terms_stop: a list of tokenized tweets, each element in terms_stop corresponds to a tweet.
        num_tweet: the number of tweets in the data files.     
    '''
    infile = bz2.BZ2File(inpath, 'r')
    terms_stop = []
    num_tweet = 0
    for line in infile:
        try:
            tweet = json.loads(line.decode('utf-8'))
        except ValueError:
            continue
        
        if 'text' in tweet:
            text = tweet["text"].lower()
            train = results.sub(r'',text)
            train = re.sub('[^A-Za-z#@0-9]+', ' ', train)
            if keyword.lower() not in text:
                continue
            if 'http' in text or 'https' in text:
                continue
            terms_stop.append([term.lower() for term in preprocess(train) 
                               if term.lower() not in stop and not term.startswith(('#', '@'))
                              and not term.isdigit()
                              and len(term)>2])
            num_tweet = num_tweet + 1        
        #########################################
        ############## for debug ################
        # if num_tweet > 1000:
        #     break
        #########################################
          
    infile.close()
    return terms_stop,num_tweet

def page_rank(M, P=0.85, itr=100):
    '''
    Do the page rank algorithm for M.

    inputs:
        M: the co-occurence matrix, the type should be np.narray.
        P: the probability of randomly jumping according to normalized M
        itr: the maximum number of itrations.
        
    output:
        prob1: the scores of page-rank.
    '''
    epsilon = 10**-8
    N = M.shape[0]
    M = np.dot(M,np.diag(1./np.sum(M, axis=0)))
    prob = np.ones(N)
    prob1 = np.ones(N)
    for i in range(itr):
        prob1 = np.dot(P*M + 1./N*(1-P)*np.ones((N,N)),prob)
        if(np.linalg.norm(prob1-prob)) < epsilon:
            return prob1
        prob = prob1
    return prob1


results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*|http://[a-zA-Z0-9.?/&=:]*',re.S)

# read stop words
f = open('stopwords.txt','r')
stop = []
for line in f:
    stop.append(line[:-1])
f.close()

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

# read lexicon from file
path = Path("key.txt")
lexicon = []
if path.is_file():
    with open(path.name,'r') as f: 
        for line in f:
            lexicon.append(str.strip(line))


#############################################
'''
inputs from console: 
    num_words: the number of candidates
    n_files: the number of data files
    n_comatrix: the number of the most common words
    file_name: the name of specified data file(s)
'''

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_words = int(sys.argv[1])
    else:
        num_words = 100
    if len(sys.argv) > 2:
        n_files = int(sys.argv[2])
    else:
        n_files = 1
    if len(sys.argv) > 3:
        n_comatrix = int(sys.argv[3])
    else:
        n_comatrix = 1000
    if len(sys.argv) > 4:
        file_name = sys.argv[4]
    else:
        file_name = '-'
    if len(sys.argv) > 5:
        keyword = sys.argv[5]
    else:
        keyword = ''


################## test for ipynb##############
# num_words = 100
# n_files = 1
# n_comatrix = 500
# file_name = '-'
# keyword = ''
################################################
# read data file
fs = generate_files(n_files, file_name);
terms_stop = []
num_tweet = 0
print(fs)
for fname in fs:
    print('starting reading ' + fname)
    terms_stop_,num_tweet_ = read_tweets(fname,keyword)
    terms_stop.extend(terms_stop_)
    num_tweet = num_tweet + num_tweet_
    print('finishing reading ' + fname)

# construct co-occurrence matrix
count_all = Counter()
for j in range(len(terms_stop)):
    count_all.update(terms_stop[j])
print('the total number of nodes: ', len(count_all))
if len(count_all) < n_comatrix:
    n_comatrix = len(count_all)
list_node = count_all.most_common(n_comatrix)
list_node = list(dict(list_node).keys())
print('the number of nodes we take: ', len(list_node))
rows, cols = num_tweet, len(list_node)
sps_acc = sps.lil_matrix((rows, cols)) # empty sparse matrix
for i in range(rows):
    for j in range(cols):
        if list_node[j] in terms_stop[i]:
            sps_acc[i,j] = 1

comatrix =  sps_acc.T.dot(sps_acc)  #.todense()

# find the index for iterms in lexicon
lexicon_index = []

for i in lexicon:
    if i in list_node:
        lexicon_index.append(list_node.index(i))
        
# do the page rank of comatrix        
comatrix = np.asarray(comatrix.todense())
print('co-occurrence matrix size: ',comatrix.shape)
comatrix[range(cols),range(cols)] = 10**-5 # avoid dividing by zero.
prob1 = page_rank(comatrix)
comatrix = comatrix*prob1
ranks = np.argsort(np.sum(comatrix[:,lexicon_index],axis=1)/(np.sum(comatrix,axis=1)))[::-1]

# write the candidates (for expanding the lexicon) to the file.
today = datetimeLib.date.today()
path = 'update_key'+str(today)+'.txt'
with open(path, 'w') as f:
    j = 0
    for i in ranks:
        if j < num_words:
            if list_node[i] not in lexicon:
                f.writelines(list_node[i]+'\n')
                j = j+1


