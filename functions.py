import json
import numpy as np
import re  
import string
from nltk.corpus import stopwords
from collections import Counter
from collections import defaultdict
from datetime import datetime, timedelta
from email.utils import parsedate_tz
import os  
import bz2
import bz2file
import time
import scipy.sparse as sps
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import comb
from tqdm import tqdm
import collections

results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*|http://[a-zA-Z0-9.?/&=:]*',re.S)

#read stop words
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

def preprocess(s):
    '''
    tokenize the sentence
    '''
    tokens = tokens_re.findall(s)
    return tokens

def to_datetime(datestring):
    '''
    change a string to datatime format
    '''
    time_tuple = parsedate_tz(datestring.strip())
    dt = datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])

def read_bz2(inpath, keyword, word_igonre):
    '''
    read a single bz2 file
    
    input:
    inpath: the location of the path
    keyword: the entity that you want to filter
    word_igonre: ignore the tweets containing one of the words in word_ignore
    
    output:
    terms_stop: the list of tokenized tweets after removing stop words, the element is a tweet after tokenization
    dates_: the published time of the tweets
    whole_tweets: the corresponding original tweets
    durre: the time for reading the file
    '''
    infile = bz2.BZ2File(inpath, 'r')
    terms_stop = []
    dates_ = []
    whole_tweets = []
    begin = time.time()
    num_tweet = 0
    for line in infile:
        try:
            tweet = json.loads(line)
        except ValueError:
            continue
        num_tweet = num_tweet + 1
        if 'text' in tweet:
            text = tweet["text"].lower()
            if keyword.lower() not in text:
                continue
            if 'http' in text or 'https' in text:
                continue
            train = results.sub(r'',text)
            train = re.sub('[^A-Za-z#@0-9]+', ' ', train)
            x = [term.lower() for term in preprocess(train) 
                               if term.lower() not in stop and not term.startswith(('#', '@'))
                              and not term.isdigit()
                              and len(term)>2]
            flag = 0
            for word in word_igonre:
                if word in x:
                    flag =1
                    break
            if flag == 1:
                continue
            if len(x) < 4:
                continue
            terms_stop.append(x)
            whole_tweets.append(tweet["text"])
            dates_.append(to_datetime(tweet['created_at']))
    end_time =  time.time()
    duree = end_time - begin
    infile.close()
    return terms_stop, dates_, whole_tweets, num_tweet, duree

def read_folder(path, keyword,list_igore, num = False):
    '''
    This is similar to read_bz2. Read all files in a folder
    '''
    files= os.listdir(path)
    if num == False or num > len(files):
        num = len(files)
    terms_stop = []
    dates = []
    whole_tweets = []
    duree = 0
    num_tweet = 0

    for fname in tqdm(range(num)):
        fname = path + "/" + files[fname] 
        terms_stop_, dates_, whole_tweets_, num_tweet_, duree_ = read_bz2(fname, keyword, list_igore)
        terms_stop.extend(terms_stop_)
        dates.extend(dates_)
        whole_tweets.extend(whole_tweets_)
        duree = duree + duree_
        num_tweet = num_tweet + num_tweet_
    return terms_stop, dates, whole_tweets, num_tweet, duree

def find_burst(num, accumulated_num, terms_stop, threshold, first_n = False):
    '''
    Given the numbers of tweets for every period, find the burst
    
    input: 
    num:the numbers of tweets for every period by funtion split_by_time
    accumulated_num: accumulated numbers until this period by funtion split_by_time
    terms_stop: the list of tokenized tweets from read_bz2 function
    threshold: only detect burst beyond this threshold, it is multiplied by the average value
    
    output:
    ind_burst: a dictionnary indicating the burst burst and the burst end
    list_count: a list indicating all words in a given burst
    '''
    ind_burst = search_maxima(num, threshold, first_n)
    begin_burst = ind_burst.keys()
    end_burst = ind_burst.values()
    list_count = []
    for i in range(len(begin_burst)):
        count_all = Counter()
        for j in range(accumulated_num[begin_burst[i]], accumulated_num[end_burst[i]+1]):
            count_all.update(terms_stop[j])
        list_count.append(count_all)
        
    return ind_burst, list_count

def search_maxima(value_list, threshold,  first_n = False):
    '''
    find local maxima which is bigger than mean value * threshold.
    the neighbors of a maximum is also considred to be in the burst if their difference is very small
    
    input:
    value_list: the numbers for a period
    threshold: the burst below mean value * threshold is ignored
    first_n: Only find the first N maxima
    
    output:
    bursts: a dictionnary indicating the burst burst and the burst end
    '''
    
    threshold = np.mean(value_list) * threshold
    len_ = len(value_list)
    local_max = []  
    
    for i in range(1, len_-1):
        if value_list[i] > threshold and value_list[i] > value_list[i-1] and value_list[i] > value_list[i+1]:
            local_max.append(i)
        
    bursts = {}
    diff = max(value_list) - min(value_list)
    
    for pic in local_max:
        for j in range(pic + 1, len_):
            if value_list[j] < threshold or abs(value_list[j] - value_list[pic]) > 0.01 * diff :
                break
        tmp = 0
        for k in range(pic-1,0,-1):
            if value_list[k] < threshold or abs(value_list[k] - value_list[pic]) > 0.01 * diff :
                tmp = k + 1
                break
        bursts[tmp] = j-1
        
    if 0 not in bursts.keys():
        if value_list[0] > threshold and value_list[0] > value_list[1]:
            bursts[0] = 0
    if len_-1 not in bursts.values():
        if value_list[len_-1] > threshold and value_list[len_-1] > value_list[len_-2]:
            bursts[len_-1] = len_-1
    
    if first_n:
        ind_n = []
        for i in bursts.keys():
            ind_n.append(i)
        value_max = [value_list[i] for i in ind_n]
        first_n_ind = np.array(value_max).argsort()[-first_n:] 
        burst_new = {}
        for i in first_n_ind:
            burst_new[ind_n[i]] = bursts[ind_n[i]]
        return burst_new
    return bursts

def plot_burst(ind_burst, figsize, frequency, threshold):
    '''
    Plot a graph showing the burst
    
    input:
    ind_burst: a dictionnary indicating the burst burst and the burst end
    figsize: the size of figure
    frequency: the numbers of tweets for every period
    threshold: the burst below mean value * threshold is ignored
    
    output:
    figure
    '''
    ind_burst_plot = collections.OrderedDict(sorted(ind_burst.items()))            

    fig, ax = plt.subplots(figsize = figsize)
    plt.plot(frequency)
    plt.plot([np.mean(frequency) * threshold]*len(frequency))
    c = -1
    for i in ind_burst_plot.keys():
        c = c + 1
        max_tmp = frequency[i]
        for j in range(i, len(frequency)):
            if j<= ind_burst_plot[i]:
                if max_tmp<frequency[j]:
                    max_tmp = frequency[j]
                if c % 2 == 0 :
                    plt.scatter(j, frequency[j],c = 'red')
                else: 
                    plt.scatter(j, frequency[j],c = 'green')
        ax.annotate(str(c+1),(i, max_tmp*1.02)) 

    plt.legend(('change trend', 'thresholod', 'burst'))
    plt.xlabel('period sequence')
    plt.ylabel('number of tweets')
    plt.title('Find burst according to tweets frequency.')


def split_by_time(dates_, periode = 3600):
    '''
    split our data by a given periode
    
    input:
    dates_: the published time of tweets
    period: the length of period, in seconds
    
    output:
    accumulated_num: accumulated numbers until this period
    num:the numbers of tweets for every period
    '''
    accumulated_num = [0]
    num = []
    begin = 0
    for i in range(1,len(dates_)):
        dur_tmp = dates_[i] - dates_[begin]
        if dur_tmp.total_seconds() > periode:
            accumulated_num.append(i)
            num.append(i-begin)
            begin = i
    return accumulated_num, num
 

def build_graph(num_words, list_node, burst_i, ind_burst, accumulated_num, terms_stop, thresh):
    '''
    Need to specify the threshold which delete all values smaller than it.
    The idea is to build the TermDocumentMatrix M then get the co-occurence matrix by M.T.dot(M)
    
    input: 
    num_words: the length of the matrix
    list_node: all considered words to build the TermDocumentMatrix
    burst_i: the ith burst in the dictionnary
    ind_burst: a dictionnary indicating the burst burst and the burst end by funtion find_burst
    accumulated_num: accumulated numbers until this period given by function split_by_time
    terms_stop: the list of tokenized tweets from read_bz2 function
    thresh: ignore the value in comatrix which is smaller than thresh * mean
    
    output:
    sps_acc: TermDocumentMatrix
    comatrix: square matrix
    '''
    burst_start = ind_burst.keys()[burst_i]
    burst_end = ind_burst[burst_start]
    burst_start= accumulated_num[burst_start]
    burst_end = accumulated_num[burst_end+1]
    num_tweet = burst_end - burst_start

    #build word frequency matrix
    rows, cols = num_tweet, num_words
    sps_acc = sps.lil_matrix((rows, cols)) # empty sparse matrix
    for i in range(burst_start, burst_end):
        for j in range(cols):
            if list_node[j] in terms_stop[i]:
                sps_acc[i - burst_start,j] = 1

    comatrix = sps_acc.T.dot(sps_acc)#.todense()
    mean = np.mean(comatrix)
    comatrix[comatrix<thresh*mean] = 0
    comatrix[range(cols),range(cols)] = 0
    
    return sps_acc, comatrix

def find_original(burst_i, ind_burst, clique_word, whole_tweets, dates_, accumulated_num, terms_stop):
    '''
    find the original tweets according to the clique words
    
    input:
    burst_i: the ith burst in the dictionnary
    ind_burst: a dictionnary indicating the burst burst and the burst end by funtion find_burst
    clique_word: the words in the clique
    ...
    
    output:
    max_count: how many words are matched in the clique and the original tweets
    whole_tweets[index]: the original tweet
    dates_[index]: the published time
    '''
    max_count = 0
    max_ind = 0
    for i in range(accumulated_num[list(ind_burst.keys())[burst_i]],  accumulated_num[list(ind_burst.values())[burst_i]+1]):
        count_appear = 0
        for j in clique_word:
            if j in terms_stop[i]:
                count_appear = count_appear + 1
            if count_appear == len(clique_word):
                max_ind = i
                break
            if count_appear > max_count:
                max_count = count_appear
                max_ind = i
    index = max_ind
    
    return  max_count, whole_tweets[index], dates_[index]
    
def find_event(keyword, ignore, ind_burst, list_word, accumulated_num, terms_stop, whole_tweets, dates, num_words= 150, thre = 0, gamma = 0.45, numClique = 12, minClique = 6):
    '''
    Detect events for all bursts.
    If a clique is overlap with previous one, it is ignored.
    
    input:
    keyword: the root word for quasi clique
    ignore: ignore the word for a better quasi clique
    list_word: all existing words, it is prepared for num_words
    num_words: how many words are candidates for quasi clique
    thre: value for function build graph
    gamma: key parameter, how much a new edge is close to the max edge weight
    numClique: max number of nodes in the clique
    minClique: how many nodes the resulting clique should contain. This is to avoid that a clique only contains 2 or 3 words.
    
    output:
    list_clique: all resulting cliques
    list_date: corresponding published time
    list_whole: corresponding original tweets
    '''
    list_date = []
    list_whole = []
    list_clique = []
    for burst_i in range(len(ind_burst)):
        list_node = list_word[burst_i].most_common(num_words)
        list_node = dict(list_node).keys()
        count_matrix, comatrix = build_graph(len(list_node), list_node, burst_i, ind_burst, accumulated_num, terms_stop, thre)
        df = pd.DataFrame(data = comatrix.todense(), columns = list_node, index = list_node)
        if ignore in list_node:
            df = df.drop([ignore]).drop(ignore,axis = 1)
        G = df.values
        #find the root word
        for i in range(len(list(df.columns))):
            if list(df.columns)[i] == keyword.lower():
                break
        list_i = findQuasiClique(G, i, gamma, numClique)
        clique_word = []
        for i in list_i: 
            clique_word.append(list(df.columns)[i])
        if len(clique_word) > minClique:
            if not isOverlop(list_clique, clique_word):
                count, whole, date = find_original(burst_i, ind_burst, clique_word, whole_tweets, dates, accumulated_num, terms_stop)
                if count < 4:
                    continue
                list_date.append(date)
                list_whole.append(whole)
                list_clique.append(clique_word)
    return list_date, list_whole, list_clique

def isOverlop(list_clique, clique_word):
    '''
    To detect if a new resulting clique has a fraction of same words with previous one
    
    input:
    list_clique:  all resulting cliques until now
    clique_word: new cliques
    
    output:
    overlap or not
    '''
    for clique in list_clique:
        overlap = 0
        for word in clique_word:
            if word in clique:
                overlap +=1
        if overlap > 0.6 * len(clique_word):
            return True
    
    return False

def findQuasiClique(G, v, gamma, s):
    '''
    Given an undirected graph, find a quasiclique.
    
    inputs:
        G: the undirected graph, the type is np.ndata
        V: the vertices set.
        gamma: the parameter of weighted gamma-quasi-clique.
        s: the maximum number of vertice in the clique.
    outputs:
        S: the set of nodes.
    '''
    V = list(np.arange(len(G))) 
    S = []
    x, y = np.meshgrid(v, np.arange(len(G)))
    u = np.argmax(G[x,y])
    loop = 7
    while  (isolate(sorted(G[u,:], reverse=True))):
        for i in range(len(G)):
            G[i,u] = 0
            G[u,i] = 0
        x, y = np.meshgrid(v, np.arange(len(G)))
        u = np.argmax(G[x,y])
        loop = loop -1
        if loop == 0:
            break
    if (G[v,u]) <= 0:
        return S
    else:
        S.append(v)
        S.append(u)
        while True:
            if len(S) >= s:
                return S
            else:
                '''find the neighbors for set S'''
                res = list(set(V) - set(S))
                if len(res)>0:
                    x,y = np.meshgrid(S, res)
                    w = G[x,y]
                    w = w.reshape(len(S), len(res))
                    flag = w > 0
                    flag = np.sum(flag, axis=0)
                    flag = flag > 0
                    index = np.int_(flag*np.arange(len(res)))
                    neighbor = np.array(res)[index]
                    if len(neighbor):
                        qs = []
                        for i in neighbor:
                            qs.append(qG(G,S+[i]))
                        index = np.argmax(qs)
                        qG_max = np.array(qs)[index]
                        if qG_max < gamma:
                            return S
                        else:
                            S = S + [neighbor[index]]
                else:
                    return S

def isolate(l):
    '''
    If a node is only connected to few nodes, it is ignored.
    If a node's weight is much bigger than the next node, it is ignored.
    '''
    if l[0] > l[1] * 2:
        return 1
    l = np.asarray(l)
    if len(l[l> 0]) < 0.1*len(l):
        return 1
    
def qG(G,S):
    '''
    To calculate a ratio used for verifying if the subgraph S of G is a weighted gamma quasi-clique.

    inputs:
        G: a graph, the type is np.ndata.
        S: a subgraph of G, the type is np.ndata.

    output:
        return the ratio if S is not empty, otherwise return 0.
    '''
    if len(S):
        x, y = np.meshgrid(S,S)
        total_w = np.sum(G[x,y])/2
        max_w = np.max(G[x,y])
        return total_w/(max_w*comb(len(S),2,exact=False))
    else:
        return 0