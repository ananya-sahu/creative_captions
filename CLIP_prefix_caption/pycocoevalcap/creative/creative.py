from os import path as os_path
import sqlite3
import ijson
import math
import numpy as np
from .creative_scorer import CreativeScorer
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import nltk.tokenize as nt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import string
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def get_vocab_size(caption):
    """
    Inputs: A generated caption
    Returns: Normalized length of the caption
    """
    cachedStopWords = stopwords.words("english")
    words_without_stop_words = [word for word in caption.split()
            if word not in cachedStopWords]
    unique = Counter(words_without_stop_words).keys()
    return len(unique) / len(caption)

def diversity(data, n_lines=None):
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    with open(data, 'r') as f:
        dataset = json.load(f)
        corpus = [d["caption"] for d in dataset]
        for caption in corpus:
            for n in range(4):
                for idx in range(len(caption)-n):
                    ngram = ' '.join(caption[idx:idx+n+1])
                    counter[n][ngram] += 1
        for n in range(4):
            total = sum(counter[n].values()) +1e-10
            for v in counter[n].values():
                etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
    return etp_score


#do entity extraction and get number of adjectives per caption 
def adjectives_per_caption(caption):
    ss=nt.sent_tokenize(caption)
    tokenized_sent=[nt.word_tokenize(sent) for sent in ss]
    pos_sentences=[nltk.pos_tag(sent) for sent in tokenized_sent]
    pos = pos_sentences[0]
    count = 0
    total = 0
    for tup in pos:
        if tup[1] == 'JJ':
            count+=1
        total +=1 
    return count/total

def tf_idf(caption, data):
    """
        caption = single caption to be tested
        data = entire dataset of captions
    """
    with open(data, 'r') as f:
        dataset = json.load(f)
    corpus = [d["caption"] for d in dataset]

    tr_idf_model  = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(corpus)
    tf_idf_array = tf_idf_vector.toarray()

    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
    cap_tf_idf = 0

    cachedStopWords = stopwords.words("english")
    words_without_stop_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in caption.split() if word.lower() not in cachedStopWords]
    for word in words_without_stop_words:
        cap_tf_idf  += df_tf_idf[word]
    return cap_tf_idf.sum() / len(cap_tf_idf)

def creative_total(caption, data, weights):
    a_w = weights[0]
    f_w = weights[1]
    d_w = weights[2]

    aggregate = a_w + f_w  + d_w
    a_w /= aggregate
    f_w /= aggregate
    d_w /= aggregate

    score = (d_w * get_vocab_size(caption)) +  (a_w*adjectives_per_caption(caption)) + (f_w * tf_idf(caption,data))

    return score 


def main():
    caption_file = '/content/captions1 (1).json'
    weights = [(17/20),(1/20),(2/20)] #max vote weights 
    diversity_score = diversity(captions_data, n_lines=None)
    with open(caption_file) as json_file:
        captions_data = json.load(json_file)
    scores = []
    for c_dict in captions_data:
        caption = c_dict["caption"]
        score = creative_total(caption, caption_file, weights,diversity_score[1]) #normalize 
        scores.append(score)
    
    print(sum(scores)/len(scores))
    print(diversity_score)

