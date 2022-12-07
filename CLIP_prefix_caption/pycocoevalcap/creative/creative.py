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
        for caption in dataset:
            i += 1
            words = caption.strip('\n').split()
            for n in range(4):
                for idx in range(len(words)-n):
                    ngram = ' '.join(words[idx:idx+n+1])
                    counter[n][ngram] += 1
            if i == n_lines:
                break

            for n in range(4):
                total = sum(counter[n].values())+ 1e-9
            for v in counter[n].values():
                etp_score[n] += - 1.0 * v /total * (np.log(v) - np.log(total))

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

class Creative:

    def __init__(self, test=None, refs=None, n=4):
        # set cider to sum over 1 to 4-grams
        self._n = n
    
    def compute_score(self, gts, res):
        """
        Main function to compute creative score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: creative (float) : computed CREAT score for the corpus 
        """



        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        creative_scorer = CreativeScorer(n=self._n)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            creative_scorer += (hypo[0], ref)

        (score, scores) = creative_scorer.compute_score()

        return score, scores

    def method(self):
        return "CREAT"

