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
nltk.download('stopwords')


#vocab size method 
def get_vocab_size(dataset):
    captions = [d["caption"] for d in dataset]
    cachedStopWords = stopwords.words("english")
    words_without_stop_words = [
        [word for word in cap.split()
            if word not in cachedStopWords]
        for cap in captions]
    length = 0
    for cap in words_without_stop_words:
      length = length + len(cap)
    return length

def diversity(hypo, n_lines=None):
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    for line in open(hypo, encoding='utf-8'):
        i += 1
        words = line.strip('\n').split()
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
        if pos[1] == 'JJ':
            count+=1
        total +=1 
    return count/total


    
# figurative lang score -- evaluate on given caption
# classifier separately trained


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

