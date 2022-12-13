from os import path as os_path
import sqlite3
import math
import numpy as np
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import nltk.tokenize as nt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import string
from collections import Counter
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


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

def tf_idf(caption, corpus):
    """
        caption = single caption to be tested
        data = entire dataset of captions
    """

    tr_idf_model  = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(corpus)
    tf_idf_array = tf_idf_vector.toarray()

    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
   
    cap_tf_idf = []

    cachedStopWords = stopwords.words("english")
    words_without_stop_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in caption.split() if word.lower() not in cachedStopWords]
    for word in words_without_stop_words:
        if word in list(df_tf_idf.columns):
            cap_tf_idf.append(df_tf_idf[word].sum()/len(df_tf_idf))
    if len(cap_tf_idf) == 0:
        return 0
    return sum(cap_tf_idf)

def creative_total(caption, corpus, weights):
    a_w = weights[0]
    f_w = weights[1]
    d_w = weights[2]

    aggregate = a_w + f_w  + d_w
    a_w /= aggregate
    f_w /= aggregate
    d_w /= aggregate

    score = (d_w * get_vocab_size(caption)) +  (a_w*adjectives_per_caption(caption)) + (f_w * tf_idf(caption,corpus))

    return score 


if __name__ == '__main__':
    corpus_fine_tune2 = ["A cart full of items is displayed on a shelf.",
"A picture of a bunch of food in a magazine.",
"A table filled with a variety of food and drinks.",
"A large plate of food is sitting on a shelf.",
"A bunch of people are eating a bunch of food.",
"A book with a pen and marker on top of it.",
"A woman sitting in front of a mirror.",
"A picture of a lamp with a lamp on it.",
"A group of people standing in front of a microscope.",
"A large open container filled with a lot of different kinds of food.",
"A picture of a Christmas tree is displayed on a shelf.",
"An overview of some of the papers in the library.",
"A woman holding a pelican in the palm of her hand.",
"A large cell with a bunch of papers on it.",
"A picture of a tennis player playing a round of tennis.",
"A bunch of books on a desk next to a computer.",
"A montage of the history of the show.",
"A view of a large window showing some of the items in the window.",
"A man and a woman sitting next to each other.",
"A man standing in a field with a white shirt and a white hat."]

    corpus_fine_tune1 = ["A vase sitting on top of a table.",
"A group of people sitting at a dining table.",
"A glass of wine and a plate of food on a table.",
"A plate of pancakes and a cup of coffee on a table.",
"A group of people sitting around a table eating food.",
"A brown teddy bear sitting on top of a laptop.",
"A woman sitting on a chair with a tennis racket.",
"A group of people standing on top of a building.",
"A group of men standing in a room holding Wii controllers.",
"A blender filled with a bunch of fruit on top of a counter.",
"A teddy bear sitting on top of a Christmas tree.",
"Two computers are sitting on a desk.",
"A woman is playing tennis on a tennis court.",
"a desk with two computers and a monitor on it",
"A man swinging a tennis racquet on top of a tennis court.",
"Two laptops sitting on a desk next to each other.",
"A man holding a microphone while holding a skateboard.",
"A living room filled with furniture and a window.",
"A man leaning on a bicycle next to a boat.",
"A man on a beach flying a kite."]

    corpus_reference = ["A vase sitting on top of a counter.",
"A group of people are sitting at a table with bowls and wine glasses.",
"three filled wine glasses and a restaurant menu sitting on a table",
"A pancake has a topping of mixed fresh fruit. ",
"A group of people sitting at a table eating food.",
"A teddy bear sitting on a laptop computer.",
"The woman is looking at the tennis racket ",
"Some people on a roof who are flying some kites.",
"One man is taking a photo and two others are holding Wii controllers.",
"The blender pitcher is filled with pieces of fruit.",
"A stuffed Christmas bear is laying on the floor next to its arm that has fallen off. ",
"A laptop sits on a table next to desktop computers. ",
"there is a woman that is playing tennis wearing a bathing suit",
"A desk topped with computer monitors and laptops.",
"A man hitting a tennis ball on the court",
"Two laptop computers sitting on top of a desk.",
"A MAN IS ON STAGE SINGING WHILE SOMEONE HANDS HIM A BEAR",
"A couple of windows in a small room.",
"A man standing next to a bicycle on a pier.",
"Kite boarder at the beach during low tide."]


    weights = [(0.749),(0.877),(0.607)] #max vote weights 
   
    dict_scores = []
    #comment
    for i in range(len(corpus_fine_tune1)):
        scores = {}
        caption1 = corpus_fine_tune2[i]
        caption2 = corpus_fine_tune1[i]
        caption3 = corpus_reference[i]
        scores["caption_finetune2"] =  creative_total(caption1, corpus_fine_tune2, weights)
        scores["caption_finetune1"] =  creative_total(caption2, corpus_fine_tune1, weights)
        scores["caption3"] =  creative_total(caption3, corpus_reference, weights)
        
        dict_scores.append(scores)
    
    all_scores = json.dumps(dict_scores)

    with open("all_scores_lin.json", "w") as outfile:
        json.dump(all_scores, outfile)
  