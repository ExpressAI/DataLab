from typing import Dict, List, Optional, Any
from typing import Callable, Mapping, Iterator
# nltk package for
import nltk
import numpy as np
#sklearn is used for tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

from .aggregating import *




@aggregating(name="get_average_length", contributor="datalab",
               task="Any", description="this function is used to ")
def get_average_length(texts:Iterator) -> int:
    """
    Package: python
    Input:
        texts: Iterator
    Output:
        int
    """
    lengths = []
    for text in texts:
        lengths.append(len(text.split(" ")))
    return np.average(lengths)




@aggregating(name="get_vocabulary", contributor="datalab",
               task="Any", description="this function is used to ")
def get_vocabulary(texts:Iterator) -> Dict:
    """
    Package: python
    Input:
        texts: Iterator
    Output:
        int
    """
    vocab = {}
    for text in texts:
        for w in text.split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1
    vocab_sorted = dict(sorted(vocab.items(), key=lambda item: item[1], reverse = True))
    return vocab_sorted







@aggregating(name="get_tfidf", contributor="scikit-learn",
               task="Any", description="this function is used to ")
def get_tfidf(texts:Iterator) -> int:
    """
    Package: python
    Input:
        texts: Iterator
    Output:
        dict
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names()
    outs = []
    for i in range(len(texts)):
        out = {}
        for j in range(len(words)):
            if tfidf[i, j] > 1e-5:
                out[words[j]] = tfidf[i, j]
        outs.append(out)
    return outs



