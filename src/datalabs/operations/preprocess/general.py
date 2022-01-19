from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
# nltk package for preprocessing
import nltk

from .preprocessing import *




@preprocessing(name="lower", contributor="datalab",
               task="Any", description="this function is used to lowercase a given text")
def lower(text:str) -> str:
    """
    Package: python
    Input:
        text:str
    Output:
        str
    """
    # text = sample['text']
    return text.lower()



@preprocessing(name="tokenize_nltk", contributor="nltk",
               task="Any", description="this function is used to tokenize a text using NLTK")
def tokenize_nltk(text:str) -> List:
    """
    Package: nltk.word_tokenize
    Input:
        text:str
    Output:
        List
    """
    # text = sample['text']
    return nltk.word_tokenize(text)



@preprocessing(name="tokenize_huggingface", contributor="huggingface",
               task="Any", description="this function is used to tokenize a text using huggingface library")
def tokenize_huggingface(text:str) -> List:
    """
    Package: huggingface:tokenizer
    Input:
        text:str
    Output:
        List
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tokenizer = Tokenizer(BPE())
    # text = sample['text']
    output = tokenizer.encode(text)
    return output.tokens





@preprocessing(name="stem", contributor="nltk",
               task="Any", description="this function is used to stem a text using NLTK")
def stem(text:str) -> List:
    """
    Package: nltk.stem
    Input:
        text:str
    Output:
        List
    """
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    # text = sample['text']
    stem_words = [porter.stem(word) for word in text.split(" ")]
    return stem_words








