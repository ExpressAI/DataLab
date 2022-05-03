from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from typing import Optional
# nltk package for preprocessing
import nltk

from datalabs.operations.tokenizer import get_tokenizer
from .preprocessing import *


@preprocessing(name="lower", contributor="datalab",
               task="Any", description="this function is used to lowercase a given text")
def lower(text: str,
          task_type: Optional[str] = None,
          language: Optional[str] = None,
          ) -> str:
    """
    Package: python
    Input:
        text:str
    Output:
        str
    """
    # text = sample['text']
    return {"text_lower": text.lower()}


@preprocessing(name="tokenize_nltk", contributor="nltk",
               task="Any", description="this function is used to tokenize a text using NLTK")
def tokenize_nltk(text: str,
                  task_type: Optional[str] = None,
                  language: Optional[str] = None,
                  ) -> List:
    """
    Package: nltk.word_tokenize
    Input:
        text:str
    Output:
        List
    """
    # text = sample['text']
    return {"text_tokenize": nltk.word_tokenize(text)}


@preprocessing(name="tokenize_huggingface", contributor="huggingface",
               task="Any", description="this function is used to tokenize a text using huggingface library")
def tokenize_huggingface(text: str,
                         task_type: Optional[str] = None,
                         language: Optional[str] = None,
                         ) -> List:
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
    # TODO: We need to use tokenizer.add_tokens() to add our vocabulary
    # before we can use this tokenizer. However current PLMs have their
    # own vocabulary and tokenizer. Maybe leave this for now.
    output = tokenizer.encode(text)
    return {"text_tokenize": output.tokens}


@preprocessing(name="stem", contributor="nltk",
               task="Any", description="this function is used to stem a text using NLTK")
def stem(text: str,
         task_type:Optional[str] = None,
         language:Optional[str] = None,
         ) -> List:
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
    return {"text_stem": stem_words}



@preprocessing(name="tokenize", contributor="datalabs",
               task="Any", description="this function is used to tokenize a text")
def tokenize(text:str,
             tokenizer_name:Optional[str] = None,
             task_type:str = None,
             language:str = None) -> List[str]:

    tokenizer = get_tokenizer(tokenizer_name, task_type, language)
    return {"text_tokenized": " ".join(tokenizer(text))}
