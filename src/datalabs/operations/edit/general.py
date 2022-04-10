from typing import Any, Callable, Dict, List, Mapping, Optional

# checklist package for editing
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb

# nltk package for editing
import nltk
import numpy as np
from operation import operation_function, OperationFunction

# spacy package for editing
import spacy

from .editing import *


@editing(
    name="strip_punctuation_checklist",
    contributor="checklist",
    task="Any",
    description="strip the punctuation of a given text. For example, "
    "Input: I love this movie. How about you? Output: I love this movie. How about you",
)
def strip_punctuation_checklist(text: str):

    nlp = spacy.load("en_core_web_sm")
    pdata = nlp(text)
    return {"text_strip_punctuation": Perturb.strip_punctuation(pdata)}


@editing(
    name="add_typos_checklist",
    contributor="checklist",
    task="Any",
    description="add typos randomly into a given text. For example,"
    "Input: I love this movie. How about you? Output: I love this movie. How about yuo?",
)
def add_typos_checklist(text: str):
    return {"text_add_typos": Perturb.add_typos(text)}
