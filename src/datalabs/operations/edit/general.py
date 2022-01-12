from typing import Dict, List, Optional, Any
from typing import Callable, Mapping
from operation import OperationFunction, operation_function
from .editing import *
import numpy as np
# nltk package for editing
import nltk

# spacy package for editing
import spacy

# checklist package for editing
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb





@editing(name = "strip_punctuation", contributor = "checklist",
         task = "Any", description="this function is used to ")
def strip_punctuation_checklist(text:str):

    nlp = spacy.load('en_core_web_sm')
    pdata = nlp(text)
    return Perturb.strip_punctuation(pdata)


@editing(name = "add_typos", contributor = "checklist",
         task = "Any", description="this function is used to ")
def add_typos_checklist(text:str):

    return Perturb.add_typos(text)



@editing(name = "contract", contributor = "checklist",
         task = "Any", description="this function is used to ")
def contract_checklist(text:str):

    return Perturb.contract(text)





