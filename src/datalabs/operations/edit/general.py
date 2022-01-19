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





@editing(name = "strip_punctuation_checklist", contributor = "checklist",
         task = "Any", description="strip the punctuation of a given text")
def strip_punctuation_checklist(text:str):

    nlp = spacy.load('en_core_web_sm')
    pdata = nlp(text)
    return Perturb.strip_punctuation(pdata)


@editing(name = "add_typos_checklist", contributor = "checklist",
         task = "Any", description="add typos randomly into a given text")
def add_typos_checklist(text:str):

    return Perturb.add_typos(text)



@editing(name = "contract_checklist", contributor = "checklist",
         task = "Any", description="contracts and expands contractions if present")
def contract_checklist(text:str):

    return Perturb.contract(text)





