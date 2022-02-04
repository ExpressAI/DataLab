import random
import json
import spacy
import os.path
#from initialize import spacy_nlp
# python -m spacy download en_core_web_sm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *


@editing(name = "abbreviate", contributor = "xl_augmenter",
         task = "Any", description="Replaces a word or phrase with its abbreviated counterpart")
def abbreviate(text, prob = 0.5, seed = 0, max_outputs = 1):
    scriptpath = os.path.dirname(__file__)
    with open(os.path.join(scriptpath, '../../../resources/phrase_abbrev_dict.json'), 'r') as file:
        phrase_abbrev_dict = json.loads(file.read())
    with open(os.path.join(scriptpath, '../../../resources/word_abbrev_dict.json'), 'r') as file:
        word_abbrev_dict = json.loads(file.read())

    spacy_nlp = spacy.load("en_core_web_sm")
    random.seed(seed)
    transf = []
    for _ in range(max_outputs):
        trans_text = text
        for phrase in phrase_abbrev_dict:
            if random.random() < prob:
                trans_text = trans_text.replace(phrase, phrase_abbrev_dict[phrase])
        doc = spacy_nlp(trans_text).doc
        trans = []
        for token in doc:
            word = token.text
            if word in word_abbrev_dict and random.random() < prob:
                trans.append(word_abbrev_dict[word])
            else:
                trans.append(word)
        trans1 = " ".join([str(word) for word in trans])
        transf.append(trans1)
    #return transf
    return {"text_abbreviate":transf[0]}



# sentence = "I will turn in the homework on Friday for sure!"
# perturbed = abbreviate(text=sentence)
# print(perturbed)


