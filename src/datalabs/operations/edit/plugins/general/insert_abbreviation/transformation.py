import itertools
import random
import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import grammaire


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *


def readfile(file):
    with open(file, encoding='utf8') as input:
        lines=input.readlines()
    return lines

def load_rules(file):
    with open(file, encoding='utf8') as input:
        str_rules = input.read()
    return str_rules

@editing(name = "insert_abbreviation", contributor = "xl_augmenter",
         task = "Any", description="This perturbation replaces in texts some well known words or expressions with (one of) their abbreviations.")
def insert_abbreviation(text:str, max_outputs = 1,  seed = 0, ):

    current_path = os.path.realpath(__file__).replace(os.path.basename(__file__), "../../../resources/")
    rulefile_en = f"{current_path}replacement_rules_en.txt"
    rules_en = load_rules(rulefile_en)
    # First we compile our rules...
    grammar_en = grammaire.compile(rules_en)

    results = grammaire.parse(text, grammar_en)
    # We now replace the strings with their label
    perturbed_texts = text
    # Each list in results is an element such as: [label, [left,right]]
    # label pertains from rules
    # left is the left offset of the isolated sequence of words
    # right is the right offset of the isolated sequence of words
    # elements are stored from last to first in the text along the offsets
    for v in results:
        from_token = v[1][0]
        to_token = v[1][1]
        perturbed_texts = perturbed_texts[:from_token] + v[0] + perturbed_texts[to_token:]
    # return [perturbed_texts]

    return {"text_insert_abbreviation":perturbed_texts}





# sentence = "Make sure you've gone online to download one of the vouchers - it's definitely not worth paying full price for!"
# perturbed = insert_abbreviation(text=sentence)
# print(perturbed)
