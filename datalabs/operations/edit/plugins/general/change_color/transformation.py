import json
import os
import random

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *

def get_(big_list, small_list):
    big_str = " ".join(big_list)
    small_str = " ".join(small_list)

    big_str.index(small_str)

@editing(name = "change_color", contributor = "xl_augmenter",
         task = "Any", description="This transformation augments the input sentence by randomly replacing colors.")
def change_color(text:str, max_outputs = 1, seed = 0, mapping: dict = None):

    scriptpath = os.path.dirname(__file__)
    with open(os.path.join(scriptpath, '../../../resources/colors.json'), 'r') as file:
        colors_dict = json.loads(file.read())

    color_names = [color["name"] for color in colors_dict.values()]

    if mapping is None: mapping = {}

    random.seed(seed)

    # Detokenize sentence
    detokenizer = TreebankWordDetokenizer()
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        words = nltk.word_tokenize(text)
    text = detokenizer.detokenize(words)

    # Detect colors in a given sentence
    colors_and_indices = []
    for color_name in color_names:
        try:
            idx = text.index(color_name)
        except ValueError:
            continue
        colors_and_indices.append((color_name, idx, idx + len(color_name)))

    # Transform colors
    new_sentences = []
    for _ in range(max_outputs):
        new_sentence = text
        for color, start_idx, end_idx in colors_and_indices[::-1]:
            # Choose color
            if color not in mapping:
                new_color = random.choice(color_names)
            else:
                new_color = random.choice(mapping[new_color])
            # Generate sentence
            new_sentence = (
                    new_sentence[:start_idx]
                    + new_color
                    + new_sentence[end_idx:]
            )
        new_sentences.append(new_sentence)

    # return new_sentences
    return {"text_change_color":new_sentences[0]}


# sentence = "I bought this pink pair of shoes today! Isn't it pretty?"
# perturbed = change_color(text=sentence)
# print(perturbed)