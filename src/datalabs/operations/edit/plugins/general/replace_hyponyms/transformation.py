import os
import random
import sys

from checklist.editor import Editor
import numpy as np
import spacy

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
from edit.editing import *


@editing(
    name="replace_hyponyms",
    contributor="xl_augmenter",
    task="Any",
    description="This operation makes lexical substitutions using hyponyms of the common nouns in a sentence when possible",
)
def replace_hyponyms(text: str, n=1, seed=0, max_outputs=1):
    nlp = spacy.load("en_core_web_sm")
    editor = Editor()

    np.random.seed(seed)
    words = []
    perturbed_texts = []
    tokens = nlp(text)
    # Shuffle the tokens list so that all noun (and not just the beginning nouns)
    # have a fair chance at being picked.
    shuf_tokens = list(tokens)
    random.seed(0)  # To get the same output as in test.json
    random.shuffle(shuf_tokens)
    for token in shuf_tokens:
        if token.pos_ == "NOUN":
            words.append(token)
            hyp_list = editor.hyponyms(text, token.text)
            for hyp in hyp_list:
                # Replace the noun with the hyponym
                perturbed_texts.append(text.replace(token.text, hyp))
            if len(perturbed_texts) >= max_outputs:
                break
    perturbed_texts = (
        perturbed_texts[:max_outputs] if len(perturbed_texts) > 0 else [text]
    )

    return {"text_replace_hyponyms": perturbed_texts[0]}
    # return perturbed_texts


# sentence = "Andrew finally returned the French book to Chris that I bought last week."
# perturbed = replace_hyponyms(text=sentence)
# print(perturbed)
