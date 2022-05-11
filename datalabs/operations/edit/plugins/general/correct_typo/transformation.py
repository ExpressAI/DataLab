import json
import os
import sys

import spacy

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


@editing(
    name="correct_typo",
    contributor="xl_augmenter",
    task="Any",
    description="This transformation perturbs text to correct common misspellings",
)
def correct_typo(text: str):

    scriptpath = os.path.dirname(__file__)
    with open(
        os.path.join(scriptpath, "../../../resources/spell_corrections.json"), "r"
    ) as file:
        COMMON_MISSPELLINGS_DICT = json.loads(file.read())

    spacy_nlp = spacy.load("en_core_web_sm")

    doc = spacy_nlp(text)

    perturbed_text = [
        COMMON_MISSPELLINGS_DICT.get(token.text, token.text) + " "
        if token.whitespace_
        else COMMON_MISSPELLINGS_DICT.get(token.text, token.text)
        for token in doc
    ]
    # return ["".join(perturbed_text)]

    return {"text_correct_typo": "".join(perturbed_text)}


# sentence = "Andrew andd Alice finally returnd the French
# book that I bought lastr week"
# perturbed = correct_typo(text=sentence)
# print(perturbed)
