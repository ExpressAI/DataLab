import hashlib
import os
import random
import sys

import spacy

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


def hash(input: str):
    """
    Function to hash a sentence
    Parameters
    ----------
    input : str
        Input sentence to hash.
    Returns
    -------
    n : int
        Hashed value of the sentence.
    """
    t_value = input.encode("utf8")
    h = hashlib.sha256(t_value)
    n = int(h.hexdigest(), base=16)
    return n


def iob_ent_dict(doc):
    """
    Given a spaCy Doc object, this creates a list of dictionaries
     mapping each token to its text, IOB embedding, and entity
    Parameters
    ----------
    doc : SpaCy Doc
        SpaCy doc object outputted by spaCy model.
    Returns
    -------
    d : list<dict>
        List of dictionaries mapping each token in a Doc object
        to its text, IOB embedding, and entity.
    """
    d = []
    for i in doc:
        d.append({"Word": i.text, "IOB": i.ent_iob_, "Ent": i.ent_type_})
    return d


def create_ents_dict(doc):
    spans = []
    ind = 0
    d = iob_ent_dict(doc)
    while ind < len(d):
        span = ""
        ent = ""
        if d[ind]["IOB"] == "O":
            span += d[ind]["Word"]
            ind += 1
        elif d[ind]["IOB"] == "B":
            ent = d[ind]["Ent"]
            span += d[ind]["Word"]
            ind += 1
            while ind < len(d) and d[ind]["IOB"] == "I":
                span += " " + d[ind]["Word"]
                ind += 1
        spans.append({"Word": span, "Entity": ent})
    return spans


@editing(
    name="change_city_name",
    contributor="xl_augmenter",
    task="Any",
    description="replaces instances of populous and well-known cities in "
    "a sentence with instances of less populous and less"
    " well-known cities.",
)
def change_city_name(text: str, seed=None):

    spacy_nlp = spacy.load("en_core_web_sm")
    doc = spacy_nlp(text)

    scriptpath = os.path.dirname(__file__)
    f_pop = open(os.path.join(scriptpath, "../../../resources/Eng_Pop.txt"))
    f_scarce = open(os.path.join(scriptpath, "../../../resources/Eng_Scarce.txt"))

    populous_cities = f_pop.read().split("\n")
    scarce_cities = f_scarce.read().split("\n")

    f_pop.close()
    f_scarce.close()

    if seed is not None:
        random.seed(seed)
    ents_dict = create_ents_dict(doc)
    sent_words = []
    for i in ents_dict:
        if i["Word"] in list(populous_cities) and (
            i["Entity"] == "GPE" or i["Entity"] == "LOC"
        ):
            sent_words.append("<CITY>")
        else:
            sent_words.append(i["Word"])
    new_sentence = " ".join(sent_words)
    while "<CITY>" in new_sentence:
        rand_city = scarce_cities[random.randint(0, len(scarce_cities))]
        new_sentence = new_sentence.replace("<CITY>", rand_city, 1)

    # return new_sentence
    return {"text_change_city_name": new_sentence}


# sentence = "The team was established in Dallas in 1898 and was a
# charter member of the NFL in 1920."
# perturbed = change_city_name(text=sentence)
# print(perturbed)
