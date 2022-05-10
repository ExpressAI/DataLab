import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *



def transformation(sentence, lowercase, acronyms):
    new_sentence = sentence
    lower_sentece = sentence.lower()
    for key in acronyms.keys():
        lower_key = key.lower()
        if key in new_sentence:
            new_sentence = new_sentence.replace(key, acronyms[key])
        elif lowercase and (lower_key in lower_sentece):
            key_index = lower_sentece.index(lower_key)
            key_end = key_index + len(key)
            new_sentence = (
                new_sentence[:key_index]
                + acronyms[key]
                + new_sentence[key_end:]
            )

    return new_sentence

@editing(name = "replace_acronyms", contributor = "xl_augmenter",
         task = "Any", description="This transformation changes abbreviations and acronyms appearing in a text to their expanded form and respectively,")
def replace_acronyms(text:str, seed=0, max_outputs=1,lowercase=False):

    acronyms_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../resources/acronyms.tsv")
    # print(acronyms_file_path)
    sep = "\t"
    encoding = "utf-8"

    # Load acronyms from file
    temp_acronyms = {}
    with open(acronyms_file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(sep)
            temp_acronyms[key] = value
    # Place long keys first to prevent overlapping
    acronyms = {}
    for k in sorted(temp_acronyms, key=len, reverse=True):
        acronyms[k] = temp_acronyms[k]

    # return [transformation(text, lowercase, acronyms)]

    return {"text_replace_acronyms":transformation(text, lowercase, acronyms)}


# sentence = "I studied at New York University and Massachusetts Institute of Technology."
# perturbed = replace_acronyms(text=sentence)
# print(perturbed)