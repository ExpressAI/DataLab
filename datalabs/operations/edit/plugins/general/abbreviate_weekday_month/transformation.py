import json
import os
import re
from typing import List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *


def weekday_month_abbreviate(text, abbreviations, expansions, max_outputs=1):

    regex = re.compile(
        "(%s)"
        % (
            "|".join([x + "(?!s)" for x in abbreviations.keys()])
            + "|"
            + "|".join([x.replace(".", "\\.") for x in expansions.keys()])
        )
    )

    return [
        regex.sub(
            lambda y: {**abbreviations, **expansions}[
                y.string[y.start() : y.end()]
            ],
            text,
        )
    ]

@editing(name = "abbreviate_weekday_month", contributor = "xl_augmenter",
         task = "Any", description="this function adds noise to all types of text sources (sentence, paragraph, etc.) containing names of weekdays or months.")
def abbreviate_weekday_month(text:str, max_outputs=1):
    abbreviations_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../resources/weekday_month_abb_en.json",
    )
    expansions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../resources/weekday_month_exp_en.json",
    )

    with open(abbreviations_path, "r") as file:
        abbreviations = json.loads(file.read())

    with open(expansions_path, "r") as file:
        expansions = json.loads(file.read())



    perturbed_texts = weekday_month_abbreviate(
        text = text,
        abbreviations = abbreviations,
        expansions = expansions,
        max_outputs =  max_outputs
    )
    #return perturbed_texts
    return {"text_weekday_month_abbreviate":perturbed_texts[0]}


# sentence = "I am busy Saturday night."
# perturbed = abbreviate_weekday_month(text=sentence)
# print(perturbed)