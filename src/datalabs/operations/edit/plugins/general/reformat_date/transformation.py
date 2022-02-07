"""Requirements
dateparser==1.0.0
Babel==2.9.1
"""

import dateparser
import numpy as np
import spacy
import os
import sys

try:
  from babel.dates import format_date
except ImportError:
  print("Trying to Install required module: babel\n")
  os.system('python -m pip install babel')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *

@editing(name = "reformat_date", contributor = "xl_augmenter",
         task = "Any", description="this function changes the format of dates appearing in text.")
def reformat_date(text:str, max_outputs = 1, seed = 0):
    spacy_nlp = spacy.load("en_core_web_sm")



    ymd_formats = ["short", "medium", "long"]
    ym_formats = ["MMM Y", "MMMM Y", "MMM YY", "MMMM YY"]
    md_formats = [
        "d MMM",
        "d MMMM",
        "dd MMM",
        "dd MMMM",
        "MMM d",
        "MMMM d",
        "MMM dd",
        "MMMM dd",
    ]

    locales_list = [
        "en_AU",
        "en_CA",
        "en_IN",
        "en_IE",
        "en_MT",
        "en_NZ",
        "en_PH",
        "en_SG",
        "en_ZA",
        "en_GB",
        "en_US",
    ]


    np.random.seed(seed)
    doc = spacy_nlp(text)
    transformed_texts = []

    for _ in range(max_outputs):
        for entity in doc.ents:
            new_value = None

            if entity.label_ == "DATE":
                date, has_year, has_month, has_day = parse_date(
                    entity.text
                )

                if date:
                    locale = np.random.choice(locales_list)
                    np.random.seed(seed)
                    if has_year and has_month and has_day:
                        format = np.random.choice(ymd_formats)
                        new_value = format_date(
                            date,
                            format=format,
                            locale=locale,
                        )
                    elif has_year and has_month:
                        format = np.random.choice(ym_formats)
                        new_value = format_date(
                            date,
                            format=format,
                            locale=locale,
                        )
                    elif has_month and has_day:
                        format = np.random.choice(md_formats)
                        new_value = format_date(
                            date,
                            format=format,
                            locale=locale,
                        )

                if new_value:
                    text = text.replace(entity.text, str(new_value))
        transformed_texts.append(text)

    # return transformed_texts
    return {"text_reformat_date":transformed_texts[0]}





def parse_date(text: str):
    """Parse the text to extract the date components and return a datetime object."""

    # By default the parser fills the missing values with current day"s values,
    # hence, using boolean values to keep track of what info is present in the text.
    date = None
    has_year = False
    has_month = False
    has_day = False

    # First check if the text contains all three parts of a date - Y, M, D.
    date = dateparser.parse(
        text, settings={"REQUIRE_PARTS": ["year", "month", "day"]}
    )

    if date is not None:
        has_year = True
        has_month = True
        has_day = True
    else:
        # Check if text contains two parts - Y and M.
        date = dateparser.parse(
            text, settings={"REQUIRE_PARTS": ["year", "month"]}
        )
        if date is not None:
            has_year = True
            has_month = True
        else:
            # Check if text contains two parts - M and D.
            date = dateparser.parse(
                text, settings={"REQUIRE_PARTS": ["month", "day"]}
            )
            if date is not None:
                has_month = True
                has_day = True

    return date, has_year, has_month, has_day



# sentence = "As of 20 June 2021, 2.66 billion doses of COVID‑19 vaccine have been administered worldwide based on official reports from national health agencies."
# perturbed = reformat_date(text=sentence)
# print(perturbed)



