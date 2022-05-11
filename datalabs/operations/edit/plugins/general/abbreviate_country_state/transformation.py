from collections import defaultdict
import json
import os
import random
import re
import sys

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


def dict_value_helper(d, key):
    if type(d[key]) is list:
        random.seed(0)
        return random.choice(d[key])
    else:
        return d[key]


@editing(
    name="abbreviate_country_state",
    contributor="xl_augmenter",
    task="Any",
    description="this function adds Country/State name/abbreviation with"
    " flexible options: Pennsylvania -> PA or PA -> Pennsylvania",
)
def abbreviate_country_state(
    text: str,
    seed=0,
    country=True,
    state=True,
    country_filter="USA",
    abbr=True,
    exp=True,
):

    abbr_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../resources/country_state_abbreviation.json",
    )

    with open(abbr_file_path, "r") as file:
        abbr_json = json.loads(file.read())

    country_abbr = {country["name"]: country["abbr"] for country in abbr_json}
    country_exp = {country["abbr"]: country["name"] for country in abbr_json}
    state_abbr = {}
    state_exp = {}

    if country_filter:
        country = next((x for x in abbr_json if x["abbr"] == country_filter), None)
        if country:
            for state in country["states"]:
                state_abbr[state["name"]] = state["abbr"]
                state_exp[state["abbr"]] = state["name"]
    else:
        state_exp = defaultdict(list)
        state_abbr = defaultdict(list)
        for country in abbr_json:
            for state in country["states"]:
                state_exp[state["abbr"]].append(state["name"])
                state_abbr[state["name"]].append(state["abbr"])

    country_mapping = {}
    state_mapping = {}
    if country:
        if abbr:
            country_mapping = {**country_mapping, **country_abbr}
        if exp:
            country_mapping = {**country_mapping, **country_exp}
    if state:
        if abbr:
            state_mapping = {**state_mapping, **state_abbr}
        if exp:
            state_mapping = {**state_mapping, **state_exp}

    # Build regex pattern for country abbreviation/full name by joining
    # country_mapping keys
    # eg. country_pattern = '...|United States|...|USA|...'
    country_pattern = "|".join(country_mapping.keys())
    # Adding backslash before '(' and ')' in country name as the escape
    # character (eg. Virgin Islands (US))
    country_pattern = country_pattern.replace("(", r"\(")
    country_pattern = country_pattern.replace(")", r"\)")
    country_regex = re.compile(r"(^|\s+)(" + country_pattern + r")(\s+|\?|!|$|\.|,)")
    perturbed_text = country_regex.sub(
        lambda y: y[1] + country_mapping[y[2]] + y[3],
        text,
    )

    # Build regex pattern for state abbreviation/full name by joining state_mapping keys
    # eg. state_pattern = '...|Pennsylvania|...|PA|...'
    state_pattern = "|".join(state_mapping.keys())
    # Adding backslash before '(' and ')' in state name as the escape character
    state_pattern = state_pattern.replace("(", r"\(")
    state_pattern = state_pattern.replace(")", r"\)")
    state_regex = re.compile(r"(^|\s+)(" + state_pattern + r")(\s+|\?|!|$|\.|,)")
    perturbed_text = state_regex.sub(
        lambda y: y[1] + dict_value_helper(state_mapping, y[2]) + y[3],
        perturbed_text,
    )
    # return [perturbed_text]
    return {"text_abbreviate_country_state": perturbed_text}


# sentence = "Texas and Oklahoma are also seeing extreme and exceptional drought."
# perturbed = abbreviate_country_state(text=sentence)
# print(perturbed)
