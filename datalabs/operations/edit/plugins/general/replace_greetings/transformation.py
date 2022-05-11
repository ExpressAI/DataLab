import itertools
import os
import random
import re
import sys

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


# Setting up a tuple of possible replaceable greetings and farewells
GREETINGS = (
    "Hi",
    "Hello",
    "Hey",
    "Howdy",
    "Greetings",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "What's up",
    "Sup",
)

# Creating a tuple of sentences that can't be simply replaced by a short greeting
SPECIAL_GREETINGS = (
    "It's nice to meet you.",
    "It's great to meet you.",
    "Pleased to meet you.",
    "How are you?",
    "How are you doing?",
    "How is it going on?",
    "How is it going?",
)

FAREWELLS = (
    "Goodbye",
    "Bye bye",
    "Bye",
    "See you soon",
    "See you",
    "See ya",
    "Best regards",
    "Have a nice day",
    "Have a great day",
    "Have a nice weekend",
    "Good night",
    "I gotta go",
)

# Compiling regex
GREETINGS_REGEX = (
    re.compile("good (morning|afternoon|evening)", re.IGNORECASE),
    re.compile(r"\b(what's up|sup)\b(\?|)", re.IGNORECASE),
    re.compile(r"\b(hi|hello|hey|howdy)\b", re.IGNORECASE),
)

SPECIAL_GREETINGS_REGEX = (
    re.compile(
        "(it('s| is| was) |)(a |)(nice|great|pleased|pleasure) to meet (you|u)(.|)",
        re.IGNORECASE,
    ),
    re.compile(r"how( are|'re|) (you|u) doin(g|'|)\?", re.IGNORECASE),
    re.compile(r"how( is|'s|) it going( on|)\?", re.IGNORECASE),
    re.compile(r"how( have|'ve|) you been\?", re.IGNORECASE),
)

FAREWELLS_REGEX = (
    re.compile("(good night|goodbye|goodnight)", re.IGNORECASE),
    re.compile("see (you|u|ya)( soon| later| tomorrow|)", re.IGNORECASE),
    re.compile("have a (great|nice|good) (day|night|week|weekend)", re.IGNORECASE),
    re.compile("best regards", re.IGNORECASE),
    re.compile(
        r"\b(by+e+)+\b", re.IGNORECASE
    ),  # it matches 'bye', 'bye bye', 'byyyyyyye', 'byeeeee', etc.
)


def greetings_and_farewells(text, seed=0, max_outputs=1):
    random.seed(seed)

    output_texts = []

    for _ in itertools.repeat(None, max_outputs):
        processed_text = text

        for regex_tuple, replaceable_choices in zip(
            [GREETINGS_REGEX, SPECIAL_GREETINGS_REGEX, FAREWELLS_REGEX],
            [GREETINGS, SPECIAL_GREETINGS, FAREWELLS],
        ):
            for regex in regex_tuple:
                processed_text = regex.sub(
                    random.choice(replaceable_choices), processed_text
                )

        output_texts.append(processed_text)

    return output_texts


@editing(
    name="replace_greetings",
    contributor="xl_augmenter",
    task="Any",
    description="This transformation will replace greetings (e.g. Hi, Howdy)"
    " and farewells (e.g. See you, Good night) by a similar one.",
)
def replace_greetings(text: str, seed=0, max_outputs=1):

    processed_text = greetings_and_farewells(
        text=text, seed=seed, max_outputs=max_outputs
    )
    # return processed_text
    return {"text_replace_greetings": processed_text[0]}


# sentence = "Good morning, John. I've sent a memo to your desk, let me know
# if you need anything else. Best regards"
# perturbed = replace_greetings(text=sentence)
# print(perturbed)
