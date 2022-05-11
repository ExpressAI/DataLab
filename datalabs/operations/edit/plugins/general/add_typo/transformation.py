import json
import os
import random
import sys

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


"""
Base Class for implementing the different input transformations
 a generation should be robust against.
"""


def generate_sentence(sentence, spell_errors, prob_of_typo, seed):
    output = []
    for word in sentence.split():
        random.seed(seed)
        if (
            word.lower() in list(spell_errors.keys())
            and random.choice(range(0, 100)) <= prob_of_typo
        ):
            output.append(random.choice(spell_errors[word.lower()]))
        else:
            output.append(word)
    output = " ".join(output)
    return output


def generate_sentences(text, prob=0.1, seed=0, max_outputs=1):

    scriptpath = os.path.dirname(__file__)
    with open(
        os.path.join(scriptpath, "../../../resources/spell_errors.json"), "r"
    ) as file:
        spell_errors = json.loads(file.read())

    prob_of_typo = int(prob * 100)

    perturbed_texts = []
    for idx in range(max_outputs):
        new_text = generate_sentence(text, spell_errors, prob_of_typo, seed + idx)
        perturbed_texts.append(new_text)
    return perturbed_texts


@editing(
    name="add_typo",
    contributor="xl_augmenter",
    task="Any",
    description="this function adds a typo into a text",
)
def add_typo(text: str, seed=0, max_outputs=2):

    perturbed_texts = generate_sentences(
        text=text,
        prob=0.20,
        seed=seed,
        max_outputs=max_outputs,
    )
    # return perturbed_texts
    return {"text_add_typo": perturbed_texts[0]}


# sentence = "Andrew finally returned the French book to Chris that I bought last week"
# perturbed = add_typo(text=sentence)
# print(perturbed)
