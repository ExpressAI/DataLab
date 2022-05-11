import json
import os
import random
import sys
from typing import List

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


# Check if any emoji(icon) from dict1 in text and substitute them with
# a random corresponding icon(emoji) from dict2
def convert(perturbed_text: str, dict1: dict, dict2: dict) -> str:
    for k in dict1:  # k is the emoji/icon type (e.g., ":)" is of type smiley
        for s in dict1[k]:
            if s in perturbed_text:
                perturbed_text = perturbed_text.replace(s, random.choice(dict2[k]))
    return perturbed_text


# generate max_outputs different perturbed texts for each text, selecting
# if translating emoji to icon or icon to emoji with emoji_to_icon variable


def emoji2icon(
    text: str,
    text2emoji: dict,
    text2icon: dict,
    seed: int = 42,
    max_outputs: int = 1,
    emoji_to_icon: bool = True,
) -> List[str]:
    random.seed(seed)

    perturbed_texts = []
    for _ in range(max_outputs):
        perturbed_text = text
        if emoji_to_icon:
            perturbed_text = convert(perturbed_text, text2emoji, text2icon)
        else:
            perturbed_text = convert(perturbed_text, text2icon, text2emoji)
        perturbed_texts.append(perturbed_text)
    return perturbed_texts


@editing(
    name="emojify",
    contributor="xl_augmenter",
    task="Any",
    description="augments the input sentence by swapping words into"
    " emojis with similar meanings.",
)
def emojify(
    text: str, seed: int = 42, max_outputs: int = 1, emoji_to_icon: bool = False
):
    text2emoji_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../resources/text2emoji.json"
    )

    text2icon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../resources/text2icon.json"
    )

    with open(text2emoji_path, "r") as file:
        text2emoji = json.loads(file.read())

    with open(text2icon_path, "r") as file:
        text2icon = json.loads(file.read())

    perturbed_texts = emoji2icon(
        text=text,
        text2emoji=text2emoji,
        text2icon=text2icon,
        seed=seed,
        max_outputs=max_outputs,
        emoji_to_icon=emoji_to_icon,
    )

    return {"text_emojify": perturbed_texts[0]}
    # return perturbed_texts


# sentence = "I am happy :) but you are sad :("
# perturbed = emojify(text=sentence)
# print(perturbed)
