import itertools
import random
import spacy
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from edit.editing import *


def slangifyPoS(
    token, modified_toks, Dictionary, PoS, probReplace, isCap, ReplPot, ReplMade
):  # performs transformation similar to all three PoS

    # Check if word is in the corresponding dictionary
    if token.lemma_ in Dictionary[0]:
        ReplPot += 1  # increment potential replacements

        repDecision = (
            random.uniform(0, 1) <= probReplace
        )  # Randomly decide whether to replace or not

        if repDecision:  # if replacement is made
            ReplMade += 1

            # Choose a new word for replacement
            # ind=Slang_Adverbs[0].index(token.lemma_)
            indAllPosRepl = [
                i for i, x in enumerate(Dictionary[0]) if x == token.lemma_
            ]  # all possible replacements
            indChosenRepl = random.randint(
                0, len(indAllPosRepl) - 1
            )  # choose one of the replacements
            indChosenRepl = indAllPosRepl[indChosenRepl]  # index of that replacement

            if PoS == "Noun":  # Treat plular case for nouns
                # Take plural or singular form. Note only for nouns
                if token.tag_ == "NN" or token.tag_ == "NNP":
                    temp = Dictionary[1][
                        indChosenRepl
                    ]  # pick the word chosen for replcacement
                else:
                    temp = Dictionary[2][
                        indChosenRepl
                    ]  # pick the word chosen for replcacement
            else:
                temp = Dictionary[1][
                    indChosenRepl
                ]  # pick the word chosen for replcacement

            if isCap:  # Make the fist letter capital if necessary
                temp = temp[0].upper() + temp[1:]

            modified_toks.append(temp + token.whitespace_)

        else:  # if no replacement is made
            modified_toks.append(token.text + token.whitespace_)

    else:  # if not in the dictionary
        modified_toks.append(token.text + token.whitespace_)

@editing(name = "slangificator", contributor = "xl_augmenter",
         task = "Any", description="This transformation replaces some of the words (in particular, nouns, adjectives, and adverbs) of the original text with their corresponding slang. ")
def slangificator(
    text,
    probReplaceNoun=1.0,
    probReplaceAdjective=1.0,
    probReplaceAdverb=1.0,
    seed=0,
    max_outputs=1,
):
    pathDic = os.path.dirname(os.path.abspath(__file__))

    nlp = spacy.load("en_core_web_sm")  # get an instance of the tokenizer

    # Load dictionaries
    fin = open(os.path.join(pathDic, "../../../resources/Slang_Nouns.txt"), "r")
    Slang_Nouns = [
        line.strip("\n\r").split(",")
        for line in fin
    ]
    fin.close()

    fin = open(os.path.join(pathDic, "../../../resources/Slang_Adverbs.txt"), "r")
    Slang_Adverbs = [
        line.strip("\n\r").split(",")
        for line in fin
    ]
    fin.close()

    fin = open(os.path.join(pathDic, "../../../resources/Slang_Adjectives.txt"), "r")
    Slang_Adjectives = [
        line.strip("\n\r").split(",")
        for line in fin
    ]
    fin.close()



    random.seed(seed)


    perturbed_texts = []  # output for all perturbed texts

    # Load dictionaries
    Slang_Nouns = Slang_Nouns
    Slang_Nouns = list(map(list, zip(*Slang_Nouns)))
    Slang_Adverbs = Slang_Adverbs
    Slang_Adverbs = list(map(list, zip(*Slang_Adverbs)))
    Slang_Adjectives = Slang_Adjectives
    Slang_Adjectives = list(map(list, zip(*Slang_Adjectives)))

    # Tags for nouns
    noun_tag = ["NN", "NNS", "NNPS", "NNP"]

    # Tokenize text
    doc = nlp(text)

    for _ in itertools.repeat(None, max_outputs):

        ReplPot = 0  # counts potential replcacements, which could have been made if the probability of replacement would be set to one for all PoS
        ReplMade = 0  # counts actual replcacements

        modified_toks = []  # modified tokens
        for token in doc:
            isCap = token.text[
                0
            ].isupper()  # check if token begins with the capital letter

            # Nouns
            if token.tag_ in noun_tag:
                slangifyPoS(
                    token,
                    modified_toks,
                    Slang_Nouns,
                    "Noun",
                    probReplaceNoun,
                    isCap,
                    ReplPot,
                    ReplMade,
                )

            # Adverbs
            elif token.tag_ == "RB":
                slangifyPoS(
                    token,
                    modified_toks,
                    Slang_Adverbs,
                    "Adverb",
                    probReplaceAdverb,
                    isCap,
                    ReplPot,
                    ReplMade,
                )

            # Adjectives
            elif token.tag_ == "JJ":
                slangifyPoS(
                    token,
                    modified_toks,
                    Slang_Adjectives,
                    "Adjective",
                    probReplaceAdjective,
                    isCap,
                    ReplPot,
                    ReplMade,
                )

            else:  # if there is no part of speech which might be replaced then just keep the original one
                modified_toks.append(token.text + token.whitespace_)

        modified_toks = "".join(modified_toks)  # Reconstruct the transformed text

        perturbed_texts.append(modified_toks)

    return {"text_slangificator":perturbed_texts[0]}
    # return perturbed_texts







# sentence = "The hooligans in balaclavas have attempted to steal jewellery."
# perturbed = slangificator(text=sentence)
# print(perturbed)