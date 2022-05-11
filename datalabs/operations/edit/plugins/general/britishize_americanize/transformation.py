import json
import os.path
import sys

from datalabs.operations.edit.editing import editing

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)


@editing(
    name="britishize_americanize",
    contributor="xl_augmenter",
    task="Any",
    description="This transformation takes a sentence and converts it "
    "from british english to american english and vice-versa",
)
def britishize_americanize(text: str):
    """
    Parameters:
        string(str): original string
        final_dict(dict): dictionary with all the different possible words
         in american and british english
    Returns:
        str: String after replacing the words
    """

    scriptpath = os.path.dirname(__file__)
    with open(
        os.path.join(scriptpath, "../../../resources/american_spellings.json"), "r"
    ) as file:
        american_spellings_dict = json.loads(file.read())
    with open(
        os.path.join(scriptpath, "../../../resources/british_spellings.json"), "r"
    ) as file:
        british_spellings_dict = json.loads(file.read())

    # Creating a custom vocab dictionary consisting of totally different
    # words for same context
    difference_british_to_american = {
        "trousers": "pants",
        "flat": "apartment",
        "bonnet": "hood",
        "boot": "trunk",
        "lorry": "truck",
        "university": "college",
        "holiday": "vacation",
        "jumper": "sweater",
        "trainers": "sneakers",
        "postbox": "mailbox",
        "biscuit": "cookie",
        "chemist": "drugstore",
        "shop": "store",
        "football": "soccer",
        "autumn": "fall",
        "barrister": "attorney",
        "bill": "check",
        "caravan": "trailer",
        "cupboard": "closet",
        "diversion": "detour",
        "dustbin": "trash can",
        "jug": "pitcher",
        "lift": "elevator",
        "mad": "crazy",
        "maize": "corn",
        "maths": "math",
        "motorbike": "motorcycle",
        "motorway": "freeway",
        "nappy": "diaper",
        "pavement": "sidewalk",
        "post": "mail",
        "postman": "mailman",
        "pub": "bar",
        "rubber": "eraser",
        "solicitor": "attorney",
        "tax": "cab",
        "timetable": "schedule",
        "torch": "flashlight",
        "waistcoat": "vest",
        "windscreen": "windshield",
        "angry": "mad",
        "caretaker": "janitor",
        "cot": "crib",
        "curtains": "drapes",
        "engine": "motor",
        "garden": "yard",
        "handbag": "purse",
        "hoarding": "billboard",
        "ill": "sick",
        "interval": "intermission",
        "luggage": "baggage",
        "nowhere": "noplace",
        "optician": "optometrist",
        "queue": "line",
        "rubbish": "trash",
    }
    # Replacing the keys with values and vice versa for the custom vocab dictionary
    # And merging both of them
    vocab_diff = dict((v, k) for k, v in difference_british_to_american.items())
    vocab_diff.update(difference_british_to_american)

    final_dict = {**american_spellings_dict, **british_spellings_dict, **vocab_diff}

    text = " ".join([final_dict.get(word, word) for word in text.split()])

    return {"text_britishize_americanize": text}
    # return [text]


# sentence = "I will turn in the homework on Friday for sure! trousers"
# perturbed = britishize_americanize(text=sentence)
# print(perturbed)
