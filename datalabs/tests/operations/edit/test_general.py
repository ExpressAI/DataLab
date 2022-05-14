import unittest

from edit import (
    abbreviate,
    abbreviate_country_state,
    abbreviate_weekday_month,
    add_filler_words,
    add_typo,
    add_typos_checklist,
    britishize_americanize,
    change_city_name,
    change_color,
    change_person_name,
    emojify,
    replace_acronyms,
    replace_greetings,
    replace_hypernyms,
    replace_hyponyms,
    replace_synonym,
    simple_cipher,
    slangificator,
    strip_punctuation_checklist,
)

from datalabs import load_dataset


class MyTestCase(unittest.TestCase):
    def test_general(self):

        dataset = load_dataset("qc")

        res = dataset["test"].apply(add_typos_checklist, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(strip_punctuation_checklist, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(abbreviate, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(abbreviate_country_state, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(abbreviate_weekday_month, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(add_filler_words, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(add_typo, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(britishize_americanize, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(change_city_name, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(change_color, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(change_person_name, mode="realtime")
        print(next(res))

        # res = dataset["test"].apply(change_person_name_by_culture)
        # print(next(res)) # null result

        res = dataset["test"].apply(emojify, mode="realtime")
        print(next(res))

        # res = dataset["test"].apply(factive_verb)  # wrong
        # print(next(res))

        # res = dataset["test"].apply(reformat_date, mode = "realtime")
        # print(next(res))

        res = dataset["test"].apply(replace_acronyms, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(replace_greetings, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(replace_hypernyms, mode="realtime")  # slow
        print(next(res))

        res = dataset["test"].apply(replace_hyponyms, mode="realtime")  # slow
        print(next(res))

        res = dataset["test"].apply(replace_synonym, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(simple_cipher, mode="realtime")
        print(next(res))

        res = dataset["test"].apply(slangificator, mode="realtime")
        print(next(res))


if __name__ == "__main__":
    unittest.main()
