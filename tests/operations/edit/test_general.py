import unittest

from datalabs import operations, load_dataset
from edit import *


class MyTestCase(unittest.TestCase):



    def test_general(self):

        dataset = load_dataset("ag_news")

        res = dataset["test"].apply(add_typos_checklist)
        print(next(res))

        res = dataset["test"].apply(strip_punctuation_checklist)
        print(next(res))

        res = dataset["test"].apply(abbreviate)
        print(next(res))

        res = dataset["test"].apply(abbreviate_country_state)
        print(next(res))

        res = dataset["test"].apply(abbreviate_weekday_month)
        print(next(res))


        res = dataset["test"].apply(add_filler_words)
        print(next(res))

        res = dataset["test"].apply(add_typo)
        print(next(res))

        res = dataset["test"].apply(britishize_americanize)
        print(next(res))

        res = dataset["test"].apply(change_city_name)
        print(next(res))

        res = dataset["test"].apply(change_color)
        print(next(res))

        res = dataset["test"].apply(change_person_name)
        print(next(res))

        # res = dataset["test"].apply(change_person_name_by_culture)
        # print(next(res)) # null result

        res = dataset["test"].apply(emojify)
        print(next(res))

        # res = dataset["test"].apply(factive_verb)  # wrong
        # print(next(res))

        res = dataset["test"].apply(insert_abbreviation)
        print(next(res))

        res = dataset["test"].apply(reformat_date)
        print(next(res))

        res = dataset["test"].apply(replace_acronyms)
        print(next(res))

        res = dataset["test"].apply(replace_greetings)
        print(next(res))

        res = dataset["test"].apply(replace_hypernyms) # slow
        print(next(res))

        res = dataset["test"].apply(replace_hyponyms) # slow
        print(next(res))

        res = dataset["test"].apply(replace_synonym)
        print(next(res))

        res = dataset["test"].apply(simple_cipher)
        print(next(res))

        res = dataset["test"].apply(slangificator)
        print(next(res))


if __name__ == '__main__':
    unittest.main()
