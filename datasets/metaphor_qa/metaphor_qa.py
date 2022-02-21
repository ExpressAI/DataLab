import json
import os
import csv
import datalabs
from datalabs.tasks import QuestionAnsweringMultipleChoices



_CITATION = """
TO BE ADDED
"""

# TODO(social_i_qa):
_DESCRIPTION = """\
Testing the Ability of Language Models to Interpret Figurative Language
"""
url_train_small = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train_s.csv"
url_train_medium = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train.csv"
url_train_large = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train_xl.csv"
url_validation = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/dev.csv"
url_test = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/test.csv"


class SocialIQa(datalabs.GeneratorBasedBuilder):
    """TODO(social_i_qa): Short description of my dataset."""

    # TODO(social_i_qa): Set up version.
    VERSION = datalabs.Version("0.1.0")

    def _info(self):
        # TODO(social_i_qa): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),  # context ->article
                    "question": datalabs.Value("string"),
                    "answers":  # answers -> label
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },
                    "options": datalabs.features.Sequence(datalabs.Value("string"))
                }


            ),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/nightingal3/metaphor-qa",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringMultipleChoices(
                    question_column="question", context_column="context", answers_column="answers",
                    options_column="options",
                    task="question-answering-multiple-choices-with-context",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = dl_manager.download_and_extract(url_train_medium)
        validation_path = dl_manager.download_and_extract(url_train_validation)
        test_path = dl_manager.download_and_extract(url_test)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": validation_path,
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                },
            ),
        ]

    def _generate_examples(self, filepath, labelpath):
        """Yields examples."""
        id_sample = 0
        with open(filepath, encoding="utf-8") as files:
            reader = csv.reader(files)
            next(reader, None)
            for id_, data in enumerate(reader):
                id_sample += 1
                context = data[0]
                option1 = data[1]
                option2 = data[2]
                option_index = int(data[3])
                valid = int(data[4])
                qid = int(data[5])

                options = [option1,option2]

                yield id_, {
                    "id": str(id_sample - 1),
                    "context": context,
                    "question": question,
                    "options": options,
                    "answers": {
                                "option_index": option_index,
                                "text": options[option_index],
                            },
                }

