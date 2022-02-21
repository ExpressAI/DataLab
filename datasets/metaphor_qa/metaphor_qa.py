import json
import os
import csv
import datalabs
from datalabs.tasks import QuestionAnsweringMultipleChoices
from featurize.general import get_features_sample_level
from datalabs.utils.more_features import prefix_dict_key, get_feature_arguments


def infer_schema_dataset_level(sample_level_schema:dict):

    dataset_level_schema = {}
    for feature_name, value in sample_level_schema.items():
        if isinstance(value, int) or isinstance(value, float):
            dataset_level_schema[feature_name] = value
    return dataset_level_schema

_CITATION = """
TO BE ADDED
"""


_DESCRIPTION = """\
Testing the Ability of Language Models to Interpret Figurative Language
"""
url_train_small = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train_s.csv"
url_train_medium = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train.csv"
url_train_large = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/train_xl.csv"
url_validation = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/dev.csv"
url_test = "https://raw.githubusercontent.com/nightingal3/metaphor-qa/master/data/filtered/test.csv"





class MetaphorQAConfig(datalabs.BuilderConfig):
    """BuilderConfig for FB15K."""

    def __init__(self, **kwargs):
        """BuilderConfig for MetaphorQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MetaphorQAConfig, self).__init__(**kwargs)


FIELD = "context"
class MetaphorQA(datalabs.GeneratorBasedBuilder):


    BUILDER_CONFIGS = [
        MetaphorQAConfig(
            name="small",
            version=datalabs.Version("1.0.0"),
            description="small training set",
        ),
        MetaphorQAConfig(
            name="medium",
            version=datalabs.Version("1.0.0"),
            description="medium training set",
        ),
        MetaphorQAConfig(
            name="large",
            version=datalabs.Version("1.0.0"),
            description="large training set",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medium"



    def _info(self):



        features_dataset = {}
        features_sample = datalabs.Features(
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


            )

        if self.feature_expanding:
            sample_level_schema = get_features_sample_level("This is a test sample")
            dict_feature_argument = get_feature_arguments(sample_level_schema, field=FIELD, feature_level="sample_level")
            additional_features = datalabs.Features(dict_feature_argument)
            features_sample.update(additional_features)


            dataset_level_schema = infer_schema_dataset_level(sample_level_schema)
            dict_feature_argument = get_feature_arguments(dataset_level_schema, field="avg" + "_" + FIELD, feature_level="dataset_level")
            features_dataset.update(datalabs.Features(dict_feature_argument))






        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=features_sample,
            features_dataset=features_dataset,
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

        train_path = ""

        if self.config.name == "small":
            train_path = dl_manager.download_and_extract(url_train_small)
        elif  self.config.name == "medium":
            train_path = dl_manager.download_and_extract(url_train_medium)
        elif self.config.name == "large":
            train_path = dl_manager.download_and_extract(url_train_large)


        validation_path = dl_manager.download_and_extract(url_validation)
        test_path = dl_manager.download_and_extract(url_test)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": train_path}
            ),
            # datalabs.SplitGenerator(
            #     name=datalabs.Split.VALIDATION,
            #     gen_kwargs={"filepath": validation_path}
            # ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
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
                valid = data[4]
                qid = data[5]

                options = [option1,option2]
                question = ""

                raw_feature_info = {
                    "id": str(id_sample - 1),
                    "context": context,
                    "question": question,
                    "options": options,
                    "answers": {
                                "option_index": option_index,
                                "text": options[option_index],
                            },
                }

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = prefix_dict_key(get_features_sample_level(context), FIELD)
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info

