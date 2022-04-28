import csv
import datalabs
from datalabs.tasks import QuestionAnsweringMultipleChoices
from datalabs.operations.featurize.qa_multiple_choices import get_features_sample_level, get_schema_of_sample_level_features

from datalabs.utils import private_utils
from datalabs.utils.logging import get_logger
from datalabs.utils.more_features import get_feature_arguments

logger = get_logger(__name__)

def infer_schema_dataset_level(sample_level_schema:dict):

    dataset_level_schema = {}
    for feature_name, value in sample_level_schema.items():
        if isinstance(value, int) or isinstance(value, float):
            dataset_level_schema[feature_name] = value
    return dataset_level_schema

_CITATION = """
"""


_DESCRIPTION = """\
This dataset contains examples of creative metaphors written by humans. It is formatted as a Winograd schema, with two
paired sentences that have opposite meanings. E.g. "Shopping for groceries is (finding shells on a sunny beach | a scavenger hunt
with a list created by a lunatic). The dataset can be used to test common-sense understanding in a non-literal way.
"""

url_train_small = "https://raw.githubusercontent.com/nightingal3/fig-qa/master/data/filtered/train_s.csv"
url_train_medium = "https://raw.githubusercontent.com/nightingal3/fig-qa/master/data/filtered/train.csv"
url_train_large = "https://raw.githubusercontent.com/nightingal3/fig-qa/master/data/filtered/train_xl.csv"
url_validation = "https://raw.githubusercontent.com/nightingal3/fig-qa/master/data/filtered/dev.csv"
url_test = f"{private_utils.PRIVATE_LOC}/fig_qa/test.csv"


class FigQAConfig(datalabs.BuilderConfig):
    """BuilderConfig for FB15K."""

    def __init__(self, **kwargs):
        """BuilderConfig for FigQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FigQAConfig, self).__init__(**kwargs)


class FigQA(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        FigQAConfig(
            name="small",
            version=datalabs.Version("1.0.0"),
            description="small training set",
        ),
        FigQAConfig(
            name="medium",
            version=datalabs.Version("1.0.0"),
            description="medium training set",
        ),
        FigQAConfig(
            name="large",
            version=datalabs.Version("1.0.0"),
            description="large training set",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medium"

    def _info(self):
        features_dataset = datalabs.Features()
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
            sample_level_schema = get_schema_of_sample_level_features()
            dict_feature_argument = get_feature_arguments(sample_level_schema, field="", feature_level="sample_level")
            additional_features = datalabs.Features(dict_feature_argument)
            features_sample.update(additional_features)

            dataset_level_schema = infer_schema_dataset_level(sample_level_schema)
            dict_feature_argument = get_feature_arguments(dataset_level_schema, field="avg", feature_level="dataset_level")
            features_dataset.update(datalabs.Features(dict_feature_argument))

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/nightingal3/fig-qa",
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

        split_gens = [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": validation_path}
            ),
        ]
        if private_utils.has_private_loc():
            test_path = dl_manager.download_and_extract(url_test)
            split_gens += [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    gen_kwargs={"filepath": test_path}
                )
            ]
        else:
            logger.warning('Skipping fig_qa test set because '
                           f'{private_utils.PRIVATE_LOC} is not set')
        return split_gens

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
                    additional_feature_info = get_features_sample_level(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    # print(additional_feature_info)
                    yield id_, raw_feature_info

