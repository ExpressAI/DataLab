import json
import csv
import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.logging import get_logger

logger = get_logger(__name__)


_CITATION = """
"""


_DESCRIPTION = """\
The recurring translation task of the WMT workshops focuses on news text.
"""

_URL = "https://datalab-hub.s3.amazonaws.com/wmt20_test/{}/test.tsv"
_LANGUAGES = [
    "csen",
    "defr", #
    "encs",
    "enja",
    "enpl",
    "enru", #
    "enta",
    "enzh", #
    "frde", #
    "iuen",
    "kmen",
    "plen", #
    "psen",
    "ruen", #
    "taen",
]




class WMT20Config(datalabs.BuilderConfig):
    """BuilderConfig for MConala."""

    def __init__(self, **kwargs):
        """BuilderConfig for WMT20.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WMT20Config, self).__init__(**kwargs)


class WMT20(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name=lang,
        )
        for lang in list(_LANGUAGES)
    ]

    def _info(self):
        features_sample = datalabs.Features(
            {
                "translation": {
                    self.config.name[0:2]: datalabs.Value("string"),
                    self.config.name[2:4]: datalabs.Value("string"),
                },
            }
        )

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # Homepage of the dataset for documentation
            homepage="https://www.statmt.org/wmt20/translation-task.html"
            # datasets.features.FeatureConnectors
            features=features_sample,
            supervised_keys=None,
            languages=[self.config.name[0:2],
                       self.config.name[2:4]],
            task_templates=[
                get_task(TaskType.machine_translation)(
                    translation_column="translation",
                    lang_sub_columns=[self.config.name[0:2],
                                      self.config.name[2:4]],
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        lang = self.config.name
        dataset_path = dl_manager.download_and_extract(_URL.format(lang))

        return [
            datalabs.SplitGenerator(
                name="test",
                gen_kwargs={"filepath": f"{dataset_path}"},
            ),
        ]



    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf8") as csv_file:
            # csv_reader = csv.reader(csv_file, delimiter="\t")
            # for line in csv_file:
            for id_, line in enumerate(csv_file):
                row = line.split("\t")
                if len(row) == 1:
                    src, ref = row[0], ""
                else:
                    src, ref = row[0], row[1]


                yield id_, {"translation": {self.config.name[0:2]: src,
                                        self.config.name[2:4]: ref}}
