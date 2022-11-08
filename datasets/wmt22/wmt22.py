import json
import csv
import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.logging import get_logger
import os

logger = get_logger(__name__)


_CITATION = """
"""


_DESCRIPTION = """\
For the second edition of the Large-Scale MT shared task, we aim to bring 
together the community on the topic of machine translation for a set of 24 
African languages. We do so by introducing a high quality benchmark, paired 
with a fair and rigorous evaluation procedure.
"""

_URL_flores101 = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
_URL_supplement = "https://dl.fbaipublicfiles.com/flores101/dataset/flores_wmt22_supplement.tar.gz"
_supplement = ["kin", "ssw", "tsn", "tso"]
_LANGUAGES_ENG = [
    "afr", "amh", "ful", "hau", "ibo", "kam", "lug", "luo", "nso", "nya", 
    "orm", "sna", "som", "ssw", "tsn", "tso", "umb", "xho", "yor", "zul"
]
_LANGUAGES_FRA = ["kin", "lin", "swh", "wol"]
pairs = [
    f"{src}-eng"
    for src in _LANGUAGES_ENG
] + [
    f"eng-{tar}"
    for tar in _LANGUAGES_ENG
] + [
    f"{src}-fra"
    for src in _LANGUAGES_FRA
] + [
    f"fra-{tar}"
    for tar in _LANGUAGES_FRA
]


class WMT22Config(datalabs.BuilderConfig):
    """BuilderConfig for MConala."""

    def __init__(self, pair, **kwargs):
        """BuilderConfig for WMT22.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.pair = pair
        super(WMT22Config, self).__init__(**kwargs)


class WMT22(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = WMT22Config
    BUILDER_CONFIGS = [
        WMT22Config(
            name=pair,
            pair=pair
        )
        for pair in pairs
    ]

    def _info(self):
        features = datalabs.Features(
            {
                "translation": datalabs.features.Translation(
                    languages=self.config.pair.split("-")
                )
            }
        )

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # Homepage of the dataset for documentation
            homepage="https://www.statmt.org/wmt22/large-scale-multilingual-translation-task.html",
            # datasets.features.FeatureConnectors
            features=features,
            supervised_keys=None,
            languages=self.config.pair.split("-"),
            task_templates=[
                get_task(TaskType.machine_translation)(
                    translation_column="translation",
                    lang_sub_columns=self.config.pair.split("-"),
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        source, target = self.config.pair.split("-")

        if (source in _supplement):
            src_dir = dl_manager.download_and_extract(_URL_supplement)
            src_dir = os.path.join(src_dir, "flores_wmt22_supplement")
        else: 
            src_dir = dl_manager.download_and_extract(_URL_flores101)
            src_dir = os.path.join(src_dir, "flores101_dataset")
        
        if (target in _supplement):
            tar_dir = dl_manager.download_and_extract(_URL_supplement)
            tar_dir = os.path.join(tar_dir, "flores_wmt22_supplement")
        else: 
            tar_dir = dl_manager.download_and_extract(_URL_flores101)
            tar_dir = os.path.join(tar_dir, "flores101_dataset")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_file": 
                        os.path.join(
                            src_dir,
                            "devtest",
                            f"{source}.devtest",
                        )
                    ,
                    "target_file":
                        os.path.join(
                            tar_dir,
                            "devtest",
                            f"{target}.devtest",
                        )
                    ,
                    "split": "test",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "source_file": 
                        os.path.join(
                            src_dir,
                            "dev",
                            f"{source}.dev",
                        )
                    ,
                    "target_file":
                        os.path.join(
                            tar_dir,
                            "dev",
                            f"{target}.dev",
                        )
                    ,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, source_file, target_file, split):
        """Yields examples."""
        id_ = 0
        source, target = self.config.pair.split("-")
        with open(source_file, encoding="utf8") as source_csv_file:
            with open(target_file, encoding="utf8") as target_csv_file:
                for source_row, target_row in zip(source_csv_file, target_csv_file):
                    source_row = source_row.strip()
                    target_row = target_row.strip()
                    yield id_, {"translation": {source: source_row, 
                                                target: target_row}}
                    id_ += 1
