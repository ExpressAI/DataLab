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
Name: WMT22 General MT Task
"""

_URL = "https://www.statmt.org/wmt22/mtdata/mtdata.recipes.wmt22-constrained.yml"
pairs_ordered = ["cs-en", "de-en", "ja-en", "ru-en", "zh-en", "fr-de", "hr-en", "liv-en", "uk-en", "uk-cs", "sah-ru"]
pairs_ref = ["ces-eng", "deu-eng", "jpn-eng", "rus-eng", "zho-eng", "fra-deu", "hrv-eng", "liv-eng", "ukr-eng", "ukr-ces", "sah-rus"]
pairs_reverse = ["en-cs", "en-de", "en-ja", "en-ru", "en-zh", "de-fr", "en-hr", "en-liv", "en-uk", "cs-uk", "ru-sah"]
pairs = pairs_ordered + pairs_reverse


class WMT22Config(datalabs.BuilderConfig):
    """BuilderConfig for WMT22."""

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
            homepage="https://www.statmt.org/wmt22/translation-task.html",
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

        if (self.config.pair == "hr-en"):
            return []

        source, target = self.config.pair.split("-")

        if (self.config.pair in pairs_ordered):
            first = source
            second = target
            ref = pairs_ref[pairs_ordered.index(f"{first}-{second}")]
            source_ref, target_ref = ref.split("-")
        else:
            first = target
            second = source
            ref = pairs_ref[pairs_ordered.index(f"{first}-{second}")]
            target_ref, source_ref = ref.split("-")

        cmd = "pip install mtdata==0.3.7"
        os.system(cmd)
        cmd = "wget https://www.statmt.org/wmt22/mtdata/mtdata.recipes.wmt22-unconstrained.yml"
        os.system(cmd)
        cmd = "mtdata get-recipe -ri wmt22-{}{} -o wmt22-{}{}".format(first, second, first, second)
        os.system(cmd)

        dir = os.path.join(os.getcwd(), f"wmt22-{first}{second}")
        test_dir = os.path.join(dir, "tests")
        test_source_file = ""
        test_target_file = ""
        for filename in os.listdir(test_dir):
            if filename.endswith(f"{source_ref}-{target_ref}.{source_ref}"):
                test_source_file = os.path.join(test_dir, filename)
            if filename.endswith(f"{source_ref}-{target_ref}.{target_ref}"):
                test_target_file = os.path.join(test_dir, filename)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "source_file": os.path.join(dir, f"train.{source_ref}"),
                    "target_file": os.path.join(dir, f"train.{target_ref}"),
                    "split": "train",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_file": test_source_file,
                    "target_file": test_target_file,
                    "split": "test",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "source_file": os.path.join(dir, f"dev.{source_ref}"),
                    "target_file": os.path.join(dir, f"dev.{target_ref}"),
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
