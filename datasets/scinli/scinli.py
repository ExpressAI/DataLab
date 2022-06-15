# coding=utf-8


import json
import os

import tqdm

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{sadat-caragea-2022-SciNLI,
        title = "SciNLI: A Corpus for Natural Language Inference on Scientific Text",
        author = "Sadat, Mobashir  and
          Caragea, Cornelia",
        booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        year = "2022",
        address = "Dublin, Ireland",
        publisher = "Association for Computational Linguistics",
    }
"""

_DESCRIPTION = """\
SCINLI is a large dataset for NLI that captures the formality in scientific text and contains 107,412
sentence pairs extracted from scholarly papers on NLP and computational linguistics.
"""


class MultiNli(datalabs.GeneratorBasedBuilder):
    """SCINLI: A Corpus for Natural Language Inference on Scientific Text"""

    def _info(self):

        features_dataset = {}
        features_sample = datalabs.Features(
            {
                "text1": datalabs.Value("string"),
                "text2": datalabs.Value("string"),
                "label": datalabs.features.ClassLabel(
                    names=["entailment", "neutral", "contrasting", "reasoning"]
                ),
            }
        )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            languages=["en"],
            homepage="https://github.com/msadat3/SciNLI",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.natural_language_inference)(
                    text1_column="text1", text2_column="text2", label_column="label"
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        downloaded_dir = dl_manager.download_and_extract(
            "https://datalab-hub.s3.amazonaws.com/scinli/SciNLI_dataset.zip"
        )
        mnli_path = os.path.join(downloaded_dir, "./")
        train_path = os.path.join(mnli_path, "train.jsonl")
        validation_path = os.path.join(mnli_path, "dev.jsonl")
        test_path = os.path.join(mnli_path, "test.jsonl")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate mnli examples"""

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                # for id_, row in enumerate(f):
                data = json.loads(row)
                if data["label"] == "-":
                    continue

                raw_feature_info = {
                    "text1": data["sentence1"],
                    "text2": data["sentence2"],
                    "label": data["label"],
                }

                yield id_, raw_feature_info
