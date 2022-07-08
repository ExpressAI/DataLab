import csv

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import importlib
import json
import os
import sys
from typing import Iterator

import datalabs
from datalabs import get_task, TaskType

# from .aggregate import fb15k_237_aggregating


_DESCRIPTION = """
WN18RR is a link prediction dataset created from WN18, which is a subset of WordNet. WN18 consists of 18 relations and 40,943 entities. However, many text triples are obtained by inverting triples from the training set. Thus the WN18RR dataset is created to ensure that the evaluation dataset does not have inverse relation test leakage. In summary, WN18RR dataset contains 93,003 triples with 40,943 entities and 11 relation types.
"""
_CITATION = """\
@inproceedings{shang2019end,
  title={End-to-end structure-aware convolutional networks for knowledge base completion},
  author={Shang, Chao and Tang, Yun and Huang, Jing and Bi, Jinbo and He, Xiaodong and Zhou, Bowen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={3060--3067},
  year={2019}
}
"""





class WN18RRConfig(datalabs.BuilderConfig):
    """BuilderConfig for FB15K."""

    def __init__(self, **kwargs):
        """BuilderConfig for FB15k237.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WN18RRConfig, self).__init__(**kwargs)


class WN18RRDataset(datalabs.GeneratorBasedBuilder):
    """FB15k237 Dataset."""

    _base_url = "https://datalab-hub.s3.amazonaws.com/kg/wn18rr/"
    BUILDER_CONFIGS = [
        WN18RRConfig(
            name="origin",
            version=datalabs.Version("1.0.0"),
            description="Files are in the original format (entities are represented as their freebase ID /m/XXXX rather than their English label)",
        )
    ]
    DEFAULT_CONFIG_NAME = "origin"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "head": datalabs.Value("string"),
                    "link": datalabs.Value("string"),
                    "tail": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://paperswithcode.com/dataset/wn18rr",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.kg_link_tail_prediction)(
                    head_column="head",
                    link_column="link",
                    tail_column="tail",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        if self.config.name == "origin":
            train_path = dl_manager.download_and_extract(self._base_url + "WN18RR-train.txt")
            val_path = dl_manager.download_and_extract(self._base_url + "WN18RR-valid.txt")
            test_path = dl_manager.download_and_extract(self._base_url + "WN18RR-test.txt")


        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": val_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate  examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                head, link, tail = row
                yield id_, {"head": head, "link": link, "tail": tail}
