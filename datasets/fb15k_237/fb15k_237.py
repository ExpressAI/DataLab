import json
import os
import sys
import datalabs
import csv
from datalabs.tasks import KGLinkPrediction
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import importlib
from typing import Iterator

# from .aggregate import fb15k_237_aggregating




_DESCRIPTION = """
The FB15k dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 relationships. FB15K-237 is a variant of the original dataset where inverse relations are removed, since it was found that a large number of test triplets could be obtained by inverting triplets in the training set.
"""
_CITATION = """\
@article{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  journal={Advances in neural information processing systems},
  volume={26},
  year={2013}
}
"""


"""Get feature
from datalabs import load_dataset
from aggregate import *
dataset = load_dataset("fb15k_237",'readable')
statistics = next(dataset['train'].apply(get_statistics))

"""



class FB15k237Config(datalabs.BuilderConfig):
    """BuilderConfig for FB15K."""

    def __init__(self, **kwargs):
        """BuilderConfig for FB15k237.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FB15k237Config, self).__init__(**kwargs)


class FB15k237Dataset(datalabs.GeneratorBasedBuilder):
    """FB15k237 Dataset."""

    _base_url = "https://raw.githubusercontent.com/neulab/ExplainaBoard/main/data/datasets/fb15k_237/"
    BUILDER_CONFIGS = [
        FB15k237Config(
            name="origin",
            version=datalabs.Version("1.0.0"),
            description="Files are in the original format (entities are represented as their freebase ID /m/XXXX rather than their English label)",
        ),
        FB15k237Config(
            name="readable",
            version=datalabs.Version("1.0.0"),
            description="The readable files are included to help with understanding and visualization.",
        ),
    ]
    # DEFAULT_CONFIG_NAME = "document"

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
            homepage="https://paperswithcode.com/dataset/fb15k",
            citation=_CITATION,
            task_templates=[KGLinkPrediction(
                head_column = "head",
                link_column = "link",
                tail_column = "tail",
            ),
            ],
        )

    def _split_generators(self, dl_manager):

        if self.config.name == "origin":
            train_path = dl_manager.download_and_extract(self._base_url + "train.txt")
            val_path = dl_manager.download_and_extract(self._base_url + "valid.txt")
            test_path = dl_manager.download_and_extract(self._base_url  + "test.txt")
        else:
            train_path = dl_manager.download_and_extract(self._base_url + "train.readable.txt")
            val_path = dl_manager.download_and_extract(self._base_url + "valid.readable.txt")
            test_path = dl_manager.download_and_extract(self._base_url + "test.readable.txt")

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
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                head, link, tail = row
                yield id_, {"head": head, "link": link, "tail":tail}