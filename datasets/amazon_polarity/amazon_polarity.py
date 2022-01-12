# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The amazon polarity dataset for text classification."""


import csv

import datalabs
from datalabs.tasks import TextClassification

_CITATION = """\
@inproceedings{mcauley2013hidden,
  title={Hidden factors and hidden topics: understanding rating dimensions with review text},
  author={McAuley, Julian and Leskovec, Jure},
  booktitle={Proceedings of the 7th ACM conference on Recommender systems},
  pages={165--172},
  year={2013}
}
"""

_DESCRIPTION = """\
The Amazon reviews dataset consists of reviews from amazon.
The data span a period of 18 years, including ~35 million reviews up to March 2013.
Reviews include product and user information, ratings, and a plaintext review.
"""

_HOMEPAGE = "https://registry.opendata.aws/"

_LICENSE = "Apache License 2.0"

_URLs = {
    "amazon_polarity": "https://drive.google.com/u/0/uc?id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM&export=download",
}


class AmazonPolarityConfig(datalabs.BuilderConfig):
    """BuilderConfig for AmazonPolarity."""

    def __init__(self, **kwargs):
        """BuilderConfig for AmazonPolarity.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AmazonPolarityConfig, self).__init__(**kwargs)


class AmazonPolarity(datalabs.GeneratorBasedBuilder):
    """Amazon Polarity Classification Dataset."""

    VERSION = datalabs.Version("3.0.0")

    BUILDER_CONFIGS = [
        AmazonPolarityConfig(
            name="amazon_polarity", version=VERSION, description="Amazon Polarity Classification Dataset."
        ),
    ]

    def _info(self):
        features = datalabs.Features(
            {
                "label": datalabs.features.ClassLabel(
                    names=[
                        "negative",
                        "positive",
                    ]
                ),
                "title": datalabs.Value("string"),
                "text": datalabs.Value("string"),
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        archive = dl_manager.download(my_urls)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": "/".join(["amazon_review_polarity_csv", "train.csv"]),
                    "files": dl_manager.iter_archive(archive),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": "/".join(["amazon_review_polarity_csv", "test.csv"]),
                    "files": dl_manager.iter_archive(archive),
                },
            ),
        ]

    def _generate_examples(self, filepath, files):
        """Yields examples."""

        textualize_label = {"1":"negative",
                                 "2":"positive"}

        for path, f in files:
            if path == filepath:
                lines = (line.decode("utf-8") for line in f)
                data = csv.reader(lines, delimiter=",", quoting=csv.QUOTE_ALL)
                for id_, row in enumerate(data):
                    yield id_, {
                        "title": row[1],
                        "text": row[2],
                        "label": textualize_label[row[0]]
                    }
                break
