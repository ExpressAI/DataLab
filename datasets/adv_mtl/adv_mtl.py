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
import datalabs
from datalabs.tasks import TextClassification
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{liu2017adversarial,
title={Adversarial Multi-task Learning for Text Classification},
author={Liu, Pengfei and Qiu, Xipeng and Huang, Xuanjing},
journal={arXiv preprint arXiv:1704.05742},
year={2017}
}
"""

# You can copy an official description
_DESCRIPTION = """\
This datalab is used in the paper of adversarial multi-task learning in text classification including 16 different fields.
"""

_HOMEPAGE = "http://pfliu.com/paper/adv-mtl.html"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "N/A"

# The HuggingFace dataset library don't host the datalab but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = "https://raw.githubusercontent.com/ShiinaHiiragi/multi-task-dataset/master/{}.task.{}"

class AdvMtl(datalabs.GeneratorBasedBuilder):
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
        features = datalabs.Features(
            {
                "text": datalabs.Value("string"),
                "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
            }
        )
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalab page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datalab.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        result = []
        data_types = ["train", "test"]
        fields = [
            "apparel",
            "baby",
            "books",
            "camera_photo",
            "dvd",
            "electronics",
            "health_personal_care",
            "imdb",
            "kitchen_housewares",
            "magazines",
            "MR",
            "music",
            "software",
            "sports_outdoors",
            "toys_games",
            "video"
        ]

        for data_field in fields:
            for data_type in data_types:
                split_name = data_field + "_" + data_type
                result.append(datalabs.SplitGenerator(
                    name=split_name,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": dl_manager.download_and_extract(
                            _URLs.format(data_field, data_type)
                        ),
                        "split": data_type,
                    },
                ))

        return result

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        with open(filepath, "rb") as f:
            for id_, row in enumerate(f):
                row = row.decode("utf-8", "ignore")
                datas = row.split("\t")
                text = datas[1]
                label = "positive" if datas[0] == "1" else "negative"
                yield id_, {
                    "text": text,
                    "label": label,
                }
