# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
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
import csv
from email import header
import datalabs
from datalabs import get_task, TaskType


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb".   
"""

# You can copy an official description
_DESCRIPTION = """\
This reviews dataset contains more than 10,000 user reviews from a Chinese food delivery platform,
with about 4000 positive reviews and about 8000 negative ones. 
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb". 
"""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "N/A"

_HOMEPAGE = "https://github.com/SophonPlus/ChineseNlpCorpus"

_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/waimai/waimai_10k.csv"

class WAIMAI(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalab page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.text_classification)()],
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datalab.download.DownloadManager that can be used to download and extract URLs
        train_path = dl_manager.download_and_extract(_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path})
        ]

    def _generate_examples(self, filepath):
        """Generate WAIMAI examples."""

        # map the label into textual string
        textualize_label = {
            "1": "positive",
            "0": "negative"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for id_, row in enumerate(csv_reader):
                label, text = row
                if label == "0" or label == "1":
                    label = textualize_label.get(label)
                    yield id_, {"text": text, "label": label}
